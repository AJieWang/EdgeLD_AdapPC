import sys

sys.path.append("../..")
sys.path.append("..")

from node_test.network_op import Network_init_datanode, Network_init_namenode
from node_test.network_op import AllReduceDataNode
from node_test.num_set_up import Num_set_up
from VGG.mydefine_VGG13 import VGG_model
from VGG.tensor_op import SlicedVGGExecutor
from VGG.tensor_op import VGG13_POOL_LAYERS, VGG13_CONV_LAYERS
import torch.nn as nn
import torch, time
import threading
import queue

num_set_up = Num_set_up()
namenode_num = num_set_up.get_namenode_num()
datanode_num = num_set_up.get_datanode_num()
datanode_name = 0

inference_model = VGG_model()
maxpool_layer = inference_model.get_maxpool_layer()
conv_length = inference_model.get_conv_length()
total_length = inference_model.get_total_length()
c_out_list = inference_model.get_c_out()

WARM_UP_ROUNDS = 3
VALID_ROUNDS = 2
TOTAL_ROUNDS = WARM_UP_ROUNDS + VALID_ROUNDS


class AllReduceDataNodeExecutor:
    """
    AllReduce模式下子节点的切片执行器

    使用 SlicedVGGExecutor 执行切片卷积
    """

    def __init__(self, full_model):
        self.model = full_model
        self.executor = SlicedVGGExecutor(full_model, model_type='VGG13')

    def execute_sliced_layers(self, input_tensor, start_layer, end_layer, start_filter, end_filter):
        """
        执行连续多层切片卷积

        参数:
            input_tensor: [N, C_in, H, W] 输入特征图
            start_layer: 起始层ID
            end_layer: 结束层ID
            start_filter: 起始通道索引
            end_filter: 结束通道索引

        返回:
            output_tensor: 计算结果
        """
        current = input_tensor
        pool_layers = VGG13_POOL_LAYERS
        conv_layers = VGG13_CONV_LAYERS

        for layer_id in range(start_layer, end_layer + 1):
            if layer_id in pool_layers:
                current = torch.nn.functional.max_pool2d(current, kernel_size=2, stride=2)
                print(f"  Layer {layer_id}: MaxPool2d, output {current.size()}")

            elif layer_id in conv_layers:
                if layer_id == start_layer:
                    s_f, e_f = start_filter, end_filter
                else:
                    s_f, e_f = 0, current.size(1)

                current = self.executor.execute_sliced_conv(current, layer_id, s_f, e_f)
                print(f"  Layer {layer_id}: Conv2d [{s_f}:{e_f}], output {current.size()}")

        return current

    def execute_sliced_fc(self, input_tensor, start_layer, end_layer, start_neuron, end_neuron):
        """
        执行切片全连接层

        参数:
            input_tensor: [N, in_features] 输入
            start_layer: 起始层ID
            end_layer: 结束层ID
            start_neuron: 起始神经元索引
            end_neuron: 结束神经元索引

        返回:
            output_tensor: 计算结果
        """
        current = input_tensor

        if start_layer == 16:
            current = torch.nn.functional.adaptive_avg_pool2d(current, (7, 7))
            current = torch.flatten(current, 1)

        for layer_id in range(start_layer, end_layer + 1):
            if layer_id == start_layer:
                s_n, e_n = start_neuron, end_neuron
            else:
                s_n, e_n = 0, current.size(1)

            current = self.executor.execute_sliced_fc(current, layer_id, s_n, e_n)
            print(f"  Layer {layer_id}: Linear [{s_n}:{e_n}], output {current.size()}")

        return current


def datanode_persistent():
    print(f"\n===== DataNode {datanode_name} AllReduce Filter Splitting 启动 =====")

    datanode = Network_init_datanode(
        namenode_num=namenode_num,
        datanode_num=datanode_num,
        datanode_name=datanode_name
    )

    allreduce_datanode = AllReduceDataNode(datanode)
    executor = AllReduceDataNodeExecutor(inference_model)

    print(f"DataNode {datanode_name} 已建立连接")

    round_idx = 0

    while round_idx < TOTAL_ROUNDS:
        round_idx += 1

        print(f"\n----- DataNode {datanode_name} 第 {round_idx} 轮推理开始 -----")

        try:
            inference_count = 0

            while True:
                broadcast_data = allreduce_datanode.receive_initial_broadcast()

                start_layer = broadcast_data['start_layer']
                end_layer = broadcast_data['end_layer']
                filter_boundaries = broadcast_data['filter_boundaries']
                input_tensor = broadcast_data['input_tensor']

                node_start_filter = filter_boundaries[datanode_name]
                node_end_filter = filter_boundaries[datanode_name + 1]

                print(f"\n[Node {datanode_name}] 收到任务:")
                print(f"  Layers: {start_layer}-{end_layer}")
                print(f"  Filter: [{node_start_filter}:{node_end_filter}]")
                print(f"  Input: {input_tensor.size()}")

                if start_layer == 0 and end_layer == 0:
                    print(f"[Node {datanode_name}] 收到结束信号")
                    break

                # ========== 新增：判断通道数/神经元数是否为0 ==========
                if node_start_filter >= node_end_filter:
                    print(f"[Node {datanode_name}] 分到的通道/神经元数为0，跳过计算")
                    # 发送空标识（需与NameNode端逻辑匹配，也可发送空tensor）
                    allreduce_datanode.send_slice_to_master(end_layer, None)
                    # 接收合并结果，维持流程完整性
                    merged_data = allreduce_datanode.receive_merged_tensor()
                    next_layer_id = merged_data['next_layer_id']
                    print(f"[Node {datanode_name}] 收到合并结果, next_layer={next_layer_id}")
                    if next_layer_id == 0:
                        print(f"[Node {datanode_name}] 第 {round_idx} 轮推理完成")
                        break
                    continue  # 跳过后续计算逻辑
                # =====================================================

                compute_start = time.time()

                if start_layer > conv_length:
                    output_tensor = executor.execute_sliced_fc(
                        input_tensor, start_layer, end_layer,
                        node_start_filter, node_end_filter
                    )
                else:
                    output_tensor = executor.execute_sliced_layers(
                        input_tensor, start_layer, end_layer,
                        node_start_filter, node_end_filter
                    )

                compute_time = time.time() - compute_start
                print(f"[Node {datanode_name}] 计算完成: {output_tensor.size()}, 耗时: {compute_time:.3f}s")

                allreduce_datanode.send_slice_to_master(end_layer, output_tensor)

                # ==================== [新增逻辑开始] ====================
                # 如果当前完成的是整个模型的最后一层（比如第 18 层），则跳过接收合并结果，直接结束本轮循环
                if end_layer == total_length:
                    print(f"[Node {datanode_name}] 最后一层 (Layer {end_layer}) 计算完毕，本轮结束，准备迎接下一轮任务...")
                    break 
                    # 这里使用 break 还是 continue 取决于你子节点的 while 循环是怎么嵌套的。
                    # 核心目的：跳出对 receive_merged_tensor 的调用，回到头部去调用 receive_initial_broadcast() 等待下一轮。
                # ==================== [新增逻辑结束] ====================

                merged_data = allreduce_datanode.receive_merged_tensor()
                next_layer_id = merged_data['next_layer_id']

                print(f"[Node {datanode_name}] 收到合并结果, next_layer={next_layer_id}")

                if next_layer_id == 0:
                    print(f"[Node {datanode_name}] 第 {round_idx} 轮推理完成")
                    break

                inference_count += 1

        except (BrokenPipeError, ConnectionResetError):
            print(f"第 {round_idx} 轮: 连接被 NameNode 关闭，等待新连接...")
            datanode.close()
            time.sleep(1)
            datanode = Network_init_datanode(
                namenode_num=namenode_num,
                datanode_num=datanode_num,
                datanode_name=datanode_name
            )
            allreduce_datanode = AllReduceDataNode(datanode)
            continue

        except Exception as e:
            print(f"第 {round_idx} 轮发生错误: {e}")
            import traceback
            traceback.print_exc()
            break

    allreduce_datanode.close()
    datanode.close()
    print(f"\n关闭 DataNode {datanode_name} 的Socket连接")
    print(f"DataNode {datanode_name} AllReduce 模式已关闭")


if __name__ == "__main__":
    datanode_persistent()

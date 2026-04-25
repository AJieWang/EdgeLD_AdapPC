import sys

sys.path.append("../..")
sys.path.append("..")

from node_test.network_op import Network_init_datanode, Network_init_namenode, datanode_ip
from node_test.network_op import EdgeDataNode, EdgeP2PCommunicator
from node_test.num_set_up import Num_set_up, MODEL_TYPE, INPUT_WIDTH
from node_test.num_set_up import pool_layers_set, conv_layers, c_out_list, conv_length, total_length, inference_model
from VGG.tensor_op import SlicedVGGExecutor
import torch.nn as nn
import torch, time
import threading
import queue

num_set_up = Num_set_up()
namenode_num = num_set_up.get_namenode_num()
datanode_num = num_set_up.get_datanode_num()
datanode_name = 1

WARM_UP_ROUNDS = 3
VALID_ROUNDS = 2
TOTAL_ROUNDS = WARM_UP_ROUNDS + VALID_ROUNDS


# class EdgeDataNodeExecutor:
#     """
#     边缘协作模式下子节点的执行器

#     接收plan（层范围），执行指定层，最后一层按DDPG边界返回部分通道
#     """

#     def __init__(self, full_model):
#         self.model = full_model
#         self.executor = SlicedVGGExecutor(full_model, model_type='VGG13')

#     def execute_layers_with_output_slice(self, input_tensor, start_layer, end_layer, boundaries):
#         """
#         执行连续多层（卷积+池化+FC），最后一层按DDPG边界分割输出

#         参数:
#             input_tensor: 输入特征图
#             start_layer: 起始层ID
#             end_layer: 结束层ID
#             boundaries: 通道/神经元分割边界，如 [0, 32, 64]

#         返回:
#             output_tensor: 只返回该节点负责的通道/神经元部分
#         """
#         current = input_tensor
#         pool_layers = VGG13_POOL_LAYERS
#         conv_layers = VGG13_CONV_LAYERS

#         for layer_id in range(start_layer, end_layer + 1):
#             if layer_id in pool_layers:
#                 current = torch.nn.functional.max_pool2d(current, kernel_size=2, stride=2)
#                 print(f"  Layer {layer_id}: MaxPool2d, output {current.size()}")

#             elif layer_id in conv_layers:
#                 if layer_id == end_layer:
#                     start_filter = boundaries[datanode_name]
#                     end_filter = boundaries[datanode_name + 1]
#                     current = self.executor.execute_sliced_conv(current, layer_id, start_filter, end_filter)
#                     print(f"  Layer {layer_id}: Conv2d [{start_filter}:{end_filter}], output {current.size()}")
#                 else:
#                     c_out = c_out_list[layer_id - 1]
#                     current = self.executor.execute_sliced_conv(current, layer_id, 0, c_out)
#                     print(f"  Layer {layer_id}: Conv2d [0:{c_out}], output {current.size()}")

#             elif layer_id > 15:
#                 if layer_id == end_layer:
#                     start_neuron = boundaries[datanode_name]
#                     end_neuron = boundaries[datanode_name + 1]
#                     current = self.executor.execute_sliced_fc(current, layer_id, start_neuron, end_neuron)
#                     print(f"  Layer {layer_id}: Linear [{start_neuron}:{end_neuron}], output {current.size()}")
#                 else:
#                     if layer_id == 16:
#                         current = torch.nn.functional.adaptive_avg_pool2d(current, (7, 7))
#                         current = torch.flatten(current, 1)
#                     c_out = c_out_list[layer_id - 1]
#                     current = self.executor.execute_sliced_fc(current, layer_id, 0, c_out)
#                     print(f"  Layer {layer_id}: Linear [0:{c_out}], output {current.size()}")

#         return current

class EdgeDataNodeExecutor:
    def __init__(self, full_model, p2p_communicator=None):
        self.model = full_model
        self.executor = SlicedVGGExecutor(full_model, model_type=MODEL_TYPE)
        self.p2p_communicator = p2p_communicator
        self.pool_layers_set = set(pool_layers_set)
        self.conv_layers_set = set(conv_layers)

    def execute_layers_with_output_slice(self, input_tensor, start_layer, end_layer, boundaries, c_out_list):
        current = input_tensor
        node_start_ratio = boundaries[datanode_name]
        node_end_ratio = boundaries[datanode_name + 1]

        for layer_id in range(start_layer, end_layer + 1):
            if layer_id in self.pool_layers_set:
                current = torch.nn.functional.max_pool2d(current, kernel_size=2, stride=2)
                print(f"  Layer {layer_id}: MaxPool2d, output {current.size()}")

            elif layer_id in self.conv_layers_set:
                c_out = c_out_list[layer_id - 1]
                start_filter = int(c_out * node_start_ratio)
                end_filter = int(c_out * node_end_ratio)

                partial_output = self.executor.execute_sliced_conv(current, layer_id, start_filter, end_filter)
                print(f"  Layer {layer_id}: Conv2d [{start_filter}:{end_filter}] / {c_out}, partial_output {partial_output.size()}")

                if layer_id != end_layer:
                    merge_start_time = time.time()
                    current = self.p2p_communicator.all_gather_tensor(partial_output)
                    print(f"  --> Edge P2P Merged: Layer {layer_id} output {current.size()}, cost: {time.time()-merge_start_time:.3f}s")
                else:
                    current = partial_output

            elif layer_id > 15:
                c_out = c_out_list[layer_id - 1]
                start_neuron = int(c_out * node_start_ratio)
                end_neuron = int(c_out * node_end_ratio)

                if (layer_id == 16 or layer_id == 19) and current.dim() > 2:
                    current = torch.nn.functional.adaptive_avg_pool2d(current, (7, 7))
                    current = torch.flatten(current, 1)

                partial_output = self.executor.execute_sliced_fc(current, layer_id, start_neuron, end_neuron)
                print(f"  Layer {layer_id}: Linear [{start_neuron}:{end_neuron}] / {c_out}, partial_output {partial_output.size()}")

                if layer_id != end_layer:
                    merge_start_time = time.time()
                    current = self.p2p_communicator.all_gather_tensor(partial_output)
                    print(f"  --> Edge P2P Merged: Layer {layer_id} output {current.size()}, cost: {time.time()-merge_start_time:.3f}s")
                else:
                    current = partial_output

        return current

# def datanode_persistent():
#     print(f"\n===== DataNode {datanode_name} 边缘协作模式启动 =====")

#     datanode = Network_init_datanode(
#         namenode_num=namenode_num,
#         datanode_num=datanode_num,
#         datanode_name=datanode_name
#     )

#     edge_datanode = EdgeDataNode(datanode)
#     executor = EdgeDataNodeExecutor(inference_model)

#     print(f"DataNode {datanode_name} 已建立连接")

def datanode_persistent():
    print(f"\n===== DataNode {datanode_name} 边缘协作模式启动 =====")

    datanode = Network_init_datanode(
        namenode_num=namenode_num,
        datanode_num=datanode_num,
        datanode_name=datanode_name
    )

    # 初始化边边通信网络 (P2P)
    p2p_communicator = EdgeP2PCommunicator(
        datanode_name=datanode_name,
        total_datanodes=datanode_num,
        ip_list=datanode_ip
    )
    p2p_communicator.initialize_p2p_network()

    edge_datanode = EdgeDataNode(datanode)
    # 将 P2P 通信器传入执行器
    executor = EdgeDataNodeExecutor(inference_model, p2p_communicator)

    print(f"DataNode {datanode_name} 已建立所有连接 (NameNode & DataNodes)")

    round_idx = 0

    while round_idx < TOTAL_ROUNDS:
        round_idx += 1

        print(f"\n----- DataNode {datanode_name} 第 {round_idx} 轮推理开始 -----")

        try:
            broadcast_data = edge_datanode.receive_initial_broadcast()

            start_layer = broadcast_data['start_layer']
            end_layer = broadcast_data['end_layer']
            boundaries = broadcast_data['filter_boundaries']
            c_out_list = broadcast_data['c_out_list']
            input_tensor = broadcast_data['input_tensor']

            print(f"\n[Node {datanode_name}] 收到任务:")
            print(f"  Layers: {start_layer}-{end_layer}")
            print(f"  通道边界: {boundaries}")
            print(f"  c_out_list: {c_out_list}")
            print(f"  Input: {input_tensor.size()}")

            if start_layer == 0 and end_layer == 0:
                print(f"[Node {datanode_name}] 收到结束信号")
                break

            node_start_ratio = boundaries[datanode_name]
            node_end_ratio = boundaries[datanode_name + 1]

            if node_start_ratio >= node_end_ratio:
                print(f"[Node {datanode_name}] 分到的比例为空，发送空tensor")
                empty_slice = torch.zeros(1, 0, 1, 1) if start_layer <= 15 else torch.zeros(1, 0)
                edge_datanode.send_slice_to_master(end_layer, empty_slice)
                print(f"[Node {datanode_name}] 第 {round_idx} 轮推理完成（空载）")
                continue

            compute_start = time.time()

            output_tensor = executor.execute_layers_with_output_slice(
                input_tensor, start_layer, end_layer, boundaries, c_out_list
            )

            compute_time = time.time() - compute_start
            print(f"[Node {datanode_name}] 计算完成: {output_tensor.size()}, 耗时: {compute_time:.3f}s")

            edge_datanode.send_slice_to_master(end_layer, output_tensor)

            print(f"[Node {datanode_name}] 第 {round_idx} 轮推理完成")

        except (BrokenPipeError, ConnectionResetError):
            print(f"第 {round_idx} 轮: 连接被 NameNode 关闭，等待新连接...")
            datanode.close()
            time.sleep(1)
            datanode = Network_init_datanode(
                namenode_num=namenode_num,
                datanode_num=datanode_num,
                datanode_name=datanode_name
            )
            edge_datanode = EdgeDataNode(datanode)
            continue

        except Exception as e:
            print(f"第 {round_idx} 轮发生错误: {e}")
            import traceback
            traceback.print_exc()
            break

    p2p_communicator.close_all()
    edge_datanode.close()
    datanode.close()
    print(f"\n关闭 DataNode {datanode_name} 的Socket连接")
    print(f"DataNode {datanode_name} 边缘协作模式已关闭")


if __name__ == "__main__":
    datanode_persistent()
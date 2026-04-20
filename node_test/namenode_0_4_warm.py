import sys

sys.path.append("../..")
sys.path.append("..")

from node_test.network_op import Network_init_datanode, Network_init_namenode
from node_test.network_op import AllReduceNameNode, AllReduceDataNode
from node_test.num_set_up import Num_set_up
from node_test.scheduler import PipelineScheduler, StageBoundary, AllReduceScheduler
from node_test.ilp_solver import OffloadingPartitioner, calculate_layer_flops_vgg13, calculate_output_size_vgg13
from node_test.adaptive_partitioner import AdaptivePartitioner
import torch
import threading
import time
import numpy as np
import torch.nn as nn
from VGG.mydefine_VGG13 import VGG_model
from VGG.tensor_op import tensor_divide_by_computing_network_and_fill
from VGG.tensor_op import merge_total_tensor, merge_filter_tensor
from VGG.tensor_op import dispatch_featuremap_to_nodes, tensor_divide_by_filter_ratios
from VGG.tensor_op import VGG13_POOL_LAYERS, VGG13_CONV_LAYERS
from network_and_computing.network_and_computing_record import Network_And_Computing

WARM_UP_ROUNDS = 3
VALID_ROUNDS = 2
TOTAL_ROUNDS = WARM_UP_ROUNDS + VALID_ROUNDS

num_set_up = Num_set_up()
namenode_num = num_set_up.get_namenode_num()
datanode_num = num_set_up.get_datanode_num()

network_and_computing = Network_And_Computing()
computing_power = network_and_computing.get_computing_power(datanode_num)
network_state = network_and_computing.get_network_state(datanode_num)
computing_a = network_and_computing.get_computing_a(datanode_num)
computing_b = network_and_computing.get_computing_b(datanode_num)

device_power = (6.24e-11, 1.97e-2)
edge_cluster_power = (3.12e-11, 1.0e-2)
bandwidth = 100e6

inference_model = VGG_model()
conv_length = inference_model.get_conv_length()
total_length = inference_model.get_total_length()
c_out_list = inference_model.get_c_out()
maxpool_layer = inference_model.get_maxpool_layer()
width = 224

transfer_time = []
thread_start_time = []
thread_end_time = []
thread_time = []


class AdapCPNameNode:
    def __init__(self, namenode):
        self.namenode = namenode
        self.allreduce_namenode = AllReduceNameNode(namenode)
        self.scheduler = PipelineScheduler(datanode_num, total_length)
        self.allreduce_scheduler = AllReduceScheduler(datanode_num)
        self.offloading_partitioner = OffloadingPartitioner() # ILP 2 split_layer; 3 timing
        self.ddpg_partitioner = AdaptivePartitioner(datanode_num, state_dim = 12, use_dirichlet=True)
        self.current_split_layer = None
        self.current_ratios = None
        self.current_boundaries = None
        self.lock = threading.Lock()

        self._init_offloading_partitioner()

    def _init_offloading_partitioner(self):
        """初始化OffloadingPartitioner，添加VGG13各层信息"""
        maxpool_layers = [3, 6, 9, 12, 15]

        for layer_id in range(1, 16):
            is_maxpool = layer_id in maxpool_layers
            flops = calculate_layer_flops_vgg13(layer_id)
            output_size = calculate_output_size_vgg13(layer_id)

            if layer_id <= 3:
                c_in, c_out = 3, 64
                h, w = 224, 224
            elif layer_id <= 6:
                c_in, c_out = 64, 128
                h, w = 112, 112
            elif layer_id <= 9:
                c_in, c_out = 128, 256
                h, w = 56, 56
            elif layer_id <= 12:
                c_in, c_out = 256, 512
                h, w = 28, 28
            else:
                c_in, c_out = 512, 512
                h, w = 14, 14

            self.offloading_partitioner.add_layer(
                layer_id, flops, output_size, is_maxpool, False, c_in, c_out, h, w
            )

    def compute_offloading_point(self): # ILP 1 split_layer
        """第一步：ILP决策Device vs Edge Cluster的卸载点"""
        self.current_split_layer = self.offloading_partitioner.solve_offloading_point(
            device_power, edge_cluster_power, bandwidth, min_local_layers=3
        )
        return self.current_split_layer

    def compute_ddpg_ratios(self, computing_power_list, network_bw_list, current_load=None, epsilon=0.0):
        """第二步：DDPG决策各边缘节点的并行比例"""
        self.current_ratios = self.ddpg_partitioner.get_partition_ratios(
            computing_power_list, network_bw_list, current_load, epsilon
        )
        return self.current_ratios

    def compute_filter_boundaries(self, c_out):
        """根据DDPG比例计算Filter分割边界（3节点 → 4个边界）"""
        if self.current_ratios is None:
            raise ValueError("DDPG ratios not computed")

        # --- 修复代码：确保每个比例至少能分到1个通道 ---
        min_ratio = 1.0 / c_out
        adjusted_ratios = np.maximum(self.current_ratios, min_ratio)
        adjusted_ratios = adjusted_ratios / adjusted_ratios.sum() # 重新归一化
        # --------------------------------------------

        boundaries = [0]
        # 修复1：遍历所有比例，不要切片砍掉！
        for r in self.current_ratios:
            next_bound = int(boundaries[-1] + c_out * r)
            boundaries.append(next_bound)
        
        # 修复2：最后强制等于总通道数，保证不越界（不覆盖中间值）
        boundaries[-1] = c_out
        
        self.current_boundaries = boundaries
        return boundaries

    def get_local_layers(self):
        """获取本地计算的层范围"""
        if self.current_split_layer is None:
            return 1, 0
        return 1, self.current_split_layer

    def get_edge_layers(self):
        """获取边缘计算的层范围"""
        if self.current_split_layer is None:
            return 1, 0
        return self.current_split_layer + 1, total_length

    def run_local_inference(self, input_tensor):
        """执行本地部分推理"""
        local_start, local_end = self.get_local_layers()
        if local_start > local_end:
            return input_tensor

        print(f"\n===== 本地推理: Layers {local_start}-{local_end} =====")
        start_time = time.time()
        output = inference_model(input_tensor, local_start, local_end)
        end_time = time.time()
        print(f"本地推理耗时: {end_time - start_time:.3f}s, 输出尺寸: {output.size()}")
        return output

    def run_allreduce_inference(self, input_tensor):
        """
        执行边缘AllReduce推理（Filter Splitting模式）

        流程：
        1. 广播输入给所有DataNode
        2. 收集各节点切片
        3. 合并后广播给所有节点
        4. 重复直到所有层完成
        """
        edge_start, edge_end = self.get_edge_layers()
        if edge_start > edge_end:
            return input_tensor

        print(f"\n===== AllReduce边缘推理: Layers {edge_start}-{edge_end} =====")

        # -------------------动态计算 boundaries 的代码---------------------------
        current_input = input_tensor
        layer_groups = self._get_layer_groups(edge_start, edge_end)

        print(f"层分组: {layer_groups}")

        for group_idx, (start_l, end_l) in enumerate(layer_groups):
            print(f"\n--- Layer Group {group_idx + 1}: Layers {start_l}-{end_l} ---")

            # 【新增修复】: 动态获取当前组的输出通道数 (layer_id 从 1 开始，所以索引是 -1)
            current_group_c_out = c_out_list[start_l - 1]
            
            # 重新计算本组的分割边界
            boundaries = self.compute_filter_boundaries(current_group_c_out)
            print(f"当前组 (目标通道数 {current_group_c_out}) Filter 分割边界: {boundaries}")

            # 广播时带上正确的 boundaries
            self.allreduce_namenode.broadcast_input_to_all(
                current_input, start_l, end_l, boundaries
            )
            # -------------------动态计算 boundaries 的代码---------------------------

            slices = {}
            for _ in range(datanode_num):
                layer_id, node_id, tensor_slice = self.allreduce_namenode.collect_slice_from_datanode(
                    _, transfer_time
                )
                slices[node_id] = tensor_slice

            # 过滤掉通道数 (dim=1) 为 0 的无效 dummy tensor (即尺寸为 [1, 0, 1, 1] 的切片)
            sorted_slices = []
            for i in range(datanode_num):
                t = slices[i]
                if t.size(1) > 0:  # 只有包含实际数据的切片才参与拼接
                    sorted_slices.append(t)
            
            # 安全拼接
            if len(sorted_slices) > 0:
                merged = torch.cat(sorted_slices, dim=1)
            else:
                # 极端情况防崩溃：如果所有节点都没数据（理论上不会发生）
                merged = torch.zeros(1, 0, 1, 1)

            print(f"合并后尺寸: {merged.size()}")
            # --------------------------------------------------------------------------

            next_layer_id = layer_groups[group_idx + 1][0] if group_idx + 1 < len(layer_groups) else 0

            if next_layer_id != 0:
                self.allreduce_namenode.broadcast_merged_to_all(merged, next_layer_id)

            current_input = merged

        return current_input

    def _get_layer_groups(self, start_layer, end_layer):
        """将连续的层分成多个组（按池化层和全连接层分割）"""
        groups = []
        pool_layers = VGG13_POOL_LAYERS

        current_start = start_layer
        for layer_id in range(start_layer, end_layer + 1):
            # 新增: layer_id > 15 (即全连接层) 强制作为切分点，逐层 AllReduce 同步
            if (layer_id in pool_layers or layer_id > 15) and layer_id < end_layer:
                groups.append((current_start, layer_id))
                current_start = layer_id + 1

        if current_start <= end_layer:
            groups.append((current_start, end_layer))

        return groups

    def compute_fc_layers(self, input_tensor):
        """计算全连接层"""
        print("\n===== Computing FC Layers =====")
        for layer_id in range(conv_length + 1, total_length + 1):
            fc_start = time.time()
            input_tensor = inference_model(input_tensor, layer_id, layer_id)
            fc_end = time.time()
            print(f'FC Layer {layer_id} 耗时: {fc_end - fc_start:.3f}s')
        return input_tensor

    def record_inference_result(self, latency):
        """记录推理结果用于DDPG训练"""
        self.ddpg_partitioner.record_experience(latency)


def run_adapcp_inference(namenode, adapcp_namenode, round_idx):
    """
    AdapCP完整推理流程

    Step 1: ILP决策卸载点 (Device vs Edge Cluster)
    Step 2: 本地执行前段 (layer 1 到 split_layer)
    Step 3: DDPG决策并行比例 (Filter Splitting)
    Step 4: AllReduce边缘并行计算
    Step 5: 执行FC层
    """
    global transfer_time, thread_start_time, thread_end_time, thread_time

    print(f"\n{'='*60}")
    print(f"第 {round_idx} 轮 AdapCP 推理 (AllReduce Filter Splitting)")
    print(f"{'='*60}")

    transfer_time = []
    thread_start_time = [0] * datanode_num
    thread_end_time = [0] * datanode_num
    thread_time = [[] for _ in range(datanode_num)]

    input_tensor = torch.rand(1, 3, width, width)
    round_start_time = time.time()

    # --------------------------------层分割算法 ILP---------------------------------------
    split_layer = adapcp_namenode.compute_offloading_point()
    plan = adapcp_namenode.offloading_partitioner.get_offloading_plan(split_layer)
    timing = adapcp_namenode.offloading_partitioner.estimate_pipeline_times(
        split_layer, device_power, edge_cluster_power, bandwidth
    )
    print(f"\n[ILP决策] {plan['description']}")
    print(f"[时间估算] 本地: {timing['local_time']:.3f}s, 传输: {timing['transfer_time']:.3f}s, 边缘: {timing['edge_time']:.3f}s")
    # --------------------------------层分割算法 ILP---------------------------------------

    # --------------------------------层内分割算法 DDPG---------------------------------------
    ratios = adapcp_namenode.compute_ddpg_ratios(
        [(c, 0.01) for c in computing_power],
        network_state,
        epsilon=0.0
    )
    print(f"\n[DDPG决策] 分割比例: {[f'{r:.3f}' for r in ratios]}")
    # --------------------------------层内分割算法 DDPG---------------------------------------

    c_out_for_boundaries = c_out_list[adapcp_namenode.current_split_layer - 1] if adapcp_namenode.current_split_layer > 0 else 64
    boundaries = adapcp_namenode.compute_filter_boundaries(c_out_for_boundaries)
    print(f"[Filter边界] {boundaries}")

    # --------------------------------层分割优化 All Reduce---------------------------------------
    # --------------------------------层分割优化 All Reduce---------------------------------------

    local_output = adapcp_namenode.run_local_inference(input_tensor)
    edge_output = adapcp_namenode.run_allreduce_inference(local_output)

    # if edge_output is not None:
    #     final_output = adapcp_namenode.compute_fc_layers(edge_output)
    # else:
    #     final_output = local_output

    # 边缘节点已经完成了所有的层，主节点直接使用最终输出
    final_output = edge_output if edge_output is not None else local_output

    round_total_time = time.time() - round_start_time
    round_total_transfer = sum(transfer_time)

    print(f"\n[完成] 第 {round_idx} 轮推理")
    print(f"[统计] 总耗时: {round_total_time:.3f}s | 传输耗时: {round_total_transfer:.3f}s")

    adapcp_namenode.record_inference_result(round_total_time)

    return round_total_time


def run_legacy_inference(namenode, adapcp_namenode, round_idx):
    """传统流水线推理模式（保留兼容性）"""
    global transfer_time, thread_start_time, thread_end_time, thread_time
    transfer_time = []
    thread_start_time = [0] * datanode_num
    thread_end_time = [0] * datanode_num
    thread_time = [[] for _ in range(datanode_num)]

    input_tensor = torch.rand(1, 3, width, width)
    middle_output = input_tensor

    round_start_time = time.time()

    if datanode_num != 1:
        num_stages = len(adapcp_namenode.stage_boundaries)
        print(f"Total stages: {num_stages}")

        for stage in range(num_stages):
            adapcp_namenode.run_pipeline_stage(stage, middle_output)

            completed = adapcp_namenode.wait_for_stage_completion(stage)
            if not completed:
                print(f"ERROR: Stage {stage} did not complete in time")
                break

            middle_output = adapcp_namenode.merge_stage_results(stage)
            if middle_output is None:
                print(f"ERROR: Failed to merge stage {stage} results")
                break

            print(f"Stage {stage} merged output: {middle_output.size()}")

        if middle_output is not None and conv_length > 0:
            middle_output = adapcp_namenode.compute_fc_layers(middle_output)
    else:
        for layer_it in range(1, total_length + 1, 1):
            print(f"计算第 {layer_it} 层")
            if layer_it > conv_length:
                linear_start = time.time()
                middle_output = inference_model(middle_output, layer_it, layer_it)
                print(f'FC层耗时: {time.time() - linear_start:.3f}s')
            elif layer_it in maxpool_layer:
                pool_start = time.time()
                middle_output = inference_model(middle_output, layer_it, layer_it)
                print(f'池化层耗时: {time.time() - pool_start:.3f}s')
            elif (layer_it == 1) or (layer_it - 1 in maxpool_layer):
                start = layer_it
                end = get_end_layer(start, maxpool_layer) - 1
                cross_layer = end - start + 1
                print(f"单节点模式, 跨层数: {cross_layer}")

    round_total_time = time.time() - round_start_time
    round_total_transfer = sum(transfer_time)

    print(f"\n第 {round_idx} 轮完成")
    print(f"本轮总耗时: {round_total_time:.3f}s | 本轮总传输耗时: {round_total_transfer:.3f}s")

    return round_total_time


def get_end_layer(start=1, maxpool_layer=[]):
    max_value = max(maxpool_layer) if maxpool_layer else 0
    if start > max_value or start < 1:
        return 0
    for layer in maxpool_layer:
        if layer > start:
            return layer
    return max_value


if __name__ == "__main__":
    INFERENCE_MODE = "adapcp"

    print("AdapCP NameNode 启动...")
    print(f"推理模式: {INFERENCE_MODE}")

    namenode = Network_init_namenode(namenode_num=namenode_num, datanode_num=datanode_num)
    time.sleep(2)

    adapcp_namenode = AdapCPNameNode(namenode)

    print("\n" + "=" * 60)
    print("开始多轮推理...")
    print("=" * 60)

    for round_idx in range(1, TOTAL_ROUNDS + 1):
        if INFERENCE_MODE == "adapcp":
            run_adapcp_inference(namenode, adapcp_namenode, round_idx)
        else:
            run_legacy_inference(namenode, adapcp_namenode, round_idx)

        time.sleep(0.5)

    namenode.close_all()
    print("\n" + "=" * 60)
    print("所有轮次完成")
    print(f"预热轮次: {WARM_UP_ROUNDS} | 有效轮次: {VALID_ROUNDS}")
    print("关闭 NameNode 所有 Socket 连接")
    print("=" * 60)

import sys
import math

sys.path.append("../..")
sys.path.append("..")

from node_test.network_op import Network_init_datanode, Network_init_namenode
from node_test.network_op import EdgeNNNode, EdgeDataNode
from node_test.num_set_up import Num_set_up, sample_tenosr, MODEL_TYPE, INPUT_WIDTH, pool_layers_set, conv_layers
from node_test.num_set_up import c_out_list, conv_length, total_length, maxpool_layer, inference_model
from node_test.scheduler import PipelineScheduler, StageBoundary
from node_test.ilp_solver import OffloadingPartitioner
from node_test.ilp_solver import calculate_layer_flops_vgg13, calculate_layer_flops_vgg16
from node_test.ilp_solver import calculate_output_size_vgg13, calculate_output_size_vgg16
from node_test.adaptive_partitioner import AdaptivePartitioner
import torch
import threading
import time
import numpy as np
import torch.nn as nn
from VGG.tensor_op import tensor_divide_by_computing_network_and_fill
from VGG.tensor_op import merge_total_tensor, merge_filter_tensor
from VGG.tensor_op import dispatch_featuremap_to_nodes, tensor_divide_by_filter_ratios
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

width = INPUT_WIDTH
ddpg_state_dim = datanode_num * 4  # comp(a,b) + bw + load

transfer_time = []
thread_start_time = []
thread_end_time = []
thread_time = []


class AdapCPNameNode:
    def __init__(self, namenode):
        self.namenode = namenode
        self.edge_namenode = EdgeNNNode(namenode)
        self.scheduler = PipelineScheduler(datanode_num, total_length)
        self.offloading_partitioner = OffloadingPartitioner()
        self.ddpg_partitioner = AdaptivePartitioner(datanode_num, state_dim = ddpg_state_dim, use_dirichlet=True)
        self.current_split_layer = None
        self.current_ratios = None
        self.current_boundaries = None
        self.lock = threading.Lock()

        self._init_offloading_partitioner()

    # def _init_offloading_partitioner(self):
    #     """初始化OffloadingPartitioner，添加VGG13各层信息"""
    #     maxpool_layers = [3, 6, 9, 12, 15]

    #     for layer_id in range(1, 16):
    #         is_maxpool = layer_id in maxpool_layers
    #         flops = calculate_layer_flops_vgg13(layer_id)
    #         output_size = calculate_output_size_vgg13(layer_id)

    #         if layer_id <= 3:
    #             c_in, c_out = 3, 64
    #             h, w = 224, 224
    #         elif layer_id <= 6:
    #             c_in, c_out = 64, 128
    #             h, w = 112, 112
    #         elif layer_id <= 9:
    #             c_in, c_out = 128, 256
    #             h, w = 56, 56
    #         elif layer_id <= 12:
    #             c_in, c_out = 256, 512
    #             h, w = 28, 28
    #         else:
    #             c_in, c_out = 512, 512
    #             h, w = 14, 14

    #         self.offloading_partitioner.add_layer(
    #             layer_id, flops, output_size, is_maxpool, False, c_in, c_out, h, w
    #         )

    def _init_offloading_partitioner(self):
        """初始化OffloadingPartitioner，添加VGG各层信息"""
        maxpool_layers_set = set(maxpool_layer)
        total_layers_to_profile = total_length if total_length > 0 else 18

        for layer_id in range(1, total_layers_to_profile + 1):
            is_maxpool = layer_id in maxpool_layers_set
            is_fc = layer_id > 15

            if MODEL_TYPE == 'VGG16':
                flops = calculate_layer_flops_vgg16(layer_id, input_width=INPUT_WIDTH)
                output_size = calculate_output_size_vgg16(layer_id, input_width=INPUT_WIDTH)
            else:
                flops = calculate_layer_flops_vgg13(layer_id, input_width=INPUT_WIDTH)
                output_size = calculate_output_size_vgg13(layer_id, input_width=INPUT_WIDTH)

            if layer_id <= 3:
                c_in, c_out = 3, 64
                h, w = INPUT_WIDTH, INPUT_WIDTH
            elif layer_id <= 6:
                c_in, c_out = 64, 128
                h, w = INPUT_WIDTH // 2, INPUT_WIDTH // 2
            elif layer_id <= 9:
                c_in, c_out = 128, 256
                h, w = INPUT_WIDTH // 4, INPUT_WIDTH // 4
            elif layer_id <= 12:
                c_in, c_out = 256, 512
                h, w = INPUT_WIDTH // 8, INPUT_WIDTH // 8
            elif layer_id <= 15:
                c_in, c_out = 512, 512
                h, w = INPUT_WIDTH // 16, INPUT_WIDTH // 16
            else:
                if layer_id == 16:
                    c_in, c_out = 512 * 7 * 7, 4096
                elif layer_id == 17:
                    c_in, c_out = 4096, 4096
                else:
                    c_in, c_out = 4096, 1000
                h, w = 1, 1

            self.offloading_partitioner.add_layer(
                layer_id, flops, output_size, is_maxpool, is_fc, c_in, c_out, h, w
            )

    def compute_offloading_point(self):
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

    def compute_filter_boundaries(self):
        """根据DDPG比例计算Filter分割边界（3节点 → 4个边界）
        返回比例 boundaries [0, r0, r0+r1, r0+r1+r2, 1.0]
        子节点根据比例 * 各层 c_out 来计算实际输出通道数
        """
        if self.current_ratios is None:
            raise ValueError("DDPG ratios not computed")

        cumulative = 0.0
        boundaries = [0.0]
        for r in self.current_ratios:
            cumulative += r
            boundaries.append(round(cumulative, 6))
        boundaries[-1] = 1.0

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

    def run_edge_inference(self, input_tensor):
        """
        边缘协作推理: 广播plan给子节点，收集部分通道输出，合并
        """
        edge_start, edge_end = self.get_edge_layers()
        if edge_start > edge_end:
            return input_tensor

        print(f"\n===== 边缘协作推理: Layers {edge_start}-{edge_end} =====")

        plan = {'start_layer': edge_start, 'end_layer': edge_end}
        boundaries = self.current_boundaries

        print(f"[广播Plan] {plan}, 通道边界: {boundaries}")

        self.edge_namenode.broadcast_plan_to_all(plan, boundaries, input_tensor, c_out_list)

        slices = {}
        for _ in range(datanode_num):
            layer_id, node_id, tensor_slice = self.edge_namenode.collect_slice_from_datanode(
                _, transfer_time
            )
            slices[node_id] = tensor_slice

        sorted_slices = []
        for i in range(datanode_num):
            t = slices[i]
            if t.size(1) > 0:
                sorted_slices.append(t)

        if len(sorted_slices) > 0:
            merged = torch.cat(sorted_slices, dim=1)
        else:
            merged = torch.zeros(1, 0, 1, 1)

        print(f"合并后尺寸: {merged.size()}")

        return merged

    def _get_layer_groups(self, start_layer, end_layer):
        groups = []
        pool_layers_set = set(pool_layers_set)

        current_start = start_layer
        for layer_id in range(start_layer, end_layer + 1):
            if (layer_id in pool_layers_set or layer_id > 15) and layer_id < end_layer:
                groups.append((current_start, layer_id))
                current_start = layer_id + 1

        if current_start <= end_layer:
            groups.append((current_start, end_layer))

        return groups

    def compute_fc_layers(self, input_tensor):
        print("\n===== Computing FC Layers =====")
        for layer_id in range(conv_length + 1, total_length + 1):
            fc_start = time.time()
            input_tensor = inference_model(input_tensor, layer_id, layer_id)
            fc_end = time.time()
            print(f'FC Layer {layer_id} 耗时: {fc_end - fc_start:.3f}s')
        return input_tensor

    def record_inference_result(self, latency):
        """
        【关键修复】 记录推理结果用于DDPG训练。
        论文强化学习目标是最小化延迟，即最大化收益。
        因此通过取对数及负号的方式，将 Latency 映射为 Reward。
        """
        # 常数 C (例如10.0) 根据网络规模进行微调以保持 reward 尺度适当
        C = 10.0
        # 确保 latency > 0 防止对数异常
        latency_safe = max(latency, 1e-5) 
        reward = C - math.log(latency_safe) 
        
        self.ddpg_partitioner.record_experience(reward)
        print(f"[DRL更新] 记录执行延迟: {latency:.3f}s -> 映射为 Reward: {reward:.3f}")


def run_adapcp_inference(namenode, adapcp_namenode, round_idx):
    global transfer_time, thread_start_time, thread_end_time, thread_time

    print(f"\n{'='*60}")
    print(f"第 {round_idx} 轮 AdapCP 边缘协作推理")
    print(f"{'='*60}")

    transfer_time = []
    thread_start_time = [0] * datanode_num
    thread_end_time = [0] * datanode_num
    thread_time = [[] for _ in range(datanode_num)]

    input_tensor = sample_tenosr
    round_start_time = time.time()

    split_layer = adapcp_namenode.compute_offloading_point()
    plan = adapcp_namenode.offloading_partitioner.get_offloading_plan(split_layer)
    timing = adapcp_namenode.offloading_partitioner.estimate_pipeline_times(
        split_layer, device_power, edge_cluster_power, bandwidth
    )
    print(f"\n[ILP决策] {plan['description']}")
    print(f"[时间估算] 本地: {timing['local_time']:.3f}s, 传输: {timing['transfer_time']:.3f}s, 边缘: {timing['edge_time']:.3f}s")

    # 训练回合加上探索因子 epsilon, 让 DDPG 可以借助 Dirichlet+OU Noise 进行搜索
    epsilon_val = 0.5 if round_idx <= WARM_UP_ROUNDS else 0.0
    ratios = adapcp_namenode.compute_ddpg_ratios(
        [(c, 0.01) for c in computing_power],
        network_state,
        epsilon=epsilon_val
    )
    print(f"\n[DDPG决策] 分割比例: {[f'{r:.3f}' for r in ratios]} (Epsilon: {epsilon_val})")

    boundaries = adapcp_namenode.compute_filter_boundaries()
    print(f"[Filter边界] {boundaries}")

    local_output = adapcp_namenode.run_local_inference(input_tensor)
    edge_output = adapcp_namenode.run_edge_inference(local_output)

    final_output = edge_output if edge_output is not None else local_output

    round_total_time = time.time() - round_start_time
    round_total_transfer = sum(transfer_time)

    print(f"\n[完成] 第 {round_idx} 轮推理")
    print(f"[统计] 总耗时: {round_total_time:.3f}s | 传输耗时: {round_total_transfer:.3f}s")

    # 执行记录并转换 Reward
    adapcp_namenode.record_inference_result(round_total_time)

    return round_total_time


def run_legacy_inference(namenode, adapcp_namenode, round_idx):
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
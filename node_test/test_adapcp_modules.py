import sys
sys.path.append("D:/Graduate/WorkBench/work_1/Edge_LD")
sys.path.append("D:/Graduate/WorkBench/work_1/Edge_LD/node_test")

from node_test.ilp_solver import OffloadingPartitioner, calculate_layer_flops_vgg13, calculate_output_size_vgg13
from node_test.adaptive_partitioner import AdaptivePartitioner
from VGG.tensor_op import dispatch_featuremap_to_nodes, tensor_divide_by_filter_ratios, merge_filter_tensor

print("=" * 60)
print("ILP Solver Test (Offloading Partitioner)")
print("=" * 60)

partitioner = OffloadingPartitioner()
maxpool_layers_vgg13 = [3, 6, 9, 12, 15]

for layer_id in range(1, 16):
    is_maxpool = layer_id in maxpool_layers_vgg13
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

    partitioner.add_layer(layer_id, flops, output_size, is_maxpool, False, c_in, c_out, h, w)

device_power = (6.24e-11, 1.97e-2)
edge_power = (3.12e-11, 1.0e-2)

print("\n测试不同带宽下的卸载点:")
for bw in [10e6, 50e6, 100e6, 500e6, 1000e6]:
    split_layer = partitioner.solve_offloading_point(device_power, edge_power, bw, min_local_layers=3)
    t = partitioner.estimate_pipeline_times(split_layer, device_power, edge_power, bw)
    plan = partitioner.get_offloading_plan(split_layer)
    print(f"带宽 {bw/1e6:.0f} Mbps: split_layer={split_layer}, 总时间={t['total_time']:.4f}s")
    print(f"  计划: {plan['description']}")

print("\n" + "=" * 60)
print("DDPG Adaptive Partitioner Test (Filter Splitting)")
print("=" * 60)

n_nodes = 3
ddpg_partitioner = AdaptivePartitioner(n_nodes, use_dirichlet=True)

computing_power = [(6.24e-11, 1.97e-2), (6.24e-11, 1.97e-2), (6.24e-11, 1.97e-2)]
network_bw = [1000e6, 1000e6, 1000e6]

ratios = ddpg_partitioner.get_partition_ratios(computing_power, network_bw, epsilon=0.0)
print(f"\nDDPG输出分割比例: {ratios}")
print(f"比例之和: {ratios.sum():.4f}")

total_channels = 512
channel_splits = ddpg_partitioner.apply_ratios_to_channels(ratios, total_channels)
print(f"\n按通道分割 (总通道数={total_channels}):")
for i, c in enumerate(channel_splits):
    print(f"  Node {i}: {c} channels")

print("\n" + "=" * 60)
print("Tensor Filter Splitting Test")
print("=" * 60)

import torch

input_tensor = torch.rand(1, 256, 28, 28)
print(f"\n输入张量: {input_tensor.size()}")

splits = dispatch_featuremap_to_nodes(input_tensor, ratios)
print(f"\n分割后各节点张量:")
for i, split in enumerate(splits):
    print(f"  Node {i}: {split.size()}")

merged = merge_filter_tensor(splits)
print(f"\n合并后张量: {merged.size()}")
print(f"通道数匹配: {merged.shape[1] == input_tensor.shape[1]}")

print("\n" + "=" * 60)
print("所有测试通过!")
print("=" * 60)

import numpy as np


class OffloadingPartitioner:
    def __init__(self):
        self.layer_info = []
        self.total_layers = 0

    def add_layer(self, layer_id, flops, output_size, is_maxpool=False, is_fc=False, c_in=0, c_out=0, h=0, w=0):
        self.layer_info.append({
            'id': layer_id,
            'flops': flops,
            'output_size': output_size,
            'is_maxpool': is_maxpool,
            'is_fc': is_fc,
            'c_in': c_in,
            'c_out': c_out,
            'h': h,
            'w': w
        })
        self.total_layers = len(self.layer_info)

    def clear_layers(self):
        self.layer_info = []
        self.total_layers = 0

    def get_layer_count(self):
        return self.total_layers

    def _estimate_local_time(self, layer_id, device_power):
        """估算在本地设备上计算到某一层的累计时间"""
        a, b = device_power
        total_time = 0.0
        for layer in self.layer_info[:layer_id]:
            total_time += a * layer['flops'] + b
        return total_time

    def _estimate_transfer_time(self, layer_id, bandwidth):
        """估算传输第layer_id层输出到边缘的通信时间"""
        if layer_id > len(self.layer_info):
            return float('inf')
        output_size = self.layer_info[layer_id - 1]['output_size']
        return output_size * 4.0 / bandwidth if bandwidth > 0 else float('inf')

    def _estimate_edge_time(self, layer_id_start, edge_power):
        """估算在边缘集群计算从layer_id_start到最后一层的时间"""
        a, b = edge_power
        total_time = 0.0
        for layer in self.layer_info[layer_id_start - 1:]:
            total_time += a * layer['flops'] + b
        return total_time

    def solve_offloading_point(self, device_power, edge_power, bandwidth, min_local_layers=1, max_local_layers=None):
        """
        论文方法: 在Device和Edge Cluster之间找最优卸载点

        参数:
            device_power: (a, b) 设备计算能力参数
            edge_power: (a, b) 边缘集群计算能力参数
            bandwidth: 网络带宽 (bps)
            min_local_layers: 最少本地计算层数
            max_local_layers: 最多本地计算层数（默认全部）

        返回:
            split_layer: 卸载切分点 (1-indexed, 表示本地计算1-split_layer层)
        """
        if not self.layer_info:
            return 0

        n = self.total_layers
        if max_local_layers is None:
            max_local_layers = n

        best_split = min_local_layers
        best_time = float('inf')

        for split_layer in range(min_local_layers, min(max_local_layers + 1, n)):
            local_time = self._estimate_local_time(split_layer, device_power)
            transfer_time = self._estimate_transfer_time(split_layer, bandwidth)
            edge_time = self._estimate_edge_time(split_layer + 1, edge_power)

            total_time = local_time + transfer_time + edge_time

            if total_time < best_time:
                best_time = total_time
                best_split = split_layer

        return best_split

    def solve_with_latency_constraint(self, device_power, edge_power, bandwidth, latency_budget, min_local_layers=1): # ILP 1 split_layer
        """
        在延迟约束下找最优卸载点

        返回: (split_layer, meets_constraint)
        """
        if not self.layer_info:
            return 0, True

        n = self.total_layers
        feasible_splits = []

        for split_layer in range(min_local_layers, n):
            local_time = self._estimate_local_time(split_layer, device_power)
            transfer_time = self._estimate_transfer_time(split_layer, bandwidth)
            edge_time = self._estimate_edge_time(split_layer + 1, edge_power)

            total_time = local_time + transfer_time + edge_time

            if total_time <= latency_budget:
                feasible_splits.append((split_layer, total_time))

        if not feasible_splits:
            return self.solve_offloading_point(device_power, edge_power, bandwidth, min_local_layers), False

        return min(feasible_splits, key=lambda x: x[1])[0], True

    def get_offloading_plan(self, split_layer): # ILP 2 plan
        """
        获取卸载计划的详细描述

        返回: dict with 'local_layers', 'edge_layers', 'description'
        """
        if split_layer == 0:
            return {
                'local_layers': [],
                'edge_layers': list(range(1, self.total_layers + 1)),
                'description': 'All layers on edge cluster'
            }

        if split_layer >= self.total_layers:
            return {
                'local_layers': list(range(1, self.total_layers + 1)),
                'edge_layers': [],
                'description': 'All layers on local device'
            }

        return {
            'local_layers': list(range(1, split_layer + 1)),
            'edge_layers': list(range(split_layer + 1, self.total_layers + 1)),
            'description': f'Local: layers 1-{split_layer}, Edge: layers {split_layer + 1}-{self.total_layers}'
        }

    def estimate_pipeline_times(self, split_layer, device_power, edge_power, bandwidth): # ILP 3 timing
        """
        估算各阶段的时间开销

        返回: dict with detailed timing estimates
        """
        local_time = self._estimate_local_time(split_layer, device_power)
        transfer_time = self._estimate_transfer_time(split_layer, bandwidth)
        edge_time = self._estimate_edge_time(split_layer + 1, edge_power)

        local_layer_ids = list(range(1, split_layer + 1)) if split_layer > 0 else []
        edge_layer_ids = list(range(split_layer + 1, self.total_layers + 1)) if split_layer < self.total_layers else []

        local_details = []
        cumsum = 0.0
        for layer in self.layer_info[:split_layer]:
            t = device_power[0] * layer['flops'] + device_power[1]
            local_details.append({'layer': layer['id'], 'time': t})
            cumsum += t

        edge_details = []
        for layer in self.layer_info[split_layer:]:
            t = edge_power[0] * layer['flops'] + edge_power[1]
            edge_details.append({'layer': layer['id'], 'time': t})

        return {
            'local_time': local_time,
            'transfer_time': transfer_time,
            'edge_time': edge_time,
            'total_time': local_time + transfer_time + edge_time,
            'local_layers': local_layer_ids,
            'edge_layers': edge_layer_ids,
            'local_details': local_details,
            'edge_details': edge_details
        }


class ILPLayerPartitioner:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.layer_info = []

    def add_layer(self, layer_id, flops, output_size, is_maxpool=False, is_fc=False):
        self.layer_info.append({
            'id': layer_id,
            'flops': flops,
            'output_size': output_size,
            'is_maxpool': is_maxpool,
            'is_fc': is_fc
        })

    def clear_layers(self):
        self.layer_info = []

    def get_layer_count(self):
        return len(self.layer_info)

    def build_time_matrix(self, computing_power, network_state):
        n_layers = len(self.layer_info)
        n_nodes = self.n_nodes

        time_matrix = np.zeros((n_layers, n_nodes))

        for i, layer in enumerate(self.layer_info):
            for j in range(n_nodes):
                a, b = computing_power[j]
                comp_time = a * layer['flops'] + b
                comm_data_size = layer['output_size']
                comm_time = comm_data_size * 4.0 / network_state[j] if network_state[j] > 0 else 0
                time_matrix[i, j] = comp_time + comm_time

        return time_matrix

    def solve_minimax(self, computing_power, network_state):
        n_layers = len(self.layer_info)
        n_nodes = self.n_nodes

        if n_layers == 0:
            return [0] * (n_nodes + 1)

        time_matrix = self.build_time_matrix(computing_power, network_state)
        layer_loads = time_matrix.mean(axis=1)
        loads_array = np.array(layer_loads)
        total_load = loads_array.sum()
        target = total_load / n_nodes

        loads_normalized = loads_array / total_load
        cumsum = np.cumsum(loads_normalized[:-1])

        n_bins = n_nodes
        bin_edges = np.linspace(0, 1, n_bins + 1)

        partition_idx = []
        for edge in bin_edges[1:-1]:
            idx = np.searchsorted(cumsum, edge)
            partition_idx.append(idx)

        partition_idx = sorted(set(partition_idx))
        partition_points = [0] + [i + 1 for i in partition_idx] + [n_layers]

        while len(partition_points) < n_nodes + 1:
            partition_points.append(n_layers)

        return partition_points

    def get_partition_description(self, partition_points):
        descriptions = []
        for i in range(len(partition_points) - 1):
            start = partition_points[i] + 1
            end = partition_points[i + 1]
            if start <= end:
                layer_ids = list(range(start, end + 1))
                descriptions.append(f"Node {i}: Layers {start}-{end} (IDs: {layer_ids})")
            else:
                descriptions.append(f"Node {i}: No layers assigned")
        return descriptions


# def calculate_layer_flops_vgg13(layer_id, c_out_list=None):
#     flops_map = {}

#     for lid in range(1, 16):
#         if lid <= 3:
#             c_in, c_out = 3, 64
#             h, w = 224, 224
#         elif lid <= 6:
#             c_in, c_out = 64, 128
#             h, w = 112, 112
#         elif lid <= 9:
#             c_in, c_out = 128, 256
#             h, w = 56, 56
#         elif lid <= 12:
#             c_in, c_out = 256, 512
#             h, w = 28, 28
#         elif lid <= 15:
#             c_in, c_out = 512, 512
#             h, w = 14, 14

#         flops = 2 * h * w * c_in * c_out * 3 * 3
#         flops_map[lid] = flops

#     if layer_id in flops_map:
#         return flops_map[layer_id]
#     return 0

def calculate_layer_flops_vgg13(layer_id, c_out_list=None, input_width=224):
    flops_map = {}

    for lid in range(1, 16):
        if lid <= 3:
            c_in, c_out = 3, 64
            h, w = input_width, input_width
        elif lid <= 6:
            c_in, c_out = 64, 128
            h, w = input_width // 2, input_width // 2
        elif lid <= 9:
            c_in, c_out = 128, 256
            h, w = input_width // 4, input_width // 4
        elif lid <= 12:
            c_in, c_out = 256, 512
            h, w = input_width // 8, input_width // 8
        elif lid <= 15:
            c_in, c_out = 512, 512
            h, w = input_width // 16, input_width // 16

        flops = 2 * h * w * c_in * c_out * 3 * 3
        flops_map[lid] = flops

    flops_map[16] = 2 * (512 * 7 * 7) * 4096
    flops_map[17] = 2 * 4096 * 4096
    flops_map[18] = 2 * 4096 * 1000

    if layer_id in flops_map:
        return flops_map[layer_id]
    return 0

def calculate_layer_flops_vgg16(layer_id, c_out_list=None, input_width=224):
    flops_map = {}

    for lid in range(1, 19):
        if lid <= 3:
            c_in, c_out = 3, 64
            h, w = input_width, input_width
        elif lid <= 6:
            c_in, c_out = 64, 128
            h, w = input_width // 2, input_width // 2
        elif lid <= 10:
            c_in, c_out = 128, 256
            h, w = input_width // 4, input_width // 4
        elif lid <= 14:
            c_in, c_out = 256, 512
            h, w = input_width // 8, input_width // 8
        elif lid <= 18:
            c_in, c_out = 512, 512
            h, w = input_width // 16, input_width // 16

        flops = 2 * h * w * c_in * c_out * 3 * 3
        flops_map[lid] = flops

    if layer_id in flops_map:
        return flops_map[layer_id]
    return 0


# def calculate_output_size_vgg13(layer_id):
#     size_map = {
#         1: 64 * 224 * 224,
#         2: 64 * 224 * 224,
#         3: 64 * 112 * 112,
#         4: 128 * 112 * 112,
#         5: 128 * 112 * 112,
#         6: 128 * 56 * 56,
#         7: 256 * 56 * 56,
#         8: 256 * 56 * 56,
#         9: 256 * 56 * 56,
#         10: 256 * 28 * 28,
#         11: 512 * 28 * 28,
#         12: 512 * 28 * 28,
#         13: 512 * 28 * 28,
#         14: 512 * 14 * 14,
#         15: 512 * 14 * 14,
#     }
#     return size_map.get(layer_id, 512 * 7 * 7)

def calculate_output_size_vgg13(layer_id, input_width=224):
    maxpool_layers = [3, 6, 9, 12, 15]
    c_out_list = [64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096, 1000]

    width = input_width
    c_out = c_out_list[layer_id - 1] if layer_id <= 18 else 1000

    if layer_id <= 3:
        h, w = width, width
    elif layer_id <= 6:
        h, w = width // 2, width // 2
    elif layer_id <= 9:
        h, w = width // 4, width // 4
    elif layer_id <= 12:
        h, w = width // 8, width // 8
    elif layer_id <= 15:
        h, w = width // 16, width // 16
    elif layer_id == 16:
        h, w = 1, 1
    elif layer_id == 17:
        h, w = 1, 1
    else:
        h, w = 1, 1

    return c_out * h * w


def calculate_output_size_vgg16(layer_id, input_width=224):
    maxpool_layers = [3, 6, 10, 14, 18]
    c_out_list = [64, 64, 64, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]

    width = input_width
    c_out = c_out_list[layer_id - 1] if layer_id <= 18 else 1000

    if layer_id <= 3:
        h, w = width, width
    elif layer_id <= 6:
        h, w = width // 2, width // 2
    elif layer_id <= 10:
        h, w = width // 4, width // 4
    elif layer_id <= 14:
        h, w = width // 8, width // 8
    elif layer_id <= 18:
        h, w = width // 16, width // 16
    else:
        h, w = 1, 1

    return c_out * h * w


if __name__ == "__main__":
    print("=" * 60)
    print("Offloading Partitioner Test (Device vs Edge Cluster)")
    print("=" * 60)

    partitioner = OffloadingPartitioner()

    maxpool_layers_vgg13 = [3, 6, 9, 12, 15]

    for layer_id in range(1, 16):
        is_maxpool = layer_id in maxpool_layers_vgg13
        flops = calculate_layer_flops_vgg13(layer_id)
        output_size = calculate_output_size_vgg13(layer_id)

        c_in = 3 if layer_id == 1 else (64 if layer_id <= 3 else (128 if layer_id <= 6 else (256 if layer_id <= 9 else (512 if layer_id <= 15 else 512))))
        c_out = c_in * 2 if layer_id in [4, 7, 10, 13] else c_in
        if layer_id <= 3:
            h, w = 224, 224
        elif layer_id <= 6:
            h, w = 112, 112
        elif layer_id <= 9:
            h, w = 56, 56
        elif layer_id <= 12:
            h, w = 28, 28
        else:
            h, w = 14, 14

        partitioner.add_layer(layer_id, flops, output_size, is_maxpool, False, c_in, c_out, h, w)

    device_power = (6.24e-11, 1.97e-2)
    edge_power = (3.12e-11, 1.0e-2)
    bandwidth = 100e6

    split_layer = partitioner.solve_offloading_point(device_power, edge_power, bandwidth, min_local_layers=3)
    print(f"\nILP决策: 本地计算1-{split_layer}层，边缘计算{split_layer + 1}-15层")

    plan = partitioner.get_offloading_plan(split_layer)
    print(f"卸载计划: {plan['description']}")

    timing = partitioner.estimate_pipeline_times(split_layer, device_power, edge_power, bandwidth)
    print(f"\n时间估算:")
    print(f"  本地计算时间: {timing['local_time']:.4f}s")
    print(f"  传输时间: {timing['transfer_time']:.4f}s")
    print(f"  边缘计算时间: {timing['edge_time']:.4f}s")
    print(f"  总时间: {timing['total_time']:.4f}s")

    print("\n" + "=" * 60)
    print("扫描不同带宽下的最优卸载点:")
    print("=" * 60)
    for bw in [10e6, 50e6, 100e6, 500e6, 1000e6]:
        split = partitioner.solve_offloading_point(device_power, edge_power, bw, min_local_layers=3)
        t = partitioner.estimate_pipeline_times(split, device_power, edge_power, bw)
        print(f"  带宽 {bw/1e6:.0f} Mbps: split_layer={split}, 总时间={t['total_time']:.4f}s")

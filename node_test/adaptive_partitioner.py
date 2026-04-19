import numpy as np
import time
from node_test.ddpg_agent import DDPGAgent, DDPGAgentDirichlet


class AdaptivePartitioner:
    def __init__(self, n_nodes, state_dim=None, use_dirichlet=True, update_interval=5):
        self.n_nodes = n_nodes
        self.state_dim = state_dim if state_dim else n_nodes * 3
        self.use_dirichlet = use_dirichlet
        self.update_interval = update_interval

        if use_dirichlet:
            self.agent = DDPGAgentDirichlet(self.state_dim, n_nodes)
        else:
            self.agent = DDPGAgent(self.state_dim, n_nodes)

        self.update_counter = 0
        self.history = []
        self.best_partition = None
        self.best_latency = float('inf')

        self.current_state = None
        self.current_action = None

    def get_partition_ratios(self, computing_power, network_bw, current_load=None, epsilon=0.1):
        """
        获取当前环境的分区比例 (用于层内Filter Splitting)

        参数:
            computing_power: [(a, b), ...] 各节点计算能力参数
            network_bw: [bandwidth, ...] 网络带宽列表
            current_load: [load, ...] 当前负载 (可选)
            epsilon: 探索因子

        返回:
            ratios: [ratio0, ratio1, ratio2] sum=1.0
            这些比例直接用于按Channel维度分割tensor
        """
        self.current_state = self._build_state(computing_power, network_bw, current_load)
        self.current_action = self.agent.select_action(self.current_state, epsilon=epsilon)
        return self.current_action.copy()

    def get_deterministic_partition_ratios(self, computing_power, network_bw, current_load=None):
        """获取确定性（不探索）的分区比例"""
        self.current_state = self._build_state(computing_power, network_bw, current_load)
        self.current_action = self.agent.select_deterministic_action(self.current_state)
        return self.current_action.copy()

    def _build_state(self, computing_power, network_bw, current_load):
        """构建状态向量"""
        max_comp_a = max([c[0] for c in computing_power] + [1e-10])
        max_comp_b = max([c[1] for c in computing_power] + [1e-10])
        max_bw = max(network_bw + [1e-10])

        comp_normalized = []
        for a, b in computing_power:
            comp_normalized.append(a / max_comp_a)
            comp_normalized.append(b / max_comp_b)

        bw_normalized = [bw / max_bw for bw in network_bw]

        if current_load is not None:
            max_load = max(current_load + [1e-10])
            load_normalized = [l / max_load for l in current_load]
        else:
            load_normalized = [0.0] * self.n_nodes

        state = np.array(comp_normalized + bw_normalized + load_normalized)
        return state

    def record_experience(self, reward, next_state=None):
        """记录经验并定期更新"""
        if self.current_state is not None and self.current_action is not None:
            if next_state is None:
                next_state = self.current_state

            self.agent.add_experience(
                self.current_state,
                self.current_action,
                reward,
                next_state,
                False
            )

            self.history.append({
                'state': self.current_state.copy(),
                'action': self.current_action.copy(),
                'reward': reward,
                'timestamp': time.time()
            })

            if reward < self.best_latency:
                self.best_latency = reward
                self.best_partition = self.current_action.copy()

            self.update_counter += 1
            if self.update_counter >= self.update_interval:
                self.agent.update()
                self.update_counter = 0

    def apply_ratios_to_channels(self, ratios, total_channels):
        """
        将DDPG输出的分割比例转换为通道分割点（用于Filter Splitting）

        参数:
            ratios: [0.5, 0.3, 0.2] 各节点工作量比例，sum=1.0
            total_channels: 输入张量的总通道数 C_out

        返回:
            channel_splits: [c1, c2, c3, ...] 各节点处理的通道数
            例如 ratios=[0.5, 0.3, 0.2], total_channels=256
            返回 [128, 77, 51] (256*0.5=128, 256*0.3=77, 256-128-77=51)
        """
        channel_splits = []

        for i, r in enumerate(ratios[:-1]):
            c = int(total_channels * r)
            channel_splits.append(c)

        remainder = total_channels - sum(channel_splits)
        channel_splits.append(remainder)

        return channel_splits

    def get_channel_partition_points(self, ratios, total_channels):
        """
        获取通道分割的边界点索引

        参数:
            ratios: [0.5, 0.3, 0.2] 各节点工作量比例
            total_channels: 总通道数

        返回:
            boundary_points: [0, 128, 205, 256] 各分割点的起始索引
        """
        channel_splits = self.apply_ratios_to_channels(ratios, total_channels)
        boundary_points = [0]
        for c in channel_splits[:-1]:
            boundary_points.append(boundary_points[-1] + c)
        return boundary_points

    def get_best_partition(self):
        """获取历史最佳分区"""
        return self.best_partition

    def save(self, filepath):
        self.agent.save(filepath)

    def load(self, filepath):
        self.agent.load(filepath)


class PeriodicUpdateManager:
    def __init__(self, update_period=10):
        self.update_period = update_period
        self.inference_count = 0
        self.best_partition = None
        self.best_latency = float('inf')
        self.latency_history = []
        self.bandwidth_history = []
        self.server_count_history = []

    def should_update(self, current_bandwidth=None, current_server_count=None):
        """
        判断是否需要更新分区决策
        根据论文的自适应更新机制，当环境变化（带宽/服务器数量）超过阈值时触发更新
        """
        self.inference_count += 1

        if current_bandwidth is not None:
            self.bandwidth_history.append(current_bandwidth)
        if current_server_count is not None:
            self.server_count_history.append(current_server_count)

        if len(self.bandwidth_history) >= 2 and len(self.server_count_history) >= 2:
            delta_bw = abs(self.bandwidth_history[-1] - self.bandwidth_history[-2]) / max(self.bandwidth_history[-2], 1e-10)
            delta_sn = abs(self.server_count_history[-1] - self.server_count_history[-2]) / max(self.server_count_history[-2], 1e-10)

            epsilon1 = 10000
            epsilon2 = 100
            C1 = 1e-10

            p_u = np.sqrt(epsilon1 / (delta_bw + C1 * epsilon2 * delta_sn + C1))

            if p_u < 1.0:
                return True

        return self.inference_count % self.update_period == 0

    def on_inference_complete(self, partition, latency):
        """记录推理结果"""
        self.latency_history.append({
            'partition': partition,
            'latency': latency,
            'count': self.inference_count
        })

        if latency < self.best_latency:
            self.best_latency = latency
            self.best_partition = partition.copy()
            return True
        return False

    def get_best_partition(self):
        return self.best_partition, self.best_latency

    def get_average_latency(self, last_n=None):
        if not self.latency_history:
            return 0.0
        if last_n is None:
            last_n = len(self.latency_history)
        recent = self.latency_history[-last_n:]
        return sum([h['latency'] for h in recent]) / len(recent)


if __name__ == "__main__":
    n_nodes = 3
    partitioner = AdaptivePartitioner(n_nodes, use_dirichlet=True)

    computing_power = [(6.24e-11, 1.97e-2), (6.24e-11, 1.97e-2), (6.24e-11, 1.97e-2)]
    network_bw = [1000e6, 1000e6, 1000e6]
    current_load = [0.5, 0.5, 0.5]

    print("=" * 60)
    print("DDPG Adaptive Partitioner Test (Filter Splitting)")
    print("=" * 60)

    ratios = partitioner.get_partition_ratios(computing_power, network_bw, current_load, epsilon=0.0)
    print(f"\nDDPG输出分割比例: {ratios}")
    print(f"比例之和: {ratios.sum():.4f}")

    total_channels = 512
    channel_splits = partitioner.apply_ratios_to_channels(ratios, total_channels)
    print(f"\n按通道分割 (总通道数={total_channels}):")
    for i, c in enumerate(channel_splits):
        print(f"  Node {i}: {c} channels")

    boundary_points = partitioner.get_channel_partition_points(ratios, total_channels)
    print(f"\n通道分割边界点: {boundary_points}")

    for i in range(50):
        reward = np.random.rand() * 10
        partitioner.record_experience(reward)

    print(f"\n历史最佳分区: {partitioner.get_best_partition()}")

    print("\n" + "=" * 60)
    print("PeriodicUpdateManager Test")
    print("=" * 60)

    update_mgr = PeriodicUpdateManager(update_period=5)

    for i in range(20):
        bw = 100e6 + np.random.rand() * 50e6
        sn = 3
        should_update = update_mgr.should_update(bw, sn)
        latency = np.random.rand() * 5 + 1
        update_mgr.on_inference_complete([0.4, 0.35, 0.25], latency)

        if should_update:
            print(f"Step {i}: 需要更新分区")

    print(f"\n最佳延迟: {update_mgr.get_best_partition()}")

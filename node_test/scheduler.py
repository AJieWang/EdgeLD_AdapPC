import threading
from collections import defaultdict
import time
import torch


class TaskInfo:
    def __init__(self, task_id, stage, datanode_id, start, end):
        self.task_id = task_id
        self.stage = stage
        self.datanode_id = datanode_id
        self.start = start
        self.end = end
        self.submitted_at = time.time()
        self.completed_at = None
        self.result = None


class PipelineScheduler:
    def __init__(self, datanode_num, total_layers):
        self.datanode_num = datanode_num
        self.total_layers = total_layers
        self.pending_tasks = {}
        self.completed_results = {}
        self.datanode_available = {i: True for i in range(datanode_num)}
        self.stage_results = defaultdict(dict)
        self.layer_merged = {}
        self.lock = threading.Lock()
        self.wait_event = threading.Event()
        self.task_counter = 0

    def generate_task_id(self, stage, datanode_id, start, end):
        self.task_counter += 1
        return f"s{stage}_d{datanode_id}_l{start}-{end}_{self.task_counter}"

    def create_task(self, stage, datanode_id, start, end):
        task_id = self.generate_task_id(stage, datanode_id, start, end)
        with self.lock:
            self.pending_tasks[task_id] = TaskInfo(task_id, stage, datanode_id, start, end)
            self.datanode_available[datanode_id] = False
        return task_id

    def on_task_complete(self, task_id, result_tensor):
        with self.lock:
            if task_id not in self.pending_tasks:
                return
            task_info = self.pending_tasks.pop(task_id)
            task_info.completed_at = time.time()
            task_info.result = result_tensor

            stage = task_info.stage
            datanode_id = task_info.datanode_id
            self.completed_results[task_id] = result_tensor
            self.stage_results[stage][datanode_id] = result_tensor
            self.datanode_available[datanode_id] = True

            self.wait_event.set()

    def is_stage_complete(self, stage):
        with self.lock:
            completed_in_stage = len(self.stage_results.get(stage, {}))
            return completed_in_stage == self.datanode_num

    def get_stage_results(self, stage):
        with self.lock:
            return self.stage_results.get(stage, {}).copy()

    def clear_stage(self, stage):
        with self.lock:
            if stage in self.stage_results:
                del self.stage_results[stage]

    def get_task_info(self, task_id):
        with self.lock:
            return self.pending_tasks.get(task_id)

    def is_all_complete(self):
        with self.lock:
            return len(self.pending_tasks) == 0

    def get_available_datanode(self):
        with self.lock:
            for dn_id, available in self.datanode_available.items():
                if available:
                    return dn_id
            return None

    def get_completion_time(self, task_id):
        with self.lock:
            if task_id in self.pending_tasks:
                return None
            if task_id in self.completed_results:
                task_info = self.pending_tasks.get(task_id)
                if task_info and task_info.completed_at:
                    return task_info.completed_at - task_info.submitted_at
            return None


class StageBoundary:
    @staticmethod
    def compute_vgg13_boundaries():
        maxpool_layer = [3, 6, 9, 12, 15]
        total_length = 18
        boundaries = []
        start = 1
        for mp_layer in maxpool_layer:
            if start <= mp_layer:
                boundaries.append((start, mp_layer))
                start = mp_layer + 1
        if start <= total_length:
            boundaries.append((start, total_length))
        return boundaries

    @staticmethod
    def compute_vgg16_boundaries():
        maxpool_layer = [3, 6, 10, 14, 18]
        total_length = 21
        boundaries = []
        start = 1
        for mp_layer in maxpool_layer:
            if start <= mp_layer:
                boundaries.append((start, mp_layer))
                start = mp_layer + 1
        if start <= total_length:
            boundaries.append((start, total_length))
        return boundaries

    @staticmethod
    def compute_boundaries(model_type='VGG13'):
        if model_type == 'VGG13':
            return StageBoundary.compute_vgg13_boundaries()
        elif model_type == 'VGG16':
            return StageBoundary.compute_vgg16_boundaries()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    scheduler = PipelineScheduler(datanode_num=3, total_layers=18)

    boundaries = StageBoundary.compute_boundaries('VGG13')
    print("VGG13 Stage Boundaries:", boundaries)

    boundaries = StageBoundary.compute_boundaries('VGG16')
    print("VGG16 Stage Boundaries:", boundaries)

    task_id = scheduler.create_task(stage=0, datanode_id=0, start=1, end=3)
    print(f"Created task: {task_id}")
    print(f"Available datanodes: {scheduler.get_available_datanode()}")

    scheduler.on_task_complete(task_id, torch.rand(1, 64, 112, 112))
    print(f"Stage 0 results: {scheduler.get_stage_results(0)}")


class SliceInfo:
    def __init__(self, layer_id, datanode_id, tensor_slice):
        self.layer_id = layer_id
        self.datanode_id = datanode_id
        self.tensor_slice = tensor_slice
        self.received_at = time.time()


class AllReduceScheduler:
    """
    AdapCP AllReduce 调度器

    负责：
    1. Gather: 收集各子节点返回的切片
    2. Merge: 在主节点合并切片 (torch.cat)
    3. Broadcast: 将合并后的完整张量广播给所有子节点
    """

    def __init__(self, datanode_num):
        self.datanode_num = datanode_num

        self.lock = threading.Lock()

        self.layer_slices = {}

        self.stage_events = {}

        self.waiting_for_slices = {}

        self.merged_results = {}

        self.broadcast_data = {}

        self.broadcast_events = {i: threading.Event() for i in range(datanode_num)}

    def submit_slice(self, layer_id, datanode_id, tensor_slice):
        """
        步骤1 (Gather): 子节点计算完成后提交切片给主节点

        参数:
            layer_id: 当前层ID
            datanode_id: 提交切片的节点ID
            tensor_slice: 该节点计算的切片结果
        """
        with self.lock:
            if layer_id not in self.layer_slices:
                self.layer_slices[layer_id] = {}
                self.stage_events[layer_id] = threading.Event()

            self.layer_slices[layer_id][datanode_id] = tensor_slice

            print(f"[Gather] Layer {layer_id}, Node {datanode_id} submitted slice: {tensor_slice.size()}")

            if len(self.layer_slices[layer_id]) == self.datanode_num:
                self.stage_events[layer_id].set()

    def wait_and_merge_all_reduce(self, layer_id):
        """
        步骤2 (Merge): 主节点等待收集齐该层所有切片，合并后返回

        参数:
            layer_id: 当前层ID

        返回:
            merged_tensor: 合并后的完整张量
        """
        if layer_id not in self.stage_events:
            with self.lock:
                self.stage_events[layer_id] = threading.Event()

        print(f"[Merge] Layer {layer_id} waiting for all slices...")
        self.stage_events[layer_id].wait()

        with self.lock:
            slices_dict = self.layer_slices[layer_id]

            sorted_slices = [slices_dict[i] for i in range(self.datanode_num)]

            print(f"[Merge] Layer {layer_id}, concatenating {len(sorted_slices)} slices")
            for i, s in enumerate(sorted_slices):
                print(f"  Slice {i}: {s.size()}")

            merged_tensor = torch.cat(sorted_slices, dim=1)

            print(f"[Merge] Layer {layer_id} merged result: {merged_tensor.size()}")

            self.merged_results[layer_id] = merged_tensor

            return merged_tensor

    def get_broadcast_data(self, datanode_id):
        """
        获取需要广播给指定节点的数据（阻塞等待）

        参数:
            datanode_id: 节点ID

        返回:
            tensor: 需要广播的张量
        """
        self.broadcast_events[datanode_id].wait()
        self.broadcast_events[datanode_id].clear()

        with self.lock:
            data = self.broadcast_data.get(datanode_id)
            return data

    def broadcast_to_all(self, tensor):
        """
        步骤3 (Broadcast): 向所有子节点广播合并后的完整张量

        参数:
            tensor: 合并后的完整张量
        """
        with self.lock:
            for dn_id in range(self.datanode_num):
                self.broadcast_data[dn_id] = tensor

            for dn_id in range(self.datanode_num):
                self.broadcast_events[dn_id].set()

        print(f"[Broadcast] Sent merged tensor {tensor.size()} to all {self.datanode_num} nodes")

    def is_layer_complete(self, layer_id):
        """检查指定层的所有切片是否已收集"""
        with self.lock:
            return len(self.layer_slices.get(layer_id, {})) == self.datanode_num

    def get_slice_count(self, layer_id):
        """获取已收集的切片数量"""
        with self.lock:
            return len(self.layer_slices.get(layer_id, {}))

    def reset_layer(self, layer_id):
        """重置指定层的状态（用于下一轮推理）"""
        with self.lock:
            if layer_id in self.layer_slices:
                del self.layer_slices[layer_id]
            if layer_id in self.stage_events:
                self.stage_events[layer_id].clear()
            if layer_id in self.merged_results:
                del self.merged_results[layer_id]

    def reset_all(self):
        """重置所有状态"""
        with self.lock:
            self.layer_slices.clear()
            self.merged_results.clear()
            self.broadcast_data.clear()
            for event in self.stage_events.values():
                event.clear()
            for event in self.broadcast_events.values():
                event.clear()


class AllReduceTaskInfo:
    """AllReduce任务信息"""
    def __init__(self, task_id, layer_id, datanode_id, start_filter, end_filter):
        self.task_id = task_id
        self.layer_id = layer_id
        self.datanode_id = datanode_id
        self.start_filter = start_filter
        self.end_filter = end_filter
        self.slice_result = None
        self.submitted_at = time.time()
        self.completed_at = None

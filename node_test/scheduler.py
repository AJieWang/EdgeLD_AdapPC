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

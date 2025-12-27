import time
import uuid
import threading
import random
from monitor import HardwareMonitor
from workloads import WorkloadExecutor
from recorder import DataRecorder

class AstraController:
    def __init__(
        self,
        simulation_mode=True,
        yolo_input_path="yolo_input",
        yolo_output_dir="yolo_output",
        yolo_output_format="all",
        yolo_max_images=None,
    ):
        self.recorder = DataRecorder()
        self.monitor = HardwareMonitor(use_simulation=simulation_mode)
        
        # 线程安全锁，用于统计当前活跃任务数供 Monitor 使用
        self.lock = threading.Lock()
        self.active_tasks_count = {'IO': 0, 'NET': 0, 'YOLO': 0}
        self.yolo_task_state = {}
        self.yolo_input_path = yolo_input_path
        self.yolo_output_dir = yolo_output_dir
        self.yolo_output_format = yolo_output_format
        self.yolo_max_images = yolo_max_images
        
        self.running = True

    def _monitor_loop(self):
        """后台监控线程"""
        while self.running:
            # 1. 获取当前活跃任务快照
            with self.lock:
                snapshot = self.active_tasks_count.copy()
            
            # 2. 采集硬件数据
            metrics = self.monitor.get_metrics(snapshot)
            with self.lock:
                yolo_tasks = [
                    {
                        "task_id": task_id,
                        "total_images": info["total_images"],
                        "remaining_images": info["remaining_images"],
                    }
                    for task_id, info in self.yolo_task_state.items()
                ]
            metrics["yolo_tasks"] = yolo_tasks
            
            # 3. 记录
            self.recorder.log_metric(metrics)
            
            # 4. 采样频率 2Hz (每0.5秒)
            time.sleep(0.5)

    def _worker_wrapper(self, task_type, duration=None, **kwargs):
        """任务执行包装器 (处理日志和计数)"""
        task_id = str(uuid.uuid4())[:8]
        
        # START Event
        with self.lock:
            self.active_tasks_count[task_type] += 1
        event_details = {}
        if task_type == 'YOLO':
            event_details = self._init_yolo_task_state(task_id, **kwargs)
        self.recorder.log_event(task_type, "START", task_id, event_details)
        print(f"[{time.strftime('%H:%M:%S')}] START Task: {task_type} (ID: {task_id})")
        
        # Execution
        if task_type == 'IO':
            WorkloadExecutor.task_io_stress(duration)
        elif task_type == 'NET':
            WorkloadExecutor.task_network_stress(duration)
        elif task_type == 'YOLO':
            self._run_yolo_task(task_id, **kwargs)
            
        # END Event
        with self.lock:
            self.active_tasks_count[task_type] -= 1
        if task_type == 'YOLO':
            self._finalize_yolo_task_state(task_id)
        self.recorder.log_event(task_type, "END", task_id)
        print(f"[{time.strftime('%H:%M:%S')}] END   Task: {task_type} (ID: {task_id})")

    def _init_yolo_task_state(self, task_id, **kwargs):
        from yolo_workload import collect_images

        input_path = kwargs.get("input_path", self.yolo_input_path)
        max_images = kwargs.get("max_images", self.yolo_max_images)
        total_images = 0
        try:
            images, _ = collect_images(input_path, max_images=max_images)
            total_images = len(images)
        except Exception as exc:
            print(f"[YOLO] Failed to scan images at {input_path}: {exc}")

        with self.lock:
            self.yolo_task_state[task_id] = {
                "total_images": total_images,
                "remaining_images": total_images,
            }
        return {"total_images": total_images}

    def _update_yolo_task_state(self, task_id, processed, total):
        remaining = max(total - processed, 0)
        with self.lock:
            self.yolo_task_state[task_id] = {
                "total_images": total,
                "remaining_images": remaining,
            }

    def _finalize_yolo_task_state(self, task_id):
        with self.lock:
            self.yolo_task_state.pop(task_id, None)

    def _run_yolo_task(self, task_id, **kwargs):
        input_path = kwargs.get("input_path", self.yolo_input_path)
        output_dir = kwargs.get("output_dir", self.yolo_output_dir)
        output_format = kwargs.get("output_format", self.yolo_output_format)
        max_images = kwargs.get("max_images", self.yolo_max_images)

        def progress_callback(processed, total):
            self._update_yolo_task_state(task_id, processed, total)

        try:
            WorkloadExecutor.task_yolo_inference_ascend(
                input_path=input_path,
                output_dir=output_dir,
                output_format=output_format,
                max_images=max_images,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            print(f"[YOLO] Task {task_id} failed: {exc}")

    def dispatch_task(self, task_type, duration=None, **kwargs):
        """派发任务到线程池"""
        t = threading.Thread(
            target=self._worker_wrapper, args=(task_type, duration), kwargs=kwargs
        )
        t.start()

    def run_simulation(self, total_time=30):
        print(f"=== ASTRA Simulation Started (Mode: {'SIM' if self.monitor.use_simulation else 'REAL'}) ===")
        print(f"Duration: {total_time} seconds")
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.start()
        
        start_ts = time.time()
        
        # === 随机调度循环 ===
        # 这里模拟卫星经过不同区域时的负载变化
        while time.time() - start_ts < total_time:
            
            # 随机概率触发任务
            dice = random.random()
            
            # 10% 概率触发 YOLO (模拟发现目标)
            if dice < 0.1:
                duration = random.randint(3, 8)
                self.dispatch_task(
                    "YOLO",
                    duration,
                    input_path=self.yolo_input_path,
                    output_dir=self.yolo_output_dir,
                    output_format=self.yolo_output_format,
                    max_images=self.yolo_max_images,
                )
                
            # 20% 概率触发 网络传输 (模拟下行数据)
            elif dice < 0.3:
                duration = random.randint(2, 5)
                self.dispatch_task("NET", duration)
                
            # 15% 概率触发 IO (模拟存图)
            elif dice < 0.45:
                duration = random.randint(1, 3)
                self.dispatch_task("IO", duration)
                
            time.sleep(1) # 调度间隔

        print("=== Simulation Time Up. Stopping... ===")
        self.running = False
        monitor_thread.join()
        
        # 生成最终数据集
        self.recorder.save_dataset()

if __name__ == "__main__":
    # 如果在真实的 Ascend 开发板上运行，将 use_simulation 设为 False
      app = AstraController(
          simulation_mode=False,              # 真实 Ascend 指标
          yolo_input_path="/home/ubuntu/data/test",
          yolo_output_dir="tmp/yolo_ascend_output",
          yolo_output_format="all",
          yolo_max_images=3000,                  # 每个 YOLO 任务最多推理 10 张
      )
      app.run_simulation(total_time=60)
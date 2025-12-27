import time
import uuid
import threading
import random
import multiprocessing as mp
import queue as py_queue
import argparse
from monitor import HardwareMonitor
from workloads import WorkloadExecutor
from recorder import DataRecorder

class AstraController:
    def __init__(
        self,
        simulation_mode=True,
        yolo_input_path="yolo_input",
        yolo_output_dir="tmp/yolo_workload",
        yolo_output_format="all",
        yolo_max_images=None,
        yolo_stop_grace=5.0,
        yolo_max_concurrent=None,
        yolo_verbose=False,
    ):
        self.recorder = DataRecorder()
        self.monitor = HardwareMonitor(use_simulation=simulation_mode)
        self.mp_context = mp.get_context("spawn")
        
        # 线程安全锁，用于统计当前活跃任务数供 Monitor 使用
        self.lock = threading.Lock()
        self.active_tasks_count = {'IO': 0, 'NET': 0, 'YOLO': 0}
        self.yolo_task_state = {}
        self.yolo_input_path = yolo_input_path
        self.yolo_output_dir = yolo_output_dir
        self.yolo_output_format = yolo_output_format
        self.yolo_max_images = yolo_max_images
        self.yolo_stop_grace = yolo_stop_grace
        self.yolo_stop_event = self.mp_context.Event()
        self.yolo_max_concurrent = yolo_max_concurrent
        self.yolo_verbose = yolo_verbose
        if yolo_max_concurrent is None:
            self.yolo_process_sema = None
        else:
            if int(yolo_max_concurrent) < 1:
                raise ValueError("yolo_max_concurrent must be >= 1")
            self.yolo_process_sema = threading.BoundedSemaphore(
                int(yolo_max_concurrent)
            )
        
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
                        "thread_name": info.get("thread_name"),
                        "thread_id": info.get("thread_id"),
                        "process_id": info.get("process_id"),
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
        
        event_details = {}
        if task_type == 'YOLO':
            if self.yolo_process_sema:
                self.yolo_process_sema.acquire()
            event_details = self._init_yolo_task_state(task_id, **kwargs)
        else:
            with self.lock:
                self.active_tasks_count[task_type] += 1
            self.recorder.log_event(task_type, "START", task_id)
            print(f"[{time.strftime('%H:%M:%S')}] START Task: {task_type} (ID: {task_id})")
        
        # Execution
        if task_type == 'IO':
            WorkloadExecutor.task_io_stress(duration)

            with self.lock:
                self.active_tasks_count[task_type] -= 1
            self.recorder.log_event(task_type, "END", task_id)
            print(f"[{time.strftime('%H:%M:%S')}] END   Task: {task_type} (ID: {task_id})")
        elif task_type == 'NET':
            WorkloadExecutor.task_network_stress(duration)

            with self.lock:
                self.active_tasks_count[task_type] -= 1
            self.recorder.log_event(task_type, "END", task_id)
            print(f"[{time.strftime('%H:%M:%S')}] END   Task: {task_type} (ID: {task_id})")
        elif task_type == 'YOLO':
            started = {"value": False}

            def on_started(process_id):
                with self.lock:
                    self.active_tasks_count[task_type] += 1
                started["value"] = True
                details = dict(event_details)
                details["process_id"] = process_id
                self.recorder.log_event(task_type, "START", task_id, details)
                print(
                    f"[{time.strftime('%H:%M:%S')}] START Task: {task_type} (ID: {task_id}, PID: {process_id})"
                )

            def on_finished(process_id):
                if started["value"]:
                    with self.lock:
                        self.active_tasks_count[task_type] -= 1
                details = {
                    "process_id": process_id,
                    "thread_name": event_details.get("thread_name"),
                    "thread_id": event_details.get("thread_id"),
                }
                self.recorder.log_event(task_type, "END", task_id, details)
                print(
                    f"[{time.strftime('%H:%M:%S')}] END   Task: {task_type} (ID: {task_id}, PID: {process_id})"
                )
                self._finalize_yolo_task_state(task_id)
                if self.yolo_process_sema:
                    self.yolo_process_sema.release()

            self._run_yolo_task(task_id, on_started, on_finished, **kwargs)

    def _init_yolo_task_state(self, task_id, **kwargs):
        from yolo_workload import collect_images

        input_path = kwargs.get("input_path", self.yolo_input_path)
        max_images = kwargs.get("max_images", self.yolo_max_images)
        thread = threading.current_thread()
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
                "thread_name": thread.name,
                "thread_id": thread.ident,
                "process_id": None,
            }
        return {
            "total_images": total_images,
            "thread_name": thread.name,
            "thread_id": thread.ident,
        }

    def _update_yolo_task_state(self, task_id, processed, total):
        remaining = max(total - processed, 0)
        with self.lock:
            current = self.yolo_task_state.get(task_id, {})
            current.update(
                {
                    "total_images": total,
                    "remaining_images": remaining,
                }
            )
            self.yolo_task_state[task_id] = current

    def _set_yolo_process_info(self, task_id, process_id):
        with self.lock:
            current = self.yolo_task_state.get(task_id, {})
            current["process_id"] = process_id
            self.yolo_task_state[task_id] = current

    def _finalize_yolo_task_state(self, task_id):
        with self.lock:
            self.yolo_task_state.pop(task_id, None)

    def _run_yolo_task(self, task_id, on_started, on_finished, **kwargs):
        from yolo_workload import run_inference_worker

        input_path = kwargs.get("input_path", self.yolo_input_path)
        output_dir = kwargs.get("output_dir", self.yolo_output_dir)
        output_format = kwargs.get("output_format", self.yolo_output_format)
        max_images = kwargs.get("max_images", self.yolo_max_images)
        weights = kwargs.get("weights")
        labels = kwargs.get("labels")
        verbose = kwargs.get("verbose", self.yolo_verbose)

        progress_queue = self.mp_context.Queue()
        process = None
        try:
            process = self.mp_context.Process(
                target=run_inference_worker,
                kwargs={
                    "input_path": input_path,
                    "output_dir": output_dir,
                    "output_format": output_format,
                    "weights": weights,
                    "labels": labels,
                    "max_images": max_images,
                    "progress_queue": progress_queue,
                    "stop_event": self.yolo_stop_event,
                    "verbose": verbose,
                },
            )
            process.start()
            self._set_yolo_process_info(task_id, process.pid)
            on_started(process.pid)

            while process.is_alive() or not progress_queue.empty():
                try:
                    msg_type, value, total = progress_queue.get(timeout=0.5)
                except py_queue.Empty:
                    continue

                if msg_type == "progress":
                    self._update_yolo_task_state(task_id, value, total)
                elif msg_type == "error":
                    print(f"[YOLO] Task {task_id} failed: {value}")
                elif msg_type == "done":
                    pass
        except Exception as exc:
            print(f"[YOLO] Task {task_id} failed: {exc}")
        finally:
            if process is not None:
                process.join()
                on_finished(process.pid)
            else:
                on_finished(None)

    def dispatch_task(self, task_type, duration=None, **kwargs):
        """派发任务到线程池"""
        t = threading.Thread(
            target=self._worker_wrapper, args=(task_type, duration), kwargs=kwargs
        )
        t.start()

    def run_simulation(self, total_time=30):
        print(f"=== ASTRA Simulation Started (Mode: {'SIM' if self.monitor.use_simulation else 'REAL'}) ===")
        print(f"Duration: {total_time} seconds")
        self.yolo_stop_event.clear()
        
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
        self._stop_yolo_tasks_after_grace()

    def _stop_yolo_tasks_after_grace(self):
        if self.yolo_stop_grace is None:
            return

        deadline = time.time() + max(self.yolo_stop_grace, 0)
        while time.time() < deadline:
            with self.lock:
                if not self.yolo_task_state:
                    return
            time.sleep(0.2)

        if self.yolo_task_state:
            print("[YOLO] Grace period elapsed, stopping remaining YOLO tasks.")
            self.yolo_stop_event.set()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASTRA scheduler with optional YOLO concurrency control"
    )
    parser.add_argument(
        "--yolo-max-concurrent",
        type=int,
        default=None,
        help="Maximum number of concurrent YOLO tasks (processes)",
    )
    parser.add_argument(
        "--total-time",
        type=int,
        default=60,
        help="Simulation duration in seconds",
    )
    args = parser.parse_args()

    # 如果在真实的 Ascend 开发板上运行，将 use_simulation 设为 False
    app = AstraController(
        simulation_mode=False,  # 真实 Ascend 指标
        yolo_input_path="/home/ubuntu/data/test",
        yolo_output_dir="tmp/yolo_workload",
        yolo_output_format="all",
        yolo_max_images=3000,  # 每个 YOLO 任务最多推理 10 张
        yolo_max_concurrent=args.yolo_max_concurrent,
    )
    app.run_simulation(total_time=args.total_time)

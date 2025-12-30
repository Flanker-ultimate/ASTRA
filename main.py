import time
import uuid
import threading
import random
import multiprocessing as mp
import queue as py_queue
import argparse
import os
from monitor import HardwareMonitor
from workloads import WorkloadExecutor
from recorder import DataRecorder

class AstraController:
    YOLO_TASK_PREFIX = "yolo_task_"

    def __init__(
        self,
        simulation_mode=True,
        yolo_input_path="yolo_input",
        yolo_output_dir="tmp/yolo_workload",
        yolo_output_format="all",
        yolo_max_images=None,
        yolo_stop_grace=5.0,
        yolo_max_concurrent=1,
        yolo_verbose=False,
        yolo_shutdown_timeout=10.0,
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
        self.yolo_max_concurrent = 1 if yolo_max_concurrent is None else yolo_max_concurrent
        self.yolo_verbose = yolo_verbose
        self.yolo_shutdown_timeout = yolo_shutdown_timeout
        self.worker_threads = []
        self.yolo_processes = {}
        self.yolo_stage_lock = threading.Lock()
        if int(self.yolo_max_concurrent) < 1:
            raise ValueError("yolo_max_concurrent must be >= 1")
        if int(self.yolo_max_concurrent) != 1:
            raise ValueError("yolo_max_concurrent must be 1 in single-process mode")
        self.yolo_process_sema = threading.BoundedSemaphore(
            int(self.yolo_max_concurrent)
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
                        "task_prefix": info.get("task_prefix"),
                        "batch_start_time": info.get("batch_start_time"),
                        "status": info.get("status"),
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
            event_details = self._init_yolo_task_state(task_id, **kwargs)
            self.recorder.log_event(task_type, "BATCH_START", task_id, event_details)
            print(
                f"[{time.strftime('%H:%M:%S')}] BATCH START Task: {task_type} (ID: {task_id})"
            )
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
            task_prefix = event_details.get("task_prefix")
            total_images = event_details.get("total_images", 0)

            def on_started(process_id):
                if process_id is not None:
                    with self.lock:
                        self.active_tasks_count[task_type] += 1
                    started["value"] = True
                    self._update_yolo_task_meta(task_id, status="running")
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
                    "batch_end_time": time.time(),
                }
                self.recorder.log_event(task_type, "END", task_id, details)
                print(
                    f"[{time.strftime('%H:%M:%S')}] END   Task: {task_type} (ID: {task_id}, PID: {process_id})"
                )
                self._finalize_yolo_task_state(task_id)

            self._run_yolo_task(
                task_id,
                on_started,
                on_finished,
                task_prefix=task_prefix,
                total_images=total_images,
                **kwargs,
            )

    def _init_yolo_task_state(self, task_id, **kwargs):
        input_path = kwargs.get("input_path", self.yolo_input_path)
        max_images = kwargs.get("max_images", self.yolo_max_images)
        batch_start_time = time.time()
        task_prefix = f"{self.YOLO_TASK_PREFIX}{task_id}_"
        thread = threading.current_thread()
        try:
            with self.yolo_stage_lock:
                total_images = self._stage_yolo_inputs(
                    input_path, task_prefix, max_images
                )
        except Exception as exc:
            print(f"[YOLO] Failed to scan images at {input_path}: {exc}")
            total_images = 0

        with self.lock:
            self.yolo_task_state[task_id] = {
                "total_images": total_images,
                "remaining_images": total_images,
                "thread_name": thread.name,
                "thread_id": thread.ident,
                "process_id": None,
                "task_prefix": task_prefix,
                "batch_start_time": batch_start_time,
                "status": "queued",
            }
        return {
            "total_images": total_images,
            "thread_name": thread.name,
            "thread_id": thread.ident,
            "task_prefix": task_prefix,
            "batch_start_time": batch_start_time,
        }

    def _update_yolo_task_state(self, task_id, processed, total):
        remaining = max(total - processed, 0)
        self._update_yolo_task_meta(
            task_id, total_images=total, remaining_images=remaining
        )

    def _update_yolo_task_meta(self, task_id, **fields):
        with self.lock:
            current = self.yolo_task_state.get(task_id, {})
            current.update(fields)
            self.yolo_task_state[task_id] = current

    def _set_yolo_process_info(self, task_id, process_id):
        self._update_yolo_task_meta(task_id, process_id=process_id)

    def _register_yolo_process(self, task_id, process):
        with self.lock:
            self.yolo_processes[task_id] = process

    def _unregister_yolo_process(self, task_id):
        with self.lock:
            self.yolo_processes.pop(task_id, None)

    def _finalize_yolo_task_state(self, task_id):
        with self.lock:
            self.yolo_task_state.pop(task_id, None)

    def _run_yolo_task(
        self, task_id, on_started, on_finished, task_prefix, total_images, **kwargs
    ):
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
        sema_acquired = False
        try:
            if total_images <= 0:
                on_started(None)
                return
            if self.yolo_stop_event.is_set():
                on_started(None)
                return
            if self.yolo_process_sema:
                self.yolo_process_sema.acquire()
                sema_acquired = True
            if self.yolo_stop_event.is_set():
                on_started(None)
                return
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
                    "file_prefix": task_prefix,
                },
            )
            process.start()
            self._register_yolo_process(task_id, process)
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
                self._unregister_yolo_process(task_id)
            else:
                on_finished(None)
            if sema_acquired and self.yolo_process_sema:
                self.yolo_process_sema.release()

    def _stage_yolo_inputs(self, input_path, task_prefix, max_images):
        from yolo_workload import collect_images

        if not os.path.isdir(input_path):
            return 0

        images, _ = collect_images(input_path)
        unassigned = []
        prefixed = []
        for full_path, rel_path in images:
            if os.path.basename(rel_path).startswith(self.YOLO_TASK_PREFIX):
                prefixed.append((full_path, rel_path))
            else:
                unassigned.append((full_path, rel_path))

        reuse_prefixed = False
        if not unassigned:
            if not prefixed:
                print(f"[YOLO] No images available under {input_path}")
                return 0
            unassigned = prefixed
            reuse_prefixed = True

        if max_images is not None:
            max_images = int(max_images)
            if max_images <= 0:
                raise ValueError("max_images must be a positive integer")
            unassigned = unassigned[:max_images]

        assigned = 0
        for full_path, rel_path in unassigned:
            dir_path = os.path.dirname(full_path)
            base_name = os.path.basename(full_path)
            if reuse_prefixed:
                base_name = self._strip_yolo_task_prefix(base_name)
            new_name = f"{task_prefix}{base_name}"
            new_path = os.path.join(dir_path, new_name)
            if new_path == full_path:
                assigned += 1
                continue
            if os.path.exists(new_path):
                suffix = 1
                while os.path.exists(new_path):
                    new_name = f"{task_prefix}{suffix}_{base_name}"
                    new_path = os.path.join(dir_path, new_name)
                    suffix += 1
            try:
                os.replace(full_path, new_path)
                assigned += 1
            except OSError as exc:
                print(f"[YOLO] Failed to stage {rel_path}: {exc}")

        return assigned

    def _strip_yolo_task_prefix(self, name):
        if not name.startswith(self.YOLO_TASK_PREFIX):
            return name
        remainder = name[len(self.YOLO_TASK_PREFIX):]
        parts = remainder.split("_", 1)
        if len(parts) == 2 and parts[0]:
            return parts[1]
        return remainder or name

    def dispatch_task(self, task_type, duration=None, **kwargs):
        """派发任务到线程池"""
        t = threading.Thread(
            target=self._worker_wrapper, args=(task_type, duration), kwargs=kwargs
        )
        t.daemon = True
        with self.lock:
            self.worker_threads.append(t)
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
        self._wait_for_workers()

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

    def _wait_for_workers(self):
        if self.yolo_shutdown_timeout is None:
            return

        deadline = time.time() + max(self.yolo_shutdown_timeout, 0)
        while time.time() < deadline:
            with self.lock:
                alive_threads = [t for t in self.worker_threads if t.is_alive()]
            if not alive_threads:
                return
            for thread in alive_threads:
                thread.join(timeout=0.2)

        self._terminate_yolo_processes()
        with self.lock:
            alive_threads = [t.name for t in self.worker_threads if t.is_alive()]
        if alive_threads:
            print(f"[Warning] Worker threads still running: {alive_threads}")

    def _terminate_yolo_processes(self):
        with self.lock:
            processes = list(self.yolo_processes.items())
        for task_id, process in processes:
            if process.is_alive():
                print(f"[YOLO] Forcing stop of task {task_id} (PID: {process.pid})")
                process.terminate()
                process.join(timeout=2.0)
        with self.lock:
            self.yolo_processes.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASTRA scheduler with optional YOLO concurrency control"
    )
    parser.add_argument(
        "--yolo-max-concurrent",
        type=int,
        default=1,
        help="Maximum number of concurrent YOLO tasks (processes), fixed to 1",
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

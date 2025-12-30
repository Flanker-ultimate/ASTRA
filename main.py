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


def _strip_yolo_task_prefix(name, task_prefix_base):
    if not name.startswith(task_prefix_base):
        return name
    remainder = name[len(task_prefix_base):]
    parts = remainder.split("_", 1)
    if len(parts) == 2 and parts[0]:
        return parts[1]
    return remainder or name


def stage_yolo_inputs(input_path, task_prefix, max_images, task_prefix_base):
    from yolo_workload import collect_images

    if not os.path.isdir(input_path):
        return 0

    images, _ = collect_images(input_path)
    unassigned = []
    prefixed = []
    for full_path, rel_path in images:
        if os.path.basename(rel_path).startswith(task_prefix_base):
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
            base_name = _strip_yolo_task_prefix(base_name, task_prefix_base)
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


def yolo_worker_loop(task_queue, status_queue, stop_event, task_prefix_base):
    import queue as local_queue
    from yolo_workload import run_inference

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        try:
            task = task_queue.get(timeout=0.5)
        except local_queue.Empty:
            continue
        if task is None:
            break

        task_id = task.get("task_id")
        input_path = task.get("input_path")
        output_dir = task.get("output_dir")
        output_format = task.get("output_format")
        max_images = task.get("max_images")
        weights = task.get("weights")
        labels = task.get("labels")
        imgsz = task.get("imgsz")
        device = task.get("device", 0)
        conf_thres = task.get("conf_thres", 0.25)
        iou_thres = task.get("iou_thres", 0.45)
        max_det = task.get("max_det", 1000)
        agnostic_nms = task.get("agnostic_nms", False)
        verbose = task.get("verbose", True)
        task_prefix = task.get("task_prefix")

        try:
            total_images = stage_yolo_inputs(
                input_path, task_prefix, max_images, task_prefix_base
            )
            status_queue.put(
                (
                    "start",
                    task_id,
                    {"process_id": os.getpid(), "total_images": total_images},
                )
            )
            if total_images <= 0:
                status_queue.put(
                    ("done", task_id, {"processed": 0, "total_images": total_images})
                )
                continue

            def _callback(processed, total):
                status_queue.put(
                    ("progress", task_id, {"processed": processed, "total": total})
                )

            processed = run_inference(
                input_path=input_path,
                output_dir=output_dir,
                output_format=output_format,
                weights=weights,
                labels=labels,
                imgsz=imgsz,
                device=device,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                agnostic_nms=agnostic_nms,
                max_images=max_images,
                progress_callback=_callback,
                stop_event=stop_event,
                verbose=verbose,
                file_prefix=task_prefix,
            )
            status_queue.put(
                (
                    "done",
                    task_id,
                    {"processed": processed, "total_images": total_images},
                )
            )
        except Exception as exc:
            status_queue.put(("error", task_id, {"error": str(exc)}))

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
        self.yolo_task_queue = self.mp_context.Queue()
        self.yolo_status_queue = self.mp_context.Queue()
        self.yolo_worker_process = None
        if int(self.yolo_max_concurrent) < 1:
            raise ValueError("yolo_max_concurrent must be >= 1")
        if int(self.yolo_max_concurrent) != 1:
            raise ValueError("yolo_max_concurrent must be 1 in single-process mode")
        
        self.running = True

    def _monitor_loop(self):
        """后台监控线程"""
        while self.running:
            self._drain_yolo_status()
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

    def _drain_yolo_status(self):
        while True:
            try:
                msg_type, task_id, payload = self.yolo_status_queue.get_nowait()
            except py_queue.Empty:
                return

            state = self.yolo_task_state.get(task_id, {})
            if msg_type == "start":
                process_id = payload.get("process_id")
                total_images = payload.get("total_images", 0)
                status = "running" if total_images > 0 else "empty"
                if total_images > 0:
                    with self.lock:
                        self.active_tasks_count["YOLO"] += 1
                self._update_yolo_task_meta(
                    task_id,
                    process_id=process_id,
                    total_images=total_images,
                    remaining_images=total_images,
                    status=status,
                )
                details = {
                    "task_prefix": state.get("task_prefix"),
                    "task_start_time": state.get("batch_start_time"),
                    "total_images": total_images,
                }
                self.recorder.log_event("YOLO", "START", task_id, details)
                print(
                    f"[{time.strftime('%H:%M:%S')}] START Task: YOLO (ID: {task_id}, PID: {process_id})"
                )
            elif msg_type == "progress":
                processed = payload.get("processed", 0)
                total = payload.get("total", 0)
                self._update_yolo_task_state(task_id, processed, total)
            elif msg_type == "done":
                processed = payload.get("processed")
                total_images = payload.get("total_images")
                if state.get("status") == "running":
                    with self.lock:
                        if self.active_tasks_count["YOLO"] > 0:
                            self.active_tasks_count["YOLO"] -= 1
                details = {
                    "task_prefix": state.get("task_prefix"),
                    "task_start_time": state.get("batch_start_time"),
                }
                if total_images is not None:
                    details["total_images"] = total_images
                if processed is not None:
                    details["processed"] = processed
                self.recorder.log_event("YOLO", "END", task_id, details)
                print(
                    f"[{time.strftime('%H:%M:%S')}] END   Task: YOLO (ID: {task_id}, PID: {state.get('process_id')})"
                )
                self._finalize_yolo_task_state(task_id)
            elif msg_type == "error":
                error = payload.get("error", "Unknown error")
                print(f"[YOLO] Task {task_id} failed: {error}")
                if state.get("status") == "running":
                    with self.lock:
                        if self.active_tasks_count["YOLO"] > 0:
                            self.active_tasks_count["YOLO"] -= 1
                details = {
                    "task_prefix": state.get("task_prefix"),
                    "task_start_time": state.get("batch_start_time"),
                    "total_images": state.get("total_images"),
                    "error": error,
                }
                self.recorder.log_event("YOLO", "END", task_id, details)
                print(
                    f"[{time.strftime('%H:%M:%S')}] END   Task: YOLO (ID: {task_id}, PID: {state.get('process_id')})"
                )
                self._finalize_yolo_task_state(task_id)

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
            task_prefix = event_details.get("task_prefix")
            self._enqueue_yolo_task(task_id, task_prefix, **kwargs)

    def _init_yolo_task_state(self, task_id, **kwargs):
        input_path = kwargs.get("input_path", self.yolo_input_path)
        max_images = kwargs.get("max_images", self.yolo_max_images)
        batch_start_time = time.time()
        task_prefix = f"{self.YOLO_TASK_PREFIX}{task_id}_"
        thread = threading.current_thread()
        try:
            from yolo_workload import collect_images

            total_images = 0
            images, _ = collect_images(input_path)
            unassigned = []
            prefixed = []
            for _, rel_path in images:
                if os.path.basename(rel_path).startswith(self.YOLO_TASK_PREFIX):
                    prefixed.append(rel_path)
                else:
                    unassigned.append(rel_path)
            if unassigned:
                total_images = len(unassigned)
            else:
                total_images = len(prefixed)
            if max_images is not None:
                max_images = int(max_images)
                if max_images <= 0:
                    raise ValueError("max_images must be a positive integer")
                total_images = min(total_images, max_images)
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
            "task_prefix": task_prefix,
            "total_images": total_images,
            "task_start_time": batch_start_time,
        }

    def _enqueue_yolo_task(self, task_id, task_prefix, **kwargs):
        task = {
            "task_id": task_id,
            "task_prefix": task_prefix,
            "input_path": kwargs.get("input_path", self.yolo_input_path),
            "output_dir": kwargs.get("output_dir", self.yolo_output_dir),
            "output_format": kwargs.get("output_format", self.yolo_output_format),
            "max_images": kwargs.get("max_images", self.yolo_max_images),
            "weights": kwargs.get("weights"),
            "labels": kwargs.get("labels"),
            "imgsz": kwargs.get("imgsz", (640, 640)),
            "device": kwargs.get("device", 0),
            "conf_thres": kwargs.get("conf_thres", 0.25),
            "iou_thres": kwargs.get("iou_thres", 0.45),
            "max_det": kwargs.get("max_det", 1000),
            "agnostic_nms": kwargs.get("agnostic_nms", False),
            "verbose": kwargs.get("verbose", self.yolo_verbose),
        }
        self._ensure_yolo_worker()
        self.yolo_task_queue.put(task)

    def _ensure_yolo_worker(self):
        if self.yolo_worker_process is not None:
            if self.yolo_worker_process.is_alive():
                return
        self.yolo_worker_process = self.mp_context.Process(
            target=yolo_worker_loop,
            kwargs={
                "task_queue": self.yolo_task_queue,
                "status_queue": self.yolo_status_queue,
                "stop_event": self.yolo_stop_event,
                "task_prefix_base": self.YOLO_TASK_PREFIX,
            },
        )
        self.yolo_worker_process.start()

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

    def _finalize_yolo_task_state(self, task_id):
        with self.lock:
            self.yolo_task_state.pop(task_id, None)

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
        self._ensure_yolo_worker()
        
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
        
        self._stop_yolo_tasks_after_grace()
        self._wait_for_workers()
        # 生成最终数据集
        self.recorder.save_dataset()

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
            self._cancel_pending_yolo_tasks()

    def _cancel_pending_yolo_tasks(self):
        pending = []
        with self.lock:
            for task_id, info in self.yolo_task_state.items():
                if info.get("status") == "queued":
                    pending.append((task_id, dict(info)))

        for task_id, info in pending:
            details = {
                "task_prefix": info.get("task_prefix"),
                "task_start_time": info.get("batch_start_time"),
                "total_images": info.get("total_images"),
                "processed": 0,
                "error": "cancelled",
            }
            self.recorder.log_event("YOLO", "END", task_id, details)
            self._finalize_yolo_task_state(task_id)

        while True:
            try:
                task = self.yolo_task_queue.get_nowait()
            except py_queue.Empty:
                break
            if task is None:
                continue

    def _wait_for_workers(self):
        if self.yolo_shutdown_timeout is None:
            return

        deadline = time.time() + max(self.yolo_shutdown_timeout, 0)
        while time.time() < deadline:
            self._drain_yolo_status()
            with self.lock:
                alive_threads = [t for t in self.worker_threads if t.is_alive()]
            if not alive_threads:
                break
            for thread in alive_threads:
                thread.join(timeout=0.2)

        self._terminate_yolo_worker()
        self._drain_yolo_status()
        with self.lock:
            alive_threads = [t.name for t in self.worker_threads if t.is_alive()]
        if alive_threads:
            print(f"[Warning] Worker threads still running: {alive_threads}")

    def _terminate_yolo_worker(self):
        if self.yolo_worker_process is None:
            return
        if self.yolo_worker_process.is_alive():
            self.yolo_stop_event.set()
            try:
                self.yolo_task_queue.put_nowait(None)
            except py_queue.Full:
                pass
            self.yolo_worker_process.join(timeout=2.0)
            if self.yolo_worker_process.is_alive():
                print(
                    f"[YOLO] Forcing stop of worker (PID: {self.yolo_worker_process.pid})"
                )
                self.yolo_worker_process.terminate()
                self.yolo_worker_process.join(timeout=2.0)
        self.yolo_worker_process = None

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

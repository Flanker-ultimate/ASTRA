import time
import uuid
import threading
import random
from monitor import HardwareMonitor
from workloads import WorkloadExecutor
from recorder import DataRecorder

class AstraController:
    def __init__(self, simulation_mode=True):
        self.recorder = DataRecorder()
        self.monitor = HardwareMonitor(use_simulation=simulation_mode)
        
        # 线程安全锁，用于统计当前活跃任务数供 Monitor 使用
        self.lock = threading.Lock()
        self.active_tasks_count = {'IO': 0, 'NET': 0, 'YOLO': 0}
        
        self.running = True

    def _monitor_loop(self):
        """后台监控线程"""
        while self.running:
            # 1. 获取当前活跃任务快照
            with self.lock:
                snapshot = self.active_tasks_count.copy()
            
            # 2. 采集硬件数据
            metrics = self.monitor.get_metrics(snapshot)
            
            # 3. 记录
            self.recorder.log_metric(metrics)
            
            # 4. 采样频率 2Hz (每0.5秒)
            time.sleep(0.5)

    def _worker_wrapper(self, task_type, duration):
        """任务执行包装器 (处理日志和计数)"""
        task_id = str(uuid.uuid4())[:8]
        
        # START Event
        with self.lock:
            self.active_tasks_count[task_type] += 1
        self.recorder.log_event(task_type, "START", task_id)
        print(f"[{time.strftime('%H:%M:%S')}] START Task: {task_type} (ID: {task_id})")
        
        # Execution
        if task_type == 'IO':
            WorkloadExecutor.task_io_stress(duration)
        elif task_type == 'NET':
            WorkloadExecutor.task_network_stress(duration)
        elif task_type == 'YOLO':
            WorkloadExecutor.task_yolo_inference(duration)
            
        # END Event
        with self.lock:
            self.active_tasks_count[task_type] -= 1
        self.recorder.log_event(task_type, "END", task_id)
        print(f"[{time.strftime('%H:%M:%S')}] END   Task: {task_type} (ID: {task_id})")

    def dispatch_task(self, task_type, duration):
        """派发任务到线程池"""
        t = threading.Thread(target=self._worker_wrapper, args=(task_type, duration))
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
                self.dispatch_task("YOLO", duration)
                
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
    app = AstraController(simulation_mode=True)
    app.run_simulation(total_time=20) # 运行20秒测试
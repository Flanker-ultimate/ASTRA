import pandas as pd
import queue
import time
import threading

class DataRecorder:
    def __init__(self):
        self.metric_queue = queue.Queue()
        self.event_queue = queue.Queue()
        self.is_running = True
        
    def log_metric(self, metric_dict):
        self.metric_queue.put(metric_dict)
        
    def log_event(self, task_type, action, task_id):
        self.event_queue.put({
            "timestamp": time.time(),
            "task_id": task_id,
            "task_type": task_type,
            "action": action
        })

    def save_dataset(self, filename_prefix="ASTRA_Dataset"):
        print("\n[Recorder] Processing data and generating dataset...")
        
        # 1. 导出 Metrics
        metrics = []
        while not self.metric_queue.empty():
            metrics.append(self.metric_queue.get())
        df_metrics = pd.DataFrame(metrics)
        
        # 2. 导出 Events
        events = []
        while not self.event_queue.empty():
            events.append(self.event_queue.get())
        df_events = pd.DataFrame(events)
        
        if df_metrics.empty:
            print("[Error] No metrics collected.")
            return

        # 3. 数据对齐 (Data Alignment)
        # 为 Metrics 表添加标签列 (Y值)
        df_metrics = df_metrics.sort_values("timestamp")
        df_metrics["active_io"] = 0
        df_metrics["active_net"] = 0
        df_metrics["active_yolo"] = 0
        
        # 使用 Pandas 的 merge_asof 或者遍历回放来标记
        # 这里使用“事件回放法”确保准确性
        if not df_events.empty:
            df_events = df_events.sort_values("timestamp")
            
            for i, row in df_metrics.iterrows():
                current_ts = row["timestamp"]
                # 找到当前时刻之前的所有事件
                past_events = df_events[df_events["timestamp"] <= current_ts]
                
                # 重建状态
                active_tasks = {}
                for _, event in past_events.iterrows():
                    tid = event["task_id"]
                    if event["action"] == "START":
                        active_tasks[tid] = event["task_type"]
                    elif event["action"] == "END":
                        if tid in active_tasks:
                            del active_tasks[tid]
                
                # 统计
                counts = {"IO": 0, "NET": 0, "YOLO": 0}
                for ttype in active_tasks.values():
                    counts[ttype] += 1
                
                df_metrics.at[i, "active_io"] = counts["IO"]
                df_metrics.at[i, "active_net"] = counts["NET"]
                df_metrics.at[i, "active_yolo"] = counts["YOLO"]

        # 4. 保存文件
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{filename_prefix}_{ts_str}.csv"
        df_metrics.to_csv(fname, index=False)
        print(f"[Recorder] Dataset saved successfully: {fname}")
        return df_metrics
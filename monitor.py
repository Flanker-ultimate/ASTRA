import time
import subprocess
import json
import psutil
import random
# 引入新写的网络监控类
from network_monitor import NetworkMonitor 

class HardwareMonitor:
    def __init__(self, use_simulation=True, device_id=0):
        self.use_simulation = use_simulation
        self.device_id = device_id
        
        # 初始化网络监控 (默认 Ping 8.8.8.8，可改为网关或其他内网IP)
        self.net_monitor = NetworkMonitor(target_host="8.8.8.8")
        
    def _parse_ascend_dmi_json(self, json_output, target_device_id):
        # ... (保持之前的解析代码不变) ...
        # 为了节省篇幅，此处省略，请保留您之前完善的解析逻辑
        try:
            data = json.loads(json_output)
            cards = data.get("hardware_details", {}).get("cards", [])
            target_device = None
            for card in cards:
                for device in card.get("devices", []):
                    if int(device.get("device_id", -1)) == target_device_id:
                        target_device = device
                        break
                if target_device: break
            
            metrics = {
                "npu_util": 0, "npu_ai_cpu": 0, "npu_temp": 0.0,
                "npu_mem_used": 0, "npu_mem_total": 0, "npu_mem_util": 0.0
            }
            if not target_device: return metrics

            metrics["npu_util"] = int(target_device.get("ai_core_information", {}).get("ai_core_usage", "0%").strip('%'))
            metrics["npu_ai_cpu"] = int(target_device.get("cpu_information", {}).get("ai_cpu_usage", "0%").strip('%'))
            
            mem_info = target_device.get("memory_information", {})
            metrics["npu_mem_used"] = int(mem_info.get("used", "0 MB").split()[0])
            metrics["npu_mem_total"] = int(mem_info.get("total", "0 MB").split()[0])
            
            if metrics["npu_mem_total"] > 0:
                metrics["npu_mem_util"] = round((metrics["npu_mem_used"] / metrics["npu_mem_total"]) * 100, 1)
            
            metrics["npu_temp"] = float(target_device.get("temperature", "0 C").split()[0])
            return metrics
        except Exception:
            return {"npu_util": 0, "npu_ai_cpu": 0, "npu_temp": 0.0, "npu_mem_used": 0, "npu_mem_total": 0, "npu_mem_util": 0.0}

    def _get_npu_real(self):
        # ... (保持之前代码不变) ...
        try:
            cmd = ["ascend-dmi", "-i", "--dt", "--fmt", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if not result.stdout.strip(): return self._parse_ascend_dmi_json("{}", self.device_id)
            return self._parse_ascend_dmi_json(result.stdout, self.device_id)
        except Exception:
            return self._parse_ascend_dmi_json("{}", self.device_id)

    def _get_npu_sim(self, active_yolo_count):
        # ... (保持之前代码不变) ...
        base_noise = random.uniform(0, 5)
        util = min(100, active_yolo_count * 35 + base_noise)
        mem_total = 7759 
        mem_used = min(712 + active_yolo_count * 600, mem_total)
        mem_util = round((mem_used / mem_total) * 100, 1)
        temp = 34.0 + util * 0.2
        return {
            "npu_util": round(util, 1), "npu_ai_cpu": round(random.uniform(0, 5) + active_yolo_count * 2, 1),
            "npu_mem_used": int(mem_used), "npu_mem_total": int(mem_total),
            "npu_mem_util": mem_util, "npu_temp": round(temp, 1)
        }

    def get_metrics(self, active_tasks_snapshot):
        """
        获取全系统指标 (Host + Network + NPU)
        """
        # 1. Host 基础指标
        vm = psutil.virtual_memory()
        host_mem_total = vm.total // (1024 * 1024)
        host_mem_used = vm.used // (1024 * 1024)
        host_mem_util = vm.percent
        cpu_util = psutil.cpu_percent(interval=None)
        
        # 2. Network 指标 (新增)
        # 注意：get_bandwidth 依赖于调用间隔来计算速率
        up_speed, down_speed = self.net_monitor.get_bandwidth() # KB/s
        latency = self.net_monitor.get_latency() # ms
        
        # 3. NPU 指标
        if self.use_simulation:
            npu_data = self._get_npu_sim(active_tasks_snapshot.get('YOLO', 0))
        else:
            npu_data = self._get_npu_real()

        return {
            "timestamp": time.time(),
            
            # Host
            "cpu_util": cpu_util,
            "host_mem_util": host_mem_util,
            "host_mem_used": host_mem_used,
            "host_mem_total": host_mem_total,
            
            # Network (New)
            "net_up_kb": up_speed,
            "net_down_kb": down_speed,
            "net_latency": latency,
            
            # NPU
            "npu_util": npu_data['npu_util'],
            "npu_ai_cpu": npu_data['npu_ai_cpu'],
            "npu_temp": npu_data['npu_temp'],
            "npu_mem_util": npu_data['npu_mem_util'],
            "npu_mem_used": npu_data['npu_mem_used'],
            "npu_mem_total": npu_data['npu_mem_total']
        }
import time
import psutil
import subprocess
import re
import platform

class NetworkMonitor:
    def __init__(self, target_host="8.8.8.8"):
        self.target_host = target_host
        # 初始化流量计数器
        self.last_io = psutil.net_io_counters()
        self.last_time = time.time()

    def get_latency(self, timeout=1):
        """
        通过 Ping 命令获取到目标主机的延迟 (ms)
        """
        # 根据系统选择 ping 命令参数
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        timeout_param = '-w' if platform.system().lower() == 'windows' else '-W'
        
        command = ['ping', param, '1', timeout_param, str(timeout), self.target_host]
        
        try:
            # 执行命令
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                # 使用正则提取时间 (支持 Windows "time=1ms" 和 Linux "time=1.23 ms")
                match = re.search(r'time[=<]\s*([\d\.]+)\s*ms', result.stdout, re.IGNORECASE)
                if match:
                    return float(match.group(1))
            return -1.0 # 超时或不可达
        except Exception:
            return -1.0

    def get_bandwidth(self):
        """
        计算自上次调用以来的上传/下载速度 (KB/s)
        """
        current_io = psutil.net_io_counters()
        current_time = time.time()
        
        # 计算时间差
        dt = current_time - self.last_time
        if dt <= 0:
            return 0.0, 0.0
            
        # 计算流量差 (Bytes)
        sent_bytes = current_io.bytes_sent - self.last_io.bytes_sent
        recv_bytes = current_io.bytes_recv - self.last_io.bytes_recv
        
        # 转换为 KB/s
        upload_speed = (sent_bytes / 1024) / dt
        download_speed = (recv_bytes / 1024) / dt
        
        # 更新状态
        self.last_io = current_io
        self.last_time = current_time
        
        return round(upload_speed, 1), round(download_speed, 1)
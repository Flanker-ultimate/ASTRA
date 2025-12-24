import time
import os
import random
import socket
import numpy as np

# 模拟的大文件路径
TEMP_FILE = "satellite_temp_data.bin"

class WorkloadExecutor:
    """具体的任务执行逻辑"""
    
    @staticmethod
    def task_io_stress(duration):
        """IO 读写压力测试"""
        start_time = time.time()
        # 创建一个 50MB 的数据块
        data = os.urandom(1024 * 1024 * 50) 
        
        while time.time() - start_time < duration:
            # 写入
            with open(TEMP_FILE, "wb") as f:
                f.write(data)
            # 读取
            with open(TEMP_FILE, "rb") as f:
                _ = f.read()
        
        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)

    @staticmethod
    def task_network_stress(duration):
        """网络传输模拟 (Client模式)"""
        # 为了演示，我们模拟向一个不存在的地址发包，或者向本地回环发包
        # 这里使用计算密集型的 sleep 模拟网络阻塞
        start_time = time.time()
        while time.time() - start_time < duration:
            # 模拟封包解包的CPU消耗
            _ = [random.random() for _ in range(1000)]
            time.sleep(0.01) # 模拟网络延迟

    @staticmethod
    def task_yolo_inference(duration):
        """YOLO 推理模拟 (或者真实调用 ACL)"""
        start_time = time.time()

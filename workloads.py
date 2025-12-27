import time
import os
import random
import threading
import sys
import subprocess
import importlib.util
import shutil

# --- HTTP 依赖 ---
from flask import Flask, request
import requests
import logging

# --- gRPC 依赖 ---
import grpc
from concurrent import futures

# 屏蔽 Flask 和 gRPC 的繁杂日志
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ==========================================
# 1. 动态 gRPC 环境构建器 (Auto-Magic)
# ==========================================
class GrpcBuilder:
    """自动生成并编译 .proto 文件，让脚本即插即用"""
    PROTO_FILE = "satellite_net.proto"
    
    @staticmethod
    def setup():
        # 1. 定义 Proto 内容
        proto_source = """
syntax = "proto3";
package satellite;

service DataLink {
  rpc TransferImage (ImageChunk) returns (TransferAck) {}
}

message ImageChunk {
  bytes content = 1;
  string image_id = 2;
  int64 timestamp = 3;
}

message TransferAck {
  bool success = 1;
  string message = 2;
}
"""
        # 2. 写入文件
        if not os.path.exists(GrpcBuilder.PROTO_FILE):
            with open(GrpcBuilder.PROTO_FILE, "w") as f:
                f.write(proto_source)
        
        # 3. 编译 Proto (相当于运行 python -m grpc_tools.protoc)
        # 生成 satellite_net_pb2.py 和 satellite_net_pb2_grpc.py
        import grpc_tools.protoc
        grpc_tools.protoc.main([
            'grpc_tools.protoc',
            '-I.',
            '--python_out=.',
            '--grpc_python_out=.',
            GrpcBuilder.PROTO_FILE
        ])
        
        # 4. 动态导入生成的模块
        # 这也是为了避免 import 错误
        spec_pb2 = importlib.util.spec_from_file_location("satellite_net_pb2", "satellite_net_pb2.py")
        pb2 = importlib.util.module_from_spec(spec_pb2)
        spec_pb2.loader.exec_module(pb2)
        
        spec_grpc = importlib.util.spec_from_file_location("satellite_net_pb2_grpc", "satellite_net_pb2_grpc.py")
        pb2_grpc = importlib.util.module_from_spec(spec_grpc)
        spec_grpc.loader.exec_module(pb2_grpc)
        
        return pb2, pb2_grpc

# 初始化 gRPC 模块
try:
    pb2, pb2_grpc = GrpcBuilder.setup()
except Exception as e:
    print(f"[Warning] gRPC setup failed: {e}. gRPC mode will not work.")
    pb2, pb2_grpc = None, None


# ==========================================
# 2. 核心工作负载类
# ==========================================

class WorkloadExecutor:
    # 定义模拟的大文件路径（项目内 tmp 目录）
    BASE_TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    IO_TMP_DIR = os.path.join(BASE_TMP_DIR, "io_workload")
    TEMP_FILE = os.path.join(IO_TMP_DIR, "satellite_temp_data.bin")
    
    @staticmethod
    def task_io_stress(duration):
        """IO 读写压力测试 (保持不变)"""
        start_time = time.time()
        # 50MB 数据
        data = os.urandom(1024 * 1024 * 50) 
        try:
            os.makedirs(WorkloadExecutor.IO_TMP_DIR, exist_ok=True)
            while time.time() - start_time < duration:
                with open(WorkloadExecutor.TEMP_FILE, "wb") as f:
                    f.write(data)
                with open(WorkloadExecutor.TEMP_FILE, "rb") as f:
                    _ = f.read()
        finally:
            if os.path.exists(WorkloadExecutor.TEMP_FILE):
                os.remove(WorkloadExecutor.TEMP_FILE)

    @staticmethod
    def task_yolo_inference(duration):
        """YOLO 推理模拟 (保持不变)"""
        start_time = time.time()
        # 模拟加载
        time.sleep(0.5)
        while time.time() - start_time < duration:
            # 模拟 CPU 预处理
            _ = [random.random() for _ in range(10000)]
            # 模拟 NPU 推理耗时
            time.sleep(0.05)

    @staticmethod
    def task_yolo_inference_ascend(
        input_path,
        output_dir,
        output_format="all",
        weights=None,
        labels=None,
        imgsz=(640, 640),
        device=0,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        agnostic_nms=False,
        max_images=None,
        progress_callback=None,
        stop_event=None,
        verbose=True,
    ):
        """YOLO inference workload using Ascend NPU."""
        from yolo_workload import run_inference

        return run_inference(
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
            progress_callback=progress_callback,
            stop_event=stop_event,
            verbose=verbose,
        )

    # ==========================================
    # 3. 全新的网络传输模拟 (HTTP & gRPC)
    # ==========================================
    
    @staticmethod
    def task_network_stress(duration, protocol='http', target_bandwidth_mb=20):
        """
        模拟卫星网络传输
        :param protocol: 'http' 或 'grpc'
        :param target_bandwidth_mb: 限制带宽 (MB/s)，模拟受限链路
        """
        host = '127.0.0.1'
        # 随机分配端口避免冲突
        port = random.randint(20000, 30000) 
        
        # 生成随机图片数据 (模拟 256KB 的图片切片)
        chunk_size = 256 * 1024 
        payload = os.urandom(chunk_size)
        
        server_stop_event = threading.Event()
        
        # --- 启动服务端 ---
        if protocol == 'grpc':
            if pb2 is None: return
            t_server = threading.Thread(
                target=WorkloadExecutor._run_grpc_server, 
                args=(port, server_stop_event)
            )
        else:
            t_server = threading.Thread(
                target=WorkloadExecutor._run_http_server, 
                args=(port, server_stop_event)
            )
            
        t_server.start()
        time.sleep(1) # 等待 Server 就绪
        
        # --- 启动客户端发送循环 ---
        try:
            start_time = time.time()
            total_bytes = 0
            
            # gRPC Client 初始化
            channel = None
            stub = None
            if protocol == 'grpc':
                channel = grpc.insecure_channel(f'{host}:{port}')
                stub = pb2_grpc.DataLinkStub(channel)
            
            while time.time() - start_time < duration:
                step_start = time.time()
                
                # 1. 发送请求
                if protocol == 'grpc':
                    # gRPC 发送
                    req = pb2.ImageChunk(
                        content=payload, 
                        image_id="img_001", 
                        timestamp=int(time.time())
                    )
                    stub.TransferImage(req)
                else:
                    # HTTP POST 发送
                    requests.post(
                        f'http://{host}:{port}/upload', 
                        data=payload,
                        headers={'Content-Type': 'application/octet-stream'}
                    )
                
                total_bytes += chunk_size
                
                # 2. 带宽控制 (Token Bucket 简易版)
                if target_bandwidth_mb:
                    expected_time = total_bytes / (target_bandwidth_mb * 1024 * 1024)
                    actual_time = time.time() - start_time
                    if actual_time < expected_time:
                        sleep_time = expected_time - actual_time
                        if sleep_time > 0:
                            time.sleep(sleep_time)

        except Exception as e:
            print(f"[Net Error] {e}")
        finally:
            server_stop_event.set()
            # 触发一次请求以打破 server 的阻塞 (针对 HTTP/gRPC 的 graceful shutdown 较复杂，这里简单处理)
            try:
                if protocol == 'http':
                    requests.get(f'http://{host}:{port}/shutdown', timeout=0.1)
            except: pass
            
            if protocol == 'grpc' and channel:
                channel.close()

    # --- HTTP Server 实现 (Flask) ---
    @staticmethod
    def _run_http_server(port, stop_event):
        app = Flask(__name__)
        
        @app.route('/upload', methods=['POST'])
        def upload():
            # 接收数据但不处理，模拟“黑洞”服务器
            _ = request.data 
            return "OK", 200

        @app.route('/shutdown', methods=['GET'])
        def shutdown():
            shutdown_func = request.environ.get('werkzeug.server.shutdown')
            if shutdown_func:
                shutdown_func()
                return 'Server shutting down...'
            return 'Shutdown not supported', 200

        # 运行在线程中
        try:
            app.run(host='127.0.0.1', port=port, threaded=True, use_reloader=False)
        except:
            pass

    # --- gRPC Server 实现 ---
    @staticmethod
    def _run_grpc_server(port, stop_event):
        class DataLinkServicer(pb2_grpc.DataLinkServicer):
            def TransferImage(self, request, context):
                # 接收到数据，返回 ACK
                return pb2.TransferAck(success=True, message="Received")

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb2_grpc.add_DataLinkServicer_to_server(DataLinkServicer(), server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        
        # 等待停止信号
        while not stop_event.is_set():
            time.sleep(0.5)
        server.stop(0)

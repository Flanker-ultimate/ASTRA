# ASTRA (Ascend Satellite Telemetry & Resource Analyzer)
```
ASTRA/
│
├── main.py              # 程序入口，负责调度和控制
├── monitor.py           # 负责 NPU/CPU 内存 网络 资源监控
├── network_monitor.py   # 网络监控工具
├── workloads.py         # 定义 I/O, 网络, YOLO 等具体任务
├── recorder.py          # 负责日志记录和数据集生成
├── yolo_workload.py     # Ascend YOLOv5 推理负载入口
└── utils.py             # 工具函数
```

## 实现思路与方法

- 监控采样：`HardwareMonitor` 以固定频率采集 CPU/内存/网络/NPU 指标，写入 `recorder` 队列。
- 任务调度：`AstraController` 按概率调度 IO/NET/YOLO 任务，YOLO 任务采用独立进程执行推理，避免 ACL 重入冲突。
- 任务记录：通过事件回放计算 `active_io/active_net/active_yolo`，并在每个时间戳记录运行中的 YOLO 任务（含总图片数、剩余图片数、线程/进程信息）。
- 推理负载：`yolo_workload.py` 复用 `yolov5-ascend` 的推理流程，支持输入路径、输出路径与最多推理图片数。

## Ascend YOLO workload

Use `yolo_workload.py` to run YOLOv5 inference on Ascend NPU for a given image file or directory.

```bash
python yolo_workload.py \
  --input /path/to/images \
  --output-dir /path/to/output \
  --output-format all
```

## 启动命令

1) 运行 ASTRA 调度（会随机触发 IO/NET/YOLO 负载）：
```bash
python main.py
```

2) 仅运行 YOLO 推理负载：
```bash
python yolo_workload.py --input /path/to/images --output-dir /path/to/output --output-format all
```

# ASTRA (Ascend Satellite Telemetry & Resource Analyzer)
```
ASTRA/
│
├── main.py              # 程序入口，负责调度和控制
├── monitor.py           # 负责 NPU/CPU 内存 网络 资源监控
├── network_monitor.py   # 网络监控工具
├── workloads.py         # 定义 I/O, 网络, YOLO 等具体任务
├── recorder.py          # 负责日志记录和数据集生成
└── utils.py             # 工具函数
```

## Ascend YOLO workload

Use `yolo_workload.py` to run YOLOv5 inference on Ascend NPU for a given image file or directory.

```bash
python yolo_workload.py \
  --input /path/to/images \
  --output-dir /path/to/output \
  --output-format all
```

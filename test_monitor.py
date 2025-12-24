import time
from monitor import HardwareMonitor

def test_monitor():
    print("初始化全系统监控器...")
    # 记得将 target_host 改为您网络中可达的 IP，例如网关或百度
    monitor = HardwareMonitor(use_simulation=False, device_id=0)
    # 修改 Ping 目标测试
    monitor.net_monitor.target_host = "8.8.8.8" 
    
    print("开始监控 (按 Ctrl+C 停止)...")
    
    # 表头设计
    header = (
        f"{'时间':<9} | "
        f"{'CPU%':<5} | {'MEM%':<5} | "
        f"{'NPU%':<5} | {'显存%':<5} | "
        f"{'Ping':<6} | {'Up(KB/s)':<9} | {'Down(KB/s)'}"
    )
    
    print("-" * 90)
    print(header)
    print("-" * 90)
    
    try:
        while True:
            # 获取数据
            metrics = monitor.get_metrics({})
            
            # 格式化时间
            ts_str = time.strftime("%H:%M:%S", time.localtime(metrics['timestamp']))
            
            # 格式化Ping (超时显示 -1)
            ping_str = f"{metrics['net_latency']:.1f}" if metrics['net_latency'] >= 0 else "N/A"
            
            print(
                f"{ts_str:<9} | "
                f"{metrics['cpu_util']:<5} | {metrics['host_mem_util']:<5} | "
                f"{metrics['npu_util']:<5} | {metrics['npu_mem_util']:<5} | "
                f"{ping_str:<6} | {metrics['net_up_kb']:<9} | {metrics['net_down_kb']}"
            )
            
            # 保持 1 秒间隔，方便带宽计算准确
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n监控停止。")

if __name__ == "__main__":
    test_monitor()
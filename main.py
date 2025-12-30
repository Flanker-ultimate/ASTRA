import argparse

from controller import AstraController


def main():
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
    parser.add_argument(
        "--yolo-queue-maxsize",
        type=int,
        default=0,
        help="Maximum number of pending YOLO tasks in the queue (0 = unlimited)",
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
        yolo_queue_maxsize=args.yolo_queue_maxsize,
    )
    app.run_simulation(total_time=args.total_time)


if __name__ == "__main__":
    main()

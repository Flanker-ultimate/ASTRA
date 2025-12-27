#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
YOLO_ROOT = ROOT / "yolov5-ascend"
DEFAULT_WEIGHTS = YOLO_ROOT / "ascend" / "yolov5s.om"
DEFAULT_LABELS = YOLO_ROOT / "ascend" / "yolov5.label"
if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))

import acl  # type: ignore

from acl_net import Net  # type: ignore
from constant import IMG_EXT  # type: ignore
from detect_yolov5_ascend import (  # type: ignore
    check_ret,
    load_label,
    process_image,
)


def _collect_images(input_path):
    path = Path(input_path)
    if path.is_file():
        return [(str(path), path.name)], path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    image_exts = {ext.lower() for ext in IMG_EXT}
    images = []
    for root, _, files in os.walk(path):
        for name in files:
            if Path(name).suffix.lower() in image_exts:
                full_path = Path(root) / name
                rel_path = full_path.relative_to(path)
                images.append((str(full_path), str(rel_path)))
    return images, path


def collect_images(input_path, max_images=None):
    images, input_root = _collect_images(input_path)
    if max_images is not None:
        max_images = int(max_images)
        if max_images <= 0:
            raise ValueError("max_images must be a positive integer")
        images = images[:max_images]
    return images, input_root


def run_inference(
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
    if output_format not in {"label", "image", "all"}:
        raise ValueError(f"Unsupported output_format: {output_format}")

    images, input_root = collect_images(input_path, max_images=max_images)
    if not images:
        print(f"No images found under: {input_path}")
        return 0
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if labels is None:
        labels = DEFAULT_LABELS
    if not Path(weights).exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not Path(labels).exists():
        raise FileNotFoundError(f"Label file not found: {labels}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acl_inited = False
    net = None
    try:
        ret = acl.init()
        check_ret("acl.init", ret)
        acl_inited = True

        net = Net(device, str(weights))
        label_list = load_label(str(labels))
        input_size = tuple(int(x) for x in imgsz)

        opt = SimpleNamespace(output_format=output_format)
        total_images = len(images)
        processed = 0
        if progress_callback is not None:
            progress_callback(0, total_images)
        for image_path, rel_path in images:
            if stop_event is not None and stop_event.is_set():
                print("Stop requested, ending inference early.")
                break
            result, elapsed = process_image(
                image_path,
                rel_path,
                net,
                label_list,
                input_size,
                conf_thres,
                iou_thres,
                agnostic_nms,
                max_det,
                None,
                opt,
                str(input_root),
                str(output_dir),
                verbose=verbose,
            )
            processed += 1
            if verbose:
                summary = result if result else "No detections"
                print(
                    f"[{processed}/{len(images)}] {rel_path} -> {summary} ({elapsed:.3f}s)"
                )
            if progress_callback is not None:
                progress_callback(processed, total_images)
        return processed
    finally:
        if net is not None:
            net.destroy()
        elif acl_inited:
            acl.finalize()


def run_inference_worker(
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
    progress_queue=None,
    stop_event=None,
    verbose=True,
):
    def _callback(processed, total):
        if progress_queue is not None:
            progress_queue.put(("progress", processed, total))

    try:
        processed = run_inference(
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
            progress_callback=_callback,
            stop_event=stop_event,
            verbose=verbose,
        )
        if progress_queue is not None:
            progress_queue.put(("done", processed, None))
    except Exception as exc:
        if progress_queue is not None:
            progress_queue.put(("error", str(exc), None))


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv5 inference workload on Huawei Ascend NPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="/home/ubuntu/data/test",
        help="Input image file or directory",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/yolo_workload",
        help="Output directory for inference results",
    )
    parser.add_argument(
        "--output-format",
        choices=["label", "image", "all"],
        default="all",
        help="Output format for results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image logs",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to model weights (.om file)",
    )
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS),
        help="Path to label file",
    )
    parser.add_argument(
        "--imgsz",
        nargs=2,
        type=int,
        default=[640, 640],
        help="Inference size [height width]",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="NPU device ID",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=1000,
        help="Maximum detections per image",
    )
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Class-agnostic NMS",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to run inference on",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(
        input_path=args.input,
        output_dir=args.output_dir,
        output_format=args.output_format,
        weights=args.weights,
        labels=args.labels,
        imgsz=args.imgsz,
        device=args.device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        max_images=args.max_images,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

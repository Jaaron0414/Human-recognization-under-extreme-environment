"""
YOLOv8 → TensorRT 导出脚本
在 Jetson Orin NX Super 上运行，可将推理速度提升 3-5x

用法
----
# 导出 FP16 精度（推荐，精度损失极小，速度最快）
python3 export_tensorrt.py --weights yolov8s.pt --precision fp16

# 导出 INT8 精度（速度更快，需要校准数据）
python3 export_tensorrt.py --weights yolov8s.pt --precision int8 --calib-data ./calib_images/

# 导出后测试速度
python3 export_tensorrt.py --weights yolov8s.pt --benchmark
"""

import argparse
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 TensorRT 导出工具")
    parser.add_argument("--weights", type=str, default="yolov8s.pt",
                        help="YOLOv8 PyTorch 权重路径")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp32", "fp16", "int8"],
                        help="量化精度（推荐 fp16）")
    parser.add_argument("--img-size", type=int, default=640,
                        help="推理分辨率")
    parser.add_argument("--workspace", type=int, default=4,
                        help="TensorRT 工作空间（GB）")
    parser.add_argument("--calib-data", type=str, default=None,
                        help="INT8 校准图像目录（精度模式为 int8 时必填）")
    parser.add_argument("--benchmark", action="store_true",
                        help="导出后进行速度基准测试")
    parser.add_argument("--benchmark-runs", type=int, default=200,
                        help="基准测试推理次数")
    return parser.parse_args()


def export_tensorrt(args) -> Path:
    """将 YOLOv8 .pt 权重导出为 TensorRT .engine 文件。"""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请先安装 ultralytics：pip3 install ultralytics")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在：{weights_path}")

    print(f"[Export] 加载模型：{weights_path}")
    model = YOLO(str(weights_path))

    engine_path = weights_path.with_suffix(".engine")

    export_kwargs = dict(
        format="engine",
        imgsz=args.img_size,
        half=(args.precision == "fp16"),
        int8=(args.precision == "int8"),
        workspace=args.workspace,
        device=0,          # GPU 0
        verbose=True,
    )

    if args.precision == "int8" and args.calib_data:
        export_kwargs["data"] = args.calib_data

    print(f"[Export] 开始导出（精度={args.precision}，分辨率={args.img_size}）...")
    print("  此过程在 Jetson Orin NX 上约需 3-8 分钟，请耐心等待...")

    model.export(**export_kwargs)

    # ultralytics 导出后文件名规则
    if not engine_path.exists():
        candidates = list(weights_path.parent.glob("*.engine"))
        if candidates:
            engine_path = candidates[0]

    print(f"[Export] 导出完成：{engine_path}")
    return engine_path


def benchmark(engine_path: Path, img_size: int, runs: int) -> None:
    """对 TensorRT engine 进行速度基准测试。"""
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        print("[Benchmark] 跳过：缺少依赖")
        return

    print(f"\n[Benchmark] 加载 TensorRT engine：{engine_path}")
    model = YOLO(str(engine_path))

    # 生成随机测试图像
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # 预热
    print("[Benchmark] 预热中（10 次）...")
    for _ in range(10):
        model.predict(dummy, verbose=False)

    # 正式测速
    print(f"[Benchmark] 正式测速（{runs} 次）...")
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    fps = 1000.0 / avg_ms

    print("\n====== TensorRT 速度测试结果 ======")
    print(f"  平均延迟：{avg_ms:.2f} ms")
    print(f"  最低延迟：{min_ms:.2f} ms")
    print(f"  最高延迟：{max_ms:.2f} ms")
    print(f"  吞吐量  ：{fps:.1f} FPS")
    print("===================================")

    if fps >= 30:
        print("  ✓ 满足实时检测要求（>=30 FPS）")
    else:
        print("  ⚠ 低于 30 FPS，建议使用更小模型（yolov8n）或降低分辨率")


def main() -> None:
    args = parse_args()

    if args.precision == "int8" and not args.calib_data:
        print("警告：INT8 模式建议提供 --calib-data 校准图像以减少精度损失")

    engine_path = export_tensorrt(args)

    if args.benchmark:
        benchmark(engine_path, args.img_size, args.benchmark_runs)

    print(f"\n[Done] TensorRT 模型路径：{engine_path}")
    print("[Done] 使用方法：")
    print(f"  python3 main.py --source /dev/video0 --model {engine_path} --device cuda --tracker")


if __name__ == "__main__":
    main()

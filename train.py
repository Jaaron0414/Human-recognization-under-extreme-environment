"""
针对夜间 / 远光灯数据集的 YOLOv8 微调训练脚本

支持的数据集（含夜间行人）：
  - NightOwls   https://www.nightowls-dataset.org/
  - ECP (EuroCity Persons) 夜间子集
  - BDD100K 夜间子集
  - 自定义 YOLO 格式数据集

用法示例
--------
# 从预训练权重微调（推荐）
python train.py --data data.yaml --epochs 50 --batch 16 --weights yolov8s.pt

# 从头训练（不推荐，需大量数据）
python train.py --data data.yaml --epochs 100 --batch 8 --weights "" --from-scratch
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 夜间行人微调")

    parser.add_argument("--data", type=str, required=True,
                        help="YOLO 格式的 data.yaml 路径")
    parser.add_argument("--weights", type=str, default="yolov8s.pt",
                        help="预训练权重，空字符串表示从头训练")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数（默认 50）")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size（默认 16）")
    parser.add_argument("--img-size", type=int, default=640,
                        help="训练分辨率（默认 640）")
    parser.add_argument("--device", type=str, default="0",
                        help="训练设备：GPU ID 或 'cpu'")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="结果保存目录")
    parser.add_argument("--name", type=str, default="highbeam_pedestrian",
                        help="实验名称")
    parser.add_argument("--from-scratch", action="store_true",
                        help="不加载预训练权重，从头训练")

    # 数据增强（夜间场景专用）
    parser.add_argument("--hsv-v", type=float, default=0.5,
                        help="亮度抖动幅度（模拟不同光照强度）")
    parser.add_argument("--degrees", type=float, default=5.0,
                        help="旋转增强角度")
    parser.add_argument("--mosaic", type=float, default=1.0,
                        help="Mosaic 增强概率")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请先安装 ultralytics：pip install ultralytics")

    # 选择基础模型
    if args.from_scratch or not args.weights:
        print("[Train] 从头训练（无预训练权重）")
        model = YOLO("yolov8s.yaml")  # 仅加载网络结构
    else:
        print(f"[Train] 加载预训练权重：{args.weights}")
        model = YOLO(args.weights)

    # 启动训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        # 夜间场景增强参数
        hsv_v=args.hsv_v,         # 亮度变化
        hsv_s=0.3,                 # 饱和度变化（夜间色彩信息少）
        degrees=args.degrees,
        mosaic=args.mosaic,
        flipud=0.0,                # 行人不会倒置
        mixup=0.1,
        # 训练稳定性
        patience=15,               # 早停
        save_period=10,
        val=True,
        plots=True,
    )

    print(f"\n[Train] 训练完成！结果保存至：{Path(args.project) / args.name}")
    print(f"[Train] 最佳权重：{Path(args.project) / args.name / 'weights' / 'best.pt'}")

    # 自动验证
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        print("\n[Val] 使用最佳权重进行验证...")
        best_model = YOLO(str(best_weights))
        metrics = best_model.val(data=args.data, imgsz=args.img_size)
        print(f"[Val] mAP50    = {metrics.box.map50:.4f}")
        print(f"[Val] mAP50-95 = {metrics.box.map:.4f}")
        print(f"[Val] Precision = {metrics.box.mp:.4f}")
        print(f"[Val] Recall    = {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()

"""
数据集统计与可视化脚本
用于检验数据集质量、分布，在训练前诊断问题

用法
----
python dataset_stats.py --dataset ./dataset
python dataset_stats.py --dataset ./dataset --show-samples 16
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="数据集统计与可视化")
    parser.add_argument("--dataset", type=str, default="./dataset",
                        help="YOLO 格式数据集根目录")
    parser.add_argument("--show-samples", type=int, default=0,
                        help="随机展示 N 张训练样本（0 则跳过）")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"])
    return parser.parse_args()


def compute_stats(dataset_dir: Path, split: str) -> dict:
    img_dir = dataset_dir / "images" / split
    lbl_dir = dataset_dir / "labels" / split

    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    stats = {
        "total_images": len(img_files),
        "images_with_pedestrians": 0,
        "total_pedestrians": 0,
        "box_widths": [],
        "box_heights": [],
        "pedestrians_per_image": [],
    }

    for img_path in img_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
        n = len(lines)
        if n == 0:
            continue

        stats["images_with_pedestrians"] += 1
        stats["total_pedestrians"] += n
        stats["pedestrians_per_image"].append(n)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h
            stats["box_widths"].append(bw)
            stats["box_heights"].append(bh)

    return stats


def print_stats(stats: dict, split: str) -> None:
    n = stats["total_images"]
    nw = stats["images_with_pedestrians"]
    total_p = stats["total_pedestrians"]
    ppi = stats["pedestrians_per_image"]
    bw = stats["box_widths"]
    bh = stats["box_heights"]

    print(f"\n{'='*45}")
    print(f"  数据集统计 [{split}]")
    print(f"{'='*45}")
    print(f"  总图像数          : {n}")
    print(f"  含行人图像数      : {nw} ({nw/max(n,1)*100:.1f}%)")
    print(f"  总行人框数        : {total_p}")
    if ppi:
        print(f"  每图平均行人数    : {sum(ppi)/len(ppi):.2f}")
        print(f"  每图最大行人数    : {max(ppi)}")
    if bw:
        print(f"\n  检测框宽度(px)")
        print(f"    均值: {sum(bw)/len(bw):.1f}  最小: {min(bw):.1f}  最大: {max(bw):.1f}")
        print(f"  检测框高度(px)")
        print(f"    均值: {sum(bh)/len(bh):.1f}  最小: {min(bh):.1f}  最大: {max(bh):.1f}")

        # 按尺寸分布（小/中/大目标）
        small = sum(1 for h in bh if h < 50)
        medium = sum(1 for h in bh if 50 <= h < 150)
        large = sum(1 for h in bh if h >= 150)
        total = len(bh)
        print(f"\n  目标尺寸分布（按高度）")
        print(f"    小目标 (<50px) : {small} ({small/total*100:.1f}%)")
        print(f"    中目标 (50-150): {medium} ({medium/total*100:.1f}%)")
        print(f"    大目标 (>150px): {large} ({large/total*100:.1f}%)")
    print(f"{'='*45}\n")


def show_samples(dataset_dir: Path, split: str, n_samples: int) -> None:
    img_dir = dataset_dir / "images" / split
    lbl_dir = dataset_dir / "labels" / split

    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not img_files:
        print("[Samples] 未找到图像")
        return

    samples = random.sample(img_files, min(n_samples, len(img_files)))
    panels = []

    for img_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 缩放到统一尺寸
        panels.append(cv2.resize(img, (320, 240)))

    if not panels:
        return

    # 拼接为网格
    cols = 4
    rows_imgs = []
    for i in range(0, len(panels), cols):
        row = panels[i:i+cols]
        while len(row) < cols:
            row.append(np.zeros((240, 320, 3), dtype=np.uint8))
        rows_imgs.append(np.hstack(row))
    grid = np.vstack(rows_imgs)

    cv2.imshow(f"Dataset Samples [{split}] - press any key to close", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在：{dataset_dir}")

    stats = compute_stats(dataset_dir, args.split)
    print_stats(stats, args.split)

    if args.show_samples > 0:
        show_samples(dataset_dir, args.split, args.show_samples)


if __name__ == "__main__":
    main()

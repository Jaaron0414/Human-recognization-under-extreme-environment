"""
BDD100K 数据集下载与预处理脚本
自动过滤夜间场景 → 提取行人标注 → 转换为 YOLO 格式

BDD100K 官网：https://www.bdd100k.com/
需要先注册账号，登录后从以下页面获取下载链接：
  https://doc.bdd100k.com/download.html

需要下载的文件：
  1. bdd100k_images_100k.zip   (~6.5 GB) 图像
  2. bdd100k_labels_release.zip (~130 MB) 标注

用法
----
# Step 1：下载文件（见上方说明），放入同一目录后运行：
python prepare_dataset.py --images-zip bdd100k_images_100k.zip --labels-zip bdd100k_labels_release.zip

# Step 2：只使用已解压的目录
python prepare_dataset.py --bdd-root ./bdd100k --output ./dataset

# Step 3：同时进行数据增强（合成远光灯效果）
python prepare_dataset.py --bdd-root ./bdd100k --output ./dataset --augment
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BDD100K 夜间行人数据集准备工具")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bdd-root", type=str,
                       help="已解压的 BDD100K 根目录（含 images/ 和 labels/ 子目录）")
    group.add_argument("--images-zip", type=str,
                       help="bdd100k_images_100k.zip 路径（配合 --labels-zip 使用）")

    parser.add_argument("--labels-zip", type=str, default=None,
                        help="bdd100k_labels_release.zip 路径")
    parser.add_argument("--output", type=str, default="./dataset",
                        help="YOLO 格式输出目录（默认 ./dataset）")
    parser.add_argument("--time-of-day", type=str, default="night",
                        choices=["night", "dawn/dusk", "daytime", "all"],
                        help="按时段过滤（默认 night）")
    parser.add_argument("--min-height", type=int, default=20,
                        help="行人最小高度（像素），过滤过小目标（默认 20）")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="验证集比例（默认 0.15）")
    parser.add_argument("--augment", action="store_true",
                        help="对训练集额外生成合成远光灯增强图像")
    parser.add_argument("--augment-ratio", type=float, default=0.5,
                        help="增强图像占训练集比例（默认 0.5）")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ──────────────────────────────────────────────
# 解压
# ──────────────────────────────────────────────

def extract_zip(zip_path: str, extract_to: str) -> None:
    print(f"[Extract] 解压 {zip_path} → {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"[Extract] 完成")


# ──────────────────────────────────────────────
# BDD100K JSON 标注解析
# ──────────────────────────────────────────────

def load_bdd_labels(labels_dir: Path) -> dict[str, dict]:
    """
    加载 BDD100K JSON 标注，返回 {filename: annotation_dict} 映射。
    支持新版（det_20/）和旧版（labels/）目录结构。
    """
    candidates = [
        labels_dir / "det_20" / "det_train.json",
        labels_dir / "det_20" / "det_val.json",
        labels_dir / "bdd100k_labels_images_train.json",
        labels_dir / "bdd100k_labels_images_val.json",
    ]

    all_annotations: dict[str, dict] = {}
    for json_path in candidates:
        if not json_path.exists():
            continue
        print(f"[Labels] 加载 {json_path} ...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 新版格式：{"frames": [...]}  旧版格式：[...]
        frames = data.get("frames", data) if isinstance(data, dict) else data
        for frame in frames:
            name = frame.get("name", "")
            all_annotations[name] = frame

    print(f"[Labels] 共加载 {len(all_annotations)} 条标注")
    return all_annotations


def filter_night_frames(
    annotations: dict[str, dict],
    time_of_day: str = "night",
) -> list[str]:
    """按 timeofday 属性过滤帧，返回符合条件的文件名列表。"""
    if time_of_day == "all":
        return list(annotations.keys())

    filtered = []
    for name, ann in annotations.items():
        attrs = ann.get("attributes", {})
        tod = attrs.get("timeofday", "").lower()
        if time_of_day == "dawn/dusk":
            if "dusk" in tod or "dawn" in tod:
                filtered.append(name)
        elif tod == time_of_day:
            filtered.append(name)

    print(f"[Filter] 时段='{time_of_day}'：{len(filtered)} 张图像")
    return filtered


# ──────────────────────────────────────────────
# YOLO 标注转换
# ──────────────────────────────────────────────

def bdd_box_to_yolo(
    box2d: dict,
    img_w: int,
    img_h: int,
) -> Optional[tuple[float, float, float, float]]:
    """
    BDD100K box2d {"x1":..,"y1":..,"x2":..,"y2":..} → YOLO (cx, cy, w, h) 归一化
    """
    x1 = float(box2d["x1"])
    y1 = float(box2d["y1"])
    x2 = float(box2d["x2"])
    y2 = float(box2d["y2"])

    if x2 <= x1 or y2 <= y1:
        return None

    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    # 边界校验
    if not (0 < cx < 1 and 0 < cy < 1 and 0 < w <= 1 and 0 < h <= 1):
        return None
    return cx, cy, w, h


def convert_frame(
    ann: dict,
    img_w: int,
    img_h: int,
    min_height: int,
) -> list[str]:
    """将单帧标注转为 YOLO 格式文本行列表（仅保留 pedestrian 类别）。"""
    lines = []
    labels = ann.get("labels", []) or []
    for label in labels:
        if label.get("category", "") != "pedestrian":
            continue
        box2d = label.get("box2d")
        if not box2d:
            continue
        # 过滤过矮的行人（噪声/远处）
        if (box2d["y2"] - box2d["y1"]) < min_height:
            continue
        result = bdd_box_to_yolo(box2d, img_w, img_h)
        if result is None:
            continue
        cx, cy, w, h = result
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


# ──────────────────────────────────────────────
# 合成远光灯数据增强
# ──────────────────────────────────────────────

def synthesize_high_beam(img: np.ndarray, intensity: float = None) -> np.ndarray:
    """
    模拟远光灯效果：在图像上方中央叠加高斯光晕，过曝局部区域。
    intensity: 光晕强度，None 则随机（0.6~1.0）
    """
    if intensity is None:
        intensity = random.uniform(0.6, 1.0)

    h, w = img.shape[:2]
    result = img.astype(np.float32)

    # 光晕中心：图像上方 1/3 处中央（模拟对向来车远光灯位置）
    center_x = w // 2 + random.randint(-w // 6, w // 6)
    center_y = h // 3 + random.randint(-h // 8, h // 8)

    # 生成高斯光晕掩码
    Y, X = np.ogrid[:h, :w]
    sigma_x = w * random.uniform(0.15, 0.35)
    sigma_y = h * random.uniform(0.10, 0.25)
    glare = np.exp(
        -(((X - center_x) ** 2) / (2 * sigma_x ** 2) +
          ((Y - center_y) ** 2) / (2 * sigma_y ** 2))
    )
    glare = (glare * 255 * intensity).astype(np.float32)
    glare_3ch = np.stack([glare] * 3, axis=-1)

    result = np.clip(result + glare_3ch, 0, 255).astype(np.uint8)
    return result


# ──────────────────────────────────────────────
# 数据集构建主流程
# ──────────────────────────────────────────────

def build_dataset(
    bdd_root: Path,
    output_dir: Path,
    annotations: dict[str, dict],
    frame_names: list[str],
    val_ratio: float,
    min_height: int,
    augment: bool,
    augment_ratio: float,
    seed: int,
) -> None:
    random.seed(seed)
    random.shuffle(frame_names)

    n_val = max(1, int(len(frame_names) * val_ratio))
    val_names = set(frame_names[:n_val])
    train_names = frame_names[n_val:]

    # 创建目录结构
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "skipped": 0, "total_pedestrians": 0}

    # 图像搜索路径
    img_search_dirs = [
        bdd_root / "images" / "100k" / "train",
        bdd_root / "images" / "100k" / "val",
        bdd_root / "images" / "train",
        bdd_root / "images" / "val",
        bdd_root / "images",
    ]

    def find_image(name: str) -> Optional[Path]:
        stem = Path(name).stem
        for d in img_search_dirs:
            for ext in [".jpg", ".jpeg", ".png"]:
                p = d / (stem + ext)
                if p.exists():
                    return p
        return None

    all_names = list(val_names) + train_names
    total = len(all_names)

    for idx, name in enumerate(all_names):
        if idx % 500 == 0:
            print(f"[Build] 处理进度：{idx}/{total}")

        img_path = find_image(name)
        if img_path is None:
            stats["skipped"] += 1
            continue

        ann = annotations.get(name, {})
        img = cv2.imread(str(img_path))
        if img is None:
            stats["skipped"] += 1
            continue

        img_h, img_w = img.shape[:2]
        yolo_lines = convert_frame(ann, img_w, img_h, min_height)

        # 跳过无行人的帧（可选：保留负样本则注释此行）
        if not yolo_lines:
            stats["skipped"] += 1
            continue

        split = "val" if name in val_names else "train"
        stem = Path(name).stem

        dst_img = output_dir / "images" / split / (stem + ".jpg")
        dst_lbl = output_dir / "labels" / split / (stem + ".txt")

        shutil.copy(str(img_path), str(dst_img))
        dst_lbl.write_text("\n".join(yolo_lines))

        stats[split] += 1
        stats["total_pedestrians"] += len(yolo_lines)

        # 合成远光灯增强（仅训练集）
        if augment and split == "train" and random.random() < augment_ratio:
            aug_img = synthesize_high_beam(img)
            aug_stem = stem + "_hb"
            cv2.imwrite(str(output_dir / "images" / "train" / (aug_stem + ".jpg")), aug_img)
            (output_dir / "labels" / "train" / (aug_stem + ".txt")).write_text("\n".join(yolo_lines))
            stats["train"] += 1

    print(f"\n[Build] 数据集构建完成：")
    print(f"  训练集：{stats['train']} 张")
    print(f"  验证集：{stats['val']} 张")
    print(f"  跳过  ：{stats['skipped']} 张（无行人或图像缺失）")
    print(f"  行人框：{stats['total_pedestrians']} 个")


def write_data_yaml(output_dir: Path) -> None:
    yaml_content = f"""# 自动生成的数据集配置文件
path: {output_dir.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: pedestrian
"""
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[YAML] 配置文件已生成：{yaml_path}")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)

    # 解压（如果提供了 zip）
    bdd_root: Path
    if args.images_zip:
        extract_dir = output_dir / "bdd100k_raw"
        extract_zip(args.images_zip, str(extract_dir))
        if args.labels_zip:
            extract_zip(args.labels_zip, str(extract_dir))
        bdd_root = extract_dir / "bdd100k"
        if not bdd_root.exists():
            bdd_root = extract_dir
    else:
        bdd_root = Path(args.bdd_root)

    if not bdd_root.exists():
        raise FileNotFoundError(f"BDD100K 根目录不存在：{bdd_root}")

    # 加载标注
    labels_dir = bdd_root / "labels"
    if not labels_dir.exists():
        labels_dir = bdd_root
    annotations = load_bdd_labels(labels_dir)

    if not annotations:
        raise RuntimeError("未找到任何标注文件，请检查目录结构")

    # 过滤夜间帧
    frame_names = filter_night_frames(annotations, args.time_of_day)

    if not frame_names:
        raise RuntimeError(f"未找到时段='{args.time_of_day}'的帧，请检查标注或尝试 --time-of-day all")

    # 构建数据集
    build_dataset(
        bdd_root=bdd_root,
        output_dir=output_dir,
        annotations=annotations,
        frame_names=frame_names,
        val_ratio=args.val_ratio,
        min_height=args.min_height,
        augment=args.augment,
        augment_ratio=args.augment_ratio,
        seed=args.seed,
    )

    # 生成 data.yaml
    write_data_yaml(output_dir)

    print(f"\n[Done] 开始训练：")
    print(f"  python train.py --data {output_dir / 'data.yaml'} --weights yolov8s.pt --epochs 50")


if __name__ == "__main__":
    main()

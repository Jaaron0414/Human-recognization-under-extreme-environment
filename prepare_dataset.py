"""
NightOwls 数据集下载与预处理脚本
（NightOwls 专为夜间行人检测设计，牛津大学 VGG 组发布，无需注册直接下载）

下载地址（直链，无需注册）：
  训练图像（135 GB）: http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_training.zip
  训练标注（32 MB） : http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_training.json
  验证图像（50 GB） : http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.zip
  验证标注（10 MB） : http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.json

数据集统计：
  279,000 帧 | 夜间 + 黎明 | 欧洲多城市 | 1024×640 分辨率

用法
----
# Step 1：自动下载标注文件（小文件）并仅下载验证图像（快速验证）
python prepare_dataset.py --download-annotations --output ./dataset

# Step 2：已下载完整数据，指定本地目录处理
python prepare_dataset.py --nightowls-root ./nightowls --output ./dataset

# Step 3：同时进行合成远光灯数据增强
python prepare_dataset.py --nightowls-root ./nightowls --output ./dataset --augment
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ──────────────────────────────────────────────
# NightOwls 直链
# ──────────────────────────────────────────────

NIGHTOWLS_URLS = {
    "train_images": "http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_training.zip",
    "train_ann":    "http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_training.json",
    "val_images":   "http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.zip",
    "val_ann":      "http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.json",
}

# NightOwls 类别（只保留行人）
PEDESTRIAN_CATEGORIES = {"pedestrian"}


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NightOwls 夜间行人数据集准备工具")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nightowls-root", type=str,
                       help="已解压的 NightOwls 根目录")
    group.add_argument("--download-annotations", action="store_true",
                       help="自动下载标注 JSON（小文件，快速）；图像需手动下载")

    parser.add_argument("--output", type=str, default="./dataset",
                        help="YOLO 格式输出目录（默认 ./dataset）")
    parser.add_argument("--min-height", type=int, default=20,
                        help="行人最小高度（像素），过滤过小目标（默认 20）")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="从训练集中划出的验证比例（默认 0.15）")
    parser.add_argument("--augment", action="store_true",
                        help="对训练集额外生成合成远光灯增强图像")
    parser.add_argument("--augment-ratio", type=float, default=0.5,
                        help="增强图像占训练集比例（默认 0.5）")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ──────────────────────────────────────────────
# 下载工具
# ──────────────────────────────────────────────

def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[Download] 已存在，跳过：{dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Download] {url}")
    print(f"         → {dest}")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 / total_size)
            mb_done = block_num * block_size / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            print(f"\r  {pct:.1f}%  {mb_done:.0f}/{mb_total:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), _progress)
    print()


def download_annotations(output_dir: Path) -> tuple[Path, Path]:
    """仅下载标注 JSON（约 42 MB），不下载图像。"""
    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    train_ann = ann_dir / "nightowls_training.json"
    val_ann   = ann_dir / "nightowls_validation.json"

    download_file(NIGHTOWLS_URLS["train_ann"], train_ann)
    download_file(NIGHTOWLS_URLS["val_ann"],   val_ann)

    return train_ann, val_ann


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f"[Extract] {zip_path.name} → {extract_to} ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(extract_to))
    print("[Extract] 完成")


# ──────────────────────────────────────────────
# NightOwls JSON 解析（COCO 格式）
# ──────────────────────────────────────────────

def load_nightowls_annotations(ann_path: Path) -> tuple[dict, dict, dict]:
    """
    解析 NightOwls JSON（COCO 格式）。
    返回 (images_dict, annotations_by_image, categories_dict)
    """
    print(f"[Annotations] 加载 {ann_path} ...")
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 类别映射
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    # 图像 id → info
    images = {img["id"]: img for img in data.get("images", [])}

    # 图像 id → [annotation, ...]
    ann_by_image: dict[int, list] = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        ann_by_image.setdefault(img_id, []).append(ann)

    print(f"[Annotations] 图像数：{len(images)}  标注数：{len(data.get('annotations', []))}")
    return images, ann_by_image, categories


def nightowls_box_to_yolo(
    bbox: list,   # [x, y, w, h] in pixel
    img_w: int,
    img_h: int,
) -> Optional[tuple[float, float, float, float]]:
    """NightOwls [x,y,w,h] → YOLO (cx, cy, w, h) 归一化。"""
    x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    if bw <= 0 or bh <= 0:
        return None
    cx = (x + bw / 2) / img_w
    cy = (y + bh / 2) / img_h
    nw = bw / img_w
    nh = bh / img_h
    if not (0 < cx < 1 and 0 < cy < 1 and 0 < nw <= 1 and 0 < nh <= 1):
        return None
    return cx, cy, nw, nh


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
# 数据集构建主流程（NightOwls）
# ──────────────────────────────────────────────

def build_dataset_nightowls(
    images_dir: Optional[Path],
    ann_path: Path,
    output_dir: Path,
    split: str,          # "train" or "val"
    min_height: int,
    augment: bool,
    augment_ratio: float,
    seed: int,
) -> dict:
    """将 NightOwls 单个 split 转换为 YOLO 格式。"""
    random.seed(seed)

    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    images, ann_by_image, categories = load_nightowls_annotations(ann_path)
    stats = {"processed": 0, "skipped": 0, "total_pedestrians": 0}

    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]
        stem = Path(file_name).stem

        # 转换标注
        anns = ann_by_image.get(img_id, [])
        yolo_lines = []
        for ann in anns:
            cat_name = categories.get(ann["category_id"], "")
            if cat_name not in PEDESTRIAN_CATEGORIES:
                continue
            bbox = ann["bbox"]  # [x, y, w, h]
            if bbox[3] < min_height:
                continue
            # 过滤 crowd / iscrowd
            if ann.get("iscrowd", 0):
                continue
            result = nightowls_box_to_yolo(bbox, img_w, img_h)
            if result is None:
                continue
            cx, cy, bw, bh = result
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            stats["skipped"] += 1
            continue

        # 写标注
        lbl_path = output_dir / "labels" / split / (stem + ".txt")
        lbl_path.write_text("\n".join(yolo_lines))

        # 复制图像（如果有图像目录）
        if images_dir is not None:
            src_img = images_dir / file_name
            if not src_img.exists():
                # 有些数据集把文件名展平存放
                src_img = images_dir / Path(file_name).name
            if src_img.exists():
                dst_img = output_dir / "images" / split / (stem + ".jpg")
                shutil.copy(str(src_img), str(dst_img))

                # 合成远光灯增强（仅训练集）
                if augment and split == "train" and random.random() < augment_ratio:
                    img = cv2.imread(str(src_img))
                    if img is not None:
                        aug_img = synthesize_high_beam(img)
                        aug_stem = stem + "_hb"
                        cv2.imwrite(
                            str(output_dir / "images" / split / (aug_stem + ".jpg")), aug_img
                        )
                        (output_dir / "labels" / split / (aug_stem + ".txt")).write_text(
                            "\n".join(yolo_lines)
                        )
                        stats["processed"] += 1
            else:
                # 没有图像也写标注（标注-only 模式，用于快速测试）
                pass

        stats["processed"] += 1
        stats["total_pedestrians"] += len(yolo_lines)

    return stats


def write_data_yaml(output_dir: Path) -> None:
    yaml_content = f"""# 自动生成的数据集配置文件（NightOwls 夜间行人）
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

    if args.download_annotations:
        # 仅下载标注 JSON（约 42 MB 合计，几分钟内完成）
        print("=" * 50)
        print(" NightOwls 标注下载模式")
        print(" 图像文件较大（训练 135 GB，验证 50 GB），")
        print(" 请手动用 wget/aria2c 下载后再运行完整转换：")
        print(f"   wget {NIGHTOWLS_URLS['train_images']}")
        print(f"   wget {NIGHTOWLS_URLS['val_images']}")
        print("=" * 50)

        train_ann, val_ann = download_annotations(output_dir)

        # 仅转换标注（无图像，用于验证标注格式）
        print("\n[Info] 转换训练集标注（无图像，验证格式）...")
        s = build_dataset_nightowls(
            images_dir=None,
            ann_path=train_ann,
            output_dir=output_dir,
            split="train",
            min_height=args.min_height,
            augment=False,
            augment_ratio=0,
            seed=args.seed,
        )
        print(f"  训练集行人框：{s['total_pedestrians']}  跳过：{s['skipped']}")

        print("\n[Info] 转换验证集标注...")
        s = build_dataset_nightowls(
            images_dir=None,
            ann_path=val_ann,
            output_dir=output_dir,
            split="val",
            min_height=args.min_height,
            augment=False,
            augment_ratio=0,
            seed=args.seed,
        )
        print(f"  验证集行人框：{s['total_pedestrians']}  跳过：{s['skipped']}")

        write_data_yaml(output_dir)
        print("\n[Done] 标注准备完成！下载图像后重新运行 --nightowls-root 模式即可。")

    else:
        # 完整模式：本地已有数据
        root = Path(args.nightowls_root)
        if not root.exists():
            raise FileNotFoundError(f"NightOwls 根目录不存在：{root}")

        # 寻找标注文件
        def find_ann(name: str) -> Optional[Path]:
            for p in [root / name, root / "annotations" / name, root / "python" / name]:
                if p.exists():
                    return p
            return None

        train_ann = find_ann("nightowls_training.json")
        val_ann   = find_ann("nightowls_validation.json")

        if not train_ann:
            raise FileNotFoundError("未找到 nightowls_training.json，请检查目录结构")

        train_img_dir = next(
            (root / d for d in ["nightowls_training", "training", "images/train"] if (root / d).exists()),
            None
        )
        val_img_dir = next(
            (root / d for d in ["nightowls_validation", "validation", "images/val"] if (root / d).exists()),
            None
        )

        print("[Build] 处理训练集...")
        s = build_dataset_nightowls(
            images_dir=train_img_dir,
            ann_path=train_ann,
            output_dir=output_dir,
            split="train",
            min_height=args.min_height,
            augment=args.augment,
            augment_ratio=args.augment_ratio,
            seed=args.seed,
        )
        print(f"  训练集：{s['processed']} 张  行人框：{s['total_pedestrians']}  跳过：{s['skipped']}")

        if val_ann:
            print("[Build] 处理验证集...")
            s = build_dataset_nightowls(
                images_dir=val_img_dir,
                ann_path=val_ann,
                output_dir=output_dir,
                split="val",
                min_height=args.min_height,
                augment=False,
                augment_ratio=0,
                seed=args.seed,
            )
            print(f"  验证集：{s['processed']} 张  行人框：{s['total_pedestrians']}  跳过：{s['skipped']}")

        write_data_yaml(output_dir)
        print(f"\n[Done] 开始训练：")
        print(f"  python train.py --data {output_dir / 'data.yaml'} --weights yolov8s.pt --epochs 50")


if __name__ == "__main__":
    main()

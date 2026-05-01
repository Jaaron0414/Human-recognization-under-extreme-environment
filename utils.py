"""
可视化与评估工具
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from detector import Detection, FrameResult


# ──────────────────────────────────────────────
# 绘制检测结果
# ──────────────────────────────────────────────

def draw_detections(
    img: np.ndarray,
    detections: List[Detection],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_conf: bool = True,
    show_track_id: bool = True,
    font_scale: float = 0.6,
) -> np.ndarray:
    """在图像上绘制检测框与标签，返回带标注的副本。"""
    vis = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        label_parts = []
        if show_track_id and det.track_id is not None:
            label_parts.append(f"ID:{det.track_id}")
        if show_conf:
            label_parts.append(f"{det.confidence:.2f}")
        label = " ".join(label_parts) if label_parts else "person"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            vis, label, (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA
        )
    return vis


def draw_stats(
    img: np.ndarray,
    result: FrameResult,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 0),
) -> np.ndarray:
    """在图像左上角绘制统计信息（行人数、推理耗时）。"""
    vis = img.copy()
    lines = [
        f"Pedestrians: {result.count}",
        f"Infer: {result.inference_ms:.1f} ms",
        f"Total: {result.total_ms:.1f} ms",
    ]
    x, y = position
    for i, line in enumerate(lines):
        cv2.putText(
            vis, line, (x, y + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA
        )
    return vis


# ──────────────────────────────────────────────
# 预处理步骤对比可视化
# ──────────────────────────────────────────────

def show_preprocessing_steps(steps: dict, window_name: str = "Preprocessing Steps") -> None:
    """
    并排显示各预处理阶段结果。
    按任意键关闭窗口。
    """
    panels = list(steps.values())
    labels = list(steps.keys())

    # 统一尺寸
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    resized = []
    for panel, label in zip(panels, labels):
        r = cv2.resize(panel, (w, h))
        cv2.putText(r, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        resized.append(r)

    # 每行最多 4 张
    cols = 4
    rows = []
    for i in range(0, len(resized), cols):
        row_imgs = resized[i: i + cols]
        # 补齐到 cols 张
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros((h, w, 3), dtype=np.uint8))
        rows.append(np.hstack(row_imgs))
    grid = np.vstack(rows)

    cv2.imshow(window_name, grid)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def save_preprocessing_steps(steps: dict, output_dir: str | Path) -> None:
    """将各预处理阶段结果保存为独立图片文件。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, img in steps.items():
        cv2.imwrite(str(output_dir / f"{name}.jpg"), img)
    print(f"[Utils] 预处理中间结果已保存至 {output_dir}")


# ──────────────────────────────────────────────
# IoU & 评估工具
# ──────────────────────────────────────────────

def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """计算两个 (x1,y1,x2,y2) 格式边界框的 IoU。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_frame(
    pred_boxes: List[Tuple[int, int, int, int]],
    gt_boxes: List[Tuple[int, int, int, int]],
    iou_thresh: float = 0.5,
) -> dict:
    """
    计算单帧的 TP / FP / FN。

    返回
    ----
    {"tp": int, "fp": int, "fn": int, "precision": float, "recall": float}
    """
    matched_gt = set()
    tp = 0
    fp = 0

    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt in enumerate(gt_boxes):
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou >= iou_thresh and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}

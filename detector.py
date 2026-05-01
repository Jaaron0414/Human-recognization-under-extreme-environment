"""
行人检测器模块
基于 YOLOv8 + 远光灯专用预处理流水线
支持图片、视频帧、摄像头流输入
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from preprocessing import HighBeamPreprocessor


@dataclass
class Detection:
    """单个行人检测结果。"""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) 像素坐标
    confidence: float
    track_id: Optional[int] = None    # 使用追踪器时填充

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class FrameResult:
    """单帧推理结果。"""
    detections: List[Detection] = field(default_factory=list)
    preprocessed_img: Optional[np.ndarray] = None
    inference_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)


class PedestrianDetector:
    """
    远光灯场景行人检测器。

    参数
    ----
    model_path : str | Path
        YOLOv8 权重文件路径（.pt 格式）。
        默认 'yolov8n.pt'（自动从 ultralytics 下载 nano 版本）。
    conf_threshold : float
        置信度阈值，低于此值的检测框被丢弃。
    iou_threshold : float
        NMS IoU 阈值。
    device : str
        推理设备，'cpu' / 'cuda' / 'mps'。
    use_tracker : bool
        是否启用 BoT-SORT 多目标追踪。
    preprocessor : HighBeamPreprocessor | None
        预处理器实例，None 则跳过预处理。
    img_size : int
        推理分辨率（宽/高均缩放到此尺寸）。
    """

    # COCO 数据集中 person 类别 ID
    _PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        use_tracker: bool = False,
        preprocessor: Optional[HighBeamPreprocessor] = None,
        img_size: int = 640,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.use_tracker = use_tracker
        self.img_size = img_size

        # 默认启用远光灯预处理
        self.preprocessor = preprocessor if preprocessor is not None else HighBeamPreprocessor()

        self._model = self._load_model(model_path)

    # ──────────────────────────────────────────────
    # 模型加载
    # ──────────────────────────────────────────────

    def _load_model(self, model_path: str | Path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "请先安装 ultralytics：pip install ultralytics"
            ) from exc

        model = YOLO(str(model_path))
        model.to(self.device)
        print(f"[Detector] 模型加载完毕：{model_path}  设备：{self.device}")
        return model

    # ──────────────────────────────────────────────
    # 核心推理
    # ──────────────────────────────────────────────

    def detect(self, img: np.ndarray, apply_preprocessing: bool = True) -> FrameResult:
        """
        对单帧 BGR 图像执行检测。

        参数
        ----
        img : np.ndarray
            BGR 格式原始图像。
        apply_preprocessing : bool
            是否在检测前执行远光灯预处理。

        返回
        ----
        FrameResult
        """
        t0 = time.perf_counter()

        # 1. 预处理
        if apply_preprocessing and self.preprocessor is not None:
            processed = self.preprocessor.process(img)
        else:
            processed = img

        # 2. YOLO 推理
        t1 = time.perf_counter()
        if self.use_tracker:
            raw_results = self._model.track(
                processed,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[self._PERSON_CLASS_ID],
                imgsz=self.img_size,
                verbose=False,
            )
        else:
            raw_results = self._model.predict(
                processed,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[self._PERSON_CLASS_ID],
                imgsz=self.img_size,
                verbose=False,
            )
        t2 = time.perf_counter()

        # 3. 解析结果
        detections = self._parse_results(raw_results)

        return FrameResult(
            detections=detections,
            preprocessed_img=processed,
            inference_ms=(t2 - t1) * 1000,
            total_ms=(t2 - t0) * 1000,
        )

    def _parse_results(self, raw_results) -> List[Detection]:
        detections: List[Detection] = []
        for result in raw_results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu())
                track_id = None
                if self.use_tracker and box.id is not None:
                    track_id = int(box.id[0].cpu())
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        track_id=track_id,
                    )
                )
        return detections

    # ──────────────────────────────────────────────
    # 批量推理（视频/数据集）
    # ──────────────────────────────────────────────

    def detect_batch(
        self, imgs: List[np.ndarray], apply_preprocessing: bool = True
    ) -> List[FrameResult]:
        """对一批图像执行检测，返回对应 FrameResult 列表。"""
        return [self.detect(img, apply_preprocessing) for img in imgs]

    # ──────────────────────────────────────────────
    # 后处理：过滤太小或置信度太低的框
    # ──────────────────────────────────────────────

    @staticmethod
    def filter_detections(
        detections: List[Detection],
        min_conf: float = 0.0,
        min_area: int = 0,
        img_shape: Optional[Tuple[int, int]] = None,  # (H, W)
        border_margin: int = 0,
    ) -> List[Detection]:
        """
        按置信度、面积、边界裁剪过滤检测结果。

        参数
        ----
        min_conf : float
            最低置信度。
        min_area : int
            最小检测框面积（像素²）。
        img_shape : (H, W) | None
            图像尺寸，用于边界裁剪过滤。
        border_margin : int
            距图像边缘小于此像素的框被丢弃（避免截断行人）。
        """
        filtered = []
        for det in detections:
            if det.confidence < min_conf:
                continue
            if det.area < min_area:
                continue
            if img_shape is not None and border_margin > 0:
                H, W = img_shape
                x1, y1, x2, y2 = det.bbox
                if x1 < border_margin or y1 < border_margin:
                    continue
                if x2 > W - border_margin or y2 > H - border_margin:
                    continue
            filtered.append(det)
        return filtered

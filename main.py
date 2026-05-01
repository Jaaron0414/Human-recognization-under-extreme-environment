"""
主推理脚本
支持三种输入来源：图片文件 / 视频文件 / 摄像头实时流

用法示例
--------
# 图片检测（显示对比窗口）
python main.py --source image.jpg

# 视频检测（保存结果）
python main.py --source video.mp4 --save

# 摄像头实时检测（追踪模式）
python main.py --source 0 --tracker

# 不使用预处理（对比效果）
python main.py --source video.mp4 --no-preprocess

# 调试预处理各阶段
python main.py --source image.jpg --debug-preprocess
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from detector import PedestrianDetector
from preprocessing import HighBeamPreprocessor
from utils import draw_detections, draw_stats, show_preprocessing_steps, save_preprocessing_steps


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="远光灯下行人识别系统")

    parser.add_argument("--source", type=str, default="0",
                        help="输入源：图片路径 / 视频路径 / 摄像头ID（整数）/ GStreamer pipeline 字符串")
    parser.add_argument("--model", type=str, default="yolov8s.engine",
                        help="YOLOv8 权重路径，Jetson 上推荐使用 .engine（TensorRT）格式")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="置信度阈值（默认 0.35）")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值（默认 0.45）")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda", "mps"],
                        help="推理设备（Jetson 上默认 cuda）")
    parser.add_argument("--img-size", type=int, default=640,
                        help="推理分辨率（默认 640）")

    # 预处理开关
    parser.add_argument("--no-preprocess", action="store_true",
                        help="跳过远光灯预处理（对比基线用）")
    parser.add_argument("--no-glare", action="store_true",
                        help="关闭眩光抑制")
    parser.add_argument("--no-retinex", action="store_true",
                        help="关闭 MSR")
    parser.add_argument("--no-clahe", action="store_true",
                        help="关闭 CLAHE")
    parser.add_argument("--debug-preprocess", action="store_true",
                        help="可视化各预处理阶段中间结果后退出")

    # 输出选项
    parser.add_argument("--save", action="store_true",
                        help="保存带标注的结果到 ./output/ 目录")
    parser.add_argument("--tracker", action="store_true",
                        help="启用 BoT-SORT 多目标追踪")
    parser.add_argument("--hide-window", action="store_true",
                        help="不弹出显示窗口（适合服务器环境）")

    return parser.parse_args()


# ──────────────────────────────────────────────
# 输入源辅助
# ──────────────────────────────────────────────

def _gstreamer_pipeline(cam_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> str:
    """
    生成 Jetson CSI 摄像头的 GStreamer pipeline 字符串。
    适用于搭载 IMX219 / IMX477 等 MIPI CSI 摄像头的 JetRover。
    """
    return (
        f"nvarguscamerasrc sensor-id={cam_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink"
    )


def open_source(source: str):
    """
    返回 (cap_or_img, is_image, is_video) 元组。
    支持：整数摄像头ID / GStreamer pipeline / 图片路径 / 视频路径
    """
    # GStreamer pipeline 字符串（以 "nvargus" 或 "gst-" 开头）
    if source.startswith("nvargus") or source.startswith("gst-"):
        cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开 GStreamer pipeline：{source}")
        return cap, False, True

    # 尝试解析为摄像头 ID
    try:
        cam_id = int(source)
        # 优先尝试 GStreamer CSI 摄像头（Jetson）
        gst_pipeline = _gstreamer_pipeline(cam_id)
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[Source] 使用 GStreamer CSI 摄像头（ID={cam_id}）")
            return cap, False, True
        # 回退到标准 V4L2 摄像头
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {cam_id}")
        print(f"[Source] 使用 V4L2 摄像头（ID={cam_id}）")
        return cap, False, True
    except ValueError:
        pass

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{source}")

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"无法读取图像：{source}")
        return img, True, False
    elif suffix in {".mp4", ".avi", ".mov", ".mkv", ".ts"}:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{source}")
        return cap, False, True
    else:
        raise ValueError(f"不支持的文件格式：{suffix}")


def make_video_writer(output_path: str, cap, fourcc: str = "mp4v"):
    """根据输入视频参数创建 VideoWriter。"""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (w, h),
    )


# ──────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────

def run_image(args, detector: PedestrianDetector, img: np.ndarray, source_path: str) -> None:
    """单张图片检测流程。"""
    # 调试预处理
    if args.debug_preprocess and detector.preprocessor is not None:
        steps = detector.preprocessor.visualize_steps(img)
        out_dir = Path("output") / "preprocess_steps"
        save_preprocessing_steps(steps, out_dir)
        if not args.hide_window:
            show_preprocessing_steps(steps)
        return

    apply_pre = not args.no_preprocess
    result = detector.detect(img, apply_preprocessing=apply_pre)

    # 绘制
    vis_raw = draw_detections(img, result.detections, color=(0, 255, 0))
    vis_raw = draw_stats(vis_raw, result)

    if apply_pre and result.preprocessed_img is not None:
        vis_pre = draw_detections(result.preprocessed_img, result.detections, color=(0, 200, 255))
        combined = np.hstack([vis_raw, vis_pre])
        caption = "Left: Original + Detection  |  Right: Preprocessed + Detection"
    else:
        combined = vis_raw
        caption = "Detection Result (no preprocessing)"

    print(f"[Result] 检测到行人：{result.count} 人  推理耗时：{result.inference_ms:.1f} ms")
    for i, det in enumerate(result.detections):
        print(f"  #{i+1}  conf={det.confidence:.3f}  bbox={det.bbox}")

    if args.save:
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / (Path(source_path).stem + "_result.jpg")
        cv2.imwrite(str(out_path), combined)
        print(f"[Save] 结果已保存：{out_path}")

    if not args.hide_window:
        cv2.imshow(caption, combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(args, detector: PedestrianDetector, cap: cv2.VideoCapture, source: str) -> None:
    """视频 / 摄像头实时检测流程。"""
    apply_pre = not args.no_preprocess
    writer = None

    if args.save:
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        out_name = Path(source).stem if not source.isdigit() else "camera"
        out_path = str(out_dir / f"{out_name}_result.mp4")
        writer = make_video_writer(out_path, cap)
        print(f"[Save] 将保存结果视频至：{out_path}")

    frame_idx = 0
    fps_timer = time.perf_counter()
    display_fps = 0.0

    print("[Run] 开始推理，按 'q' 退出，按 'p' 暂停...")

    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[Run] 视频结束或读帧失败，退出。")
                break

            result = detector.detect(frame, apply_preprocessing=apply_pre)

            vis = draw_detections(frame, result.detections, color=(0, 255, 0))
            vis = draw_stats(vis, result)

            # FPS 计算（每 30 帧更新一次）
            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.perf_counter() - fps_timer
                display_fps = 30 / elapsed
                fps_timer = time.perf_counter()
            cv2.putText(vis, f"FPS: {display_fps:.1f}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if writer is not None:
                writer.write(vis)

            if not args.hide_window:
                cv2.imshow("远光灯行人识别 (q退出 p暂停)", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[Run] 共处理 {frame_idx} 帧")


def main() -> None:
    args = parse_args()

    # 构建预处理器
    if args.no_preprocess:
        preprocessor = None
    else:
        preprocessor = HighBeamPreprocessor(
            use_glare_suppression=not args.no_glare,
            use_retinex=not args.no_retinex,
            use_clahe=not args.no_clahe,
        )

    # 构建检测器
    detector = PedestrianDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        use_tracker=args.tracker,
        preprocessor=preprocessor,
        img_size=args.img_size,
    )

    # 打开输入源
    source_obj, is_image, is_video = open_source(args.source)

    if is_image:
        run_image(args, detector, source_obj, args.source)
    elif is_video:
        run_video(args, detector, source_obj, args.source)


if __name__ == "__main__":
    main()

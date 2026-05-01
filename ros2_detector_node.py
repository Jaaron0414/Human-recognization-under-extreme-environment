"""
ROS2 行人检测节点
将我们的检测器封装为 ROS2 节点，与 JetRover 机器人系统集成

功能：
  - 订阅摄像头图像话题（/camera/image_raw 或 /image_raw）
  - 发布检测结果（BoundingBox2DArray）
  - 发布带标注的可视化图像
  - 发布行人计数（用于导航决策）

运行方式（在 JetRover 上）：
  # 终端 1：启动相机节点
  ros2 launch hiwonder_bringup camera.launch.py

  # 终端 2：启动本检测节点
  source /opt/ros/humble/setup.bash
  python3 ros2_detector_node.py

  # 终端 3：查看检测结果
  ros2 topic echo /pedestrian_detections
  ros2 run rqt_image_view rqt_image_view /pedestrian_image
"""

import sys

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Int32, String
    from vision_msgs.msg import BoundingBox2D, BoundingBox2DArray, Pose2D
    from cv_bridge import CvBridge
except ImportError:
    print("错误：ROS2 环境未初始化，请先执行：source /opt/ros/humble/setup.bash")
    sys.exit(1)

import json
import time

import cv2
import numpy as np

from detector import PedestrianDetector
from preprocessing import HighBeamPreprocessor


class PedestrianDetectorNode(Node):
    """
    ROS2 行人检测节点。

    订阅话题
    --------
    /camera/image_raw  (sensor_msgs/Image)  原始摄像头图像

    发布话题
    --------
    /pedestrian_detections  (vision_msgs/BoundingBox2DArray)  检测框列表
    /pedestrian_count       (std_msgs/Int32)                  当前帧行人数
    /pedestrian_image       (sensor_msgs/Image)               带标注的可视化图像
    /pedestrian_alert       (std_msgs/String)                 JSON 格式警报信息
    """

    def __init__(self):
        super().__init__("pedestrian_detector")

        # ── ROS2 参数声明 ──────────────────────────────
        self.declare_parameter("model_path", "yolov8s.engine")
        self.declare_parameter("conf_threshold", 0.35)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("use_tracker", True)
        self.declare_parameter("use_preprocessing", True)
        self.declare_parameter("img_size", 640)
        self.declare_parameter("camera_topic", "/camera/image_raw")
        # 警报阈值：检测到几人触发警报
        self.declare_parameter("alert_threshold", 1)
        # 最小检测框面积（像素²），过滤远处行人误检
        self.declare_parameter("min_area", 1000)

        # 读取参数
        model_path = self.get_parameter("model_path").value
        conf = self.get_parameter("conf_threshold").value
        iou = self.get_parameter("iou_threshold").value
        device = self.get_parameter("device").value
        use_tracker = self.get_parameter("use_tracker").value
        use_pre = self.get_parameter("use_preprocessing").value
        img_size = self.get_parameter("img_size").value
        camera_topic = self.get_parameter("camera_topic").value
        self.alert_threshold = self.get_parameter("alert_threshold").value
        self.min_area = self.get_parameter("min_area").value

        # ── 初始化检测器 ───────────────────────────────
        preprocessor = HighBeamPreprocessor() if use_pre else None
        self.detector = PedestrianDetector(
            model_path=model_path,
            conf_threshold=conf,
            iou_threshold=iou,
            device=device,
            use_tracker=use_tracker,
            preprocessor=preprocessor,
            img_size=img_size,
        )
        self.bridge = CvBridge()

        # ── 订阅者 ─────────────────────────────────────
        self.sub_image = self.create_subscription(
            Image,
            camera_topic,
            self._image_callback,
            10,
        )

        # ── 发布者 ─────────────────────────────────────
        self.pub_detections = self.create_publisher(BoundingBox2DArray, "/pedestrian_detections", 10)
        self.pub_count = self.create_publisher(Int32, "/pedestrian_count", 10)
        self.pub_image = self.create_publisher(Image, "/pedestrian_image", 10)
        self.pub_alert = self.create_publisher(String, "/pedestrian_alert", 10)

        # 统计
        self._frame_count = 0
        self._fps_timer = time.perf_counter()

        self.get_logger().info(
            f"行人检测节点启动 | 模型: {model_path} | 设备: {device} | 订阅: {camera_topic}"
        )

    # ──────────────────────────────────────────────────────
    # 图像回调
    # ──────────────────────────────────────────────────────

    def _image_callback(self, msg: Image) -> None:
        try:
            # ROS Image → OpenCV BGR
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"图像转换失败：{e}")
            return

        # 检测
        result = self.detector.detect(frame, apply_preprocessing=True)

        # 过滤小框
        detections = [d for d in result.detections if d.area >= self.min_area]

        # 发布各类话题
        self._publish_detections(detections, msg.header)
        self._publish_count(len(detections))
        self._publish_image(frame, detections, msg.header)
        self._publish_alert(detections, result.inference_ms)

        # 日志（每 30 帧输出一次）
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            elapsed = time.perf_counter() - self._fps_timer
            fps = 30.0 / elapsed
            self._fps_timer = time.perf_counter()
            self.get_logger().info(
                f"FPS: {fps:.1f} | 行人: {len(detections)} | 推理: {result.inference_ms:.1f}ms"
            )

    # ──────────────────────────────────────────────────────
    # 发布函数
    # ──────────────────────────────────────────────────────

    def _publish_detections(self, detections, header) -> None:
        msg = BoundingBox2DArray()
        msg.header = header
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            w = float(x2 - x1)
            h = float(y2 - y1)

            bb = BoundingBox2D()
            bb.center = Pose2D()
            bb.center.position.x = cx
            bb.center.position.y = cy
            bb.size_x = w
            bb.size_y = h
            msg.boxes.append(bb)
        self.pub_detections.publish(msg)

    def _publish_count(self, count: int) -> None:
        msg = Int32()
        msg.data = count
        self.pub_count.publish(msg)

    def _publish_image(self, frame: np.ndarray, detections, header) -> None:
        vis = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{det.track_id} {det.confidence:.2f}" if det.track_id else f"{det.confidence:.2f}"
            cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(vis, f"Pedestrians: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        img_msg.header = header
        self.pub_image.publish(img_msg)

    def _publish_alert(self, detections, inference_ms: float) -> None:
        """当行人数超过阈值时发布 JSON 格式警报，供导航节点消费。"""
        count = len(detections)
        alert_data = {
            "pedestrian_count": count,
            "alert": count >= self.alert_threshold,
            "inference_ms": round(inference_ms, 1),
            "detections": [
                {
                    "bbox": list(det.bbox),
                    "confidence": round(det.confidence, 3),
                    "track_id": det.track_id,
                }
                for det in detections
            ],
        }
        msg = String()
        msg.data = json.dumps(alert_data)
        self.pub_alert.publish(msg)

        if count >= self.alert_threshold:
            self.get_logger().warn(f"⚠ 检测到 {count} 名行人，已发布安全警报")


def main(args=None):
    rclpy.init(args=args)
    node = PedestrianDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

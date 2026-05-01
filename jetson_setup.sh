#!/bin/bash
# ============================================================
# Jetson Orin NX Super 环境配置脚本
# 适用于 JetPack 6.x (Ubuntu 22.04 + CUDA 12.x)
# 在机器人 SSH 连接后运行：bash jetson_setup.sh
# ============================================================

set -e

echo "======================================"
echo " 远光灯行人识别 - Jetson 环境配置"
echo "======================================"

# ── 1. 确认 JetPack 版本 ──────────────────────────────
echo "[1/7] 检查 JetPack / CUDA 版本..."
dpkg -l | grep nvidia-jetpack || echo "  警告：未检测到 nvidia-jetpack 包"
nvcc --version
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# ── 2. 系统依赖 ──────────────────────────────────────
echo "[2/7] 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-dev \
    libopencv-dev python3-opencv \
    v4l-utils \          # 摄像头调试工具
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev  # GStreamer（CSI 摄像头）

# ── 3. Python 依赖（Jetson 版本注意事项）──────────────
echo "[3/7] 安装 Python 依赖..."
pip3 install --upgrade pip

# Jetson 上 torch/torchvision 需用 NVIDIA 提供的预编译包
# JetPack 6.x 对应 PyTorch 2.x
pip3 install \
    "ultralytics>=8.2.0" \
    numpy \
    "opencv-python>=4.9.0"

# ── 4. TensorRT Python 绑定（JetPack 已内置，此步骤确认）──
echo "[4/7] 检查 TensorRT..."
python3 -c "import tensorrt as trt; print('TensorRT:', trt.__version__)" || \
    echo "  提示：TensorRT 应由 JetPack 提供，请确认 JetPack 安装完整"

# ── 5. 检查摄像头 ─────────────────────────────────────
echo "[5/7] 列出可用摄像头设备..."
v4l2-ctl --list-devices 2>/dev/null || echo "  未找到 V4L2 摄像头"
ls /dev/video* 2>/dev/null || echo "  未找到 /dev/video* 设备"

# ── 6. 性能模式（最大 CPU/GPU 频率）────────────────────
echo "[6/7] 设置 Jetson 为最大性能模式..."
sudo nvpmodel -m 0          # MAXN 模式（最大功耗/性能）
sudo jetson_clocks          # 锁定最高时钟频率

# ── 7. ROS2 检查（可选）──────────────────────────────
echo "[7/7] 检查 ROS2 环境..."
if [ -f /opt/ros/humble/setup.bash ]; then
    echo "  ROS2 Humble 已安装"
    source /opt/ros/humble/setup.bash
    echo "  ROS2 版本: $(ros2 --version 2>/dev/null || echo '未知')"
elif [ -f /opt/ros/iron/setup.bash ]; then
    echo "  ROS2 Iron 已安装"
else
    echo "  警告：未检测到 ROS2，如需 ROS2 集成请先安装"
    echo "  安装命令：sudo apt install ros-humble-desktop"
fi

echo ""
echo "======================================"
echo " 环境配置完成！"
echo " 下一步：运行 python3 export_tensorrt.py 导出加速模型"
echo "======================================"

# Human Recognition Under Extreme Environment
### 远光灯下行人识别系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-brightgreen)](https://docs.ros.org/en/humble/)
[![Jetson](https://img.shields.io/badge/Hardware-Jetson%20Orin%20NX-76b900)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

A real-time pedestrian detection system designed for **high-beam headlight conditions** (夜间远光灯场景). Deployed on a **Hiwonder JetRover** robot (Jetson Orin NX Super 16GB) for community patrol applications.

---

## The Problem

Standard pedestrian detectors fail under high-beam headlights due to:
- **Overexposure / pixel saturation** in the center of the frame
- **Extreme contrast** between glare regions and dark surroundings
- **Halo artifacts** that obscure pedestrian silhouettes

## Our Solution

A dedicated **image preprocessing pipeline** before detection:

```
Raw Frame
   │
   ▼
[1] Glare Suppression   ← Telea inpainting on overexposed regions
   │
   ▼
[2] Multi-Scale Retinex ← Simulates human eye light adaptation (σ=15,80,250)
   │
   ▼
[3] CLAHE Enhancement   ← Boosts dark-region contrast in LAB color space
   │
   ▼
[4] YOLOv8 Detection    ← TensorRT FP16 on Jetson (~120 FPS)
```

---

## Hardware

| Component | Spec |
|---|---|
| Robot Platform | Hiwonder JetRover with Mecanum Chassis |
| Compute | NVIDIA Jetson Orin NX Super 16GB |
| GPU | 1024-core Ampere + 32 Tensor Cores |
| Camera | CSI MIPI (IMX219 / IMX477) |
| Framework | ROS2 Humble |

---

## Project Structure

```
├── preprocessing.py        # High-beam image enhancement pipeline
├── detector.py             # YOLOv8 pedestrian detector wrapper
├── utils.py                # Visualization & evaluation utilities
├── main.py                 # Inference script (image / video / camera)
├── train.py                # Fine-tuning on nighttime datasets
├── export_tensorrt.py      # Export YOLOv8 → TensorRT for Jetson
├── ros2_detector_node.py   # ROS2 node for robot integration
├── jetson_setup.sh         # One-click Jetson environment setup
├── data.yaml.example       # Dataset config template (YOLO format)
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run on an Image
```bash
python main.py --source your_night_image.jpg
```

### 3. Real-time Camera Detection
```bash
python main.py --source 0 --tracker --device cuda
```

### 4. Debug Preprocessing Stages
```bash
python main.py --source your_image.jpg --debug-preprocess
```

---

## Jetson Deployment

```bash
# Step 1: Configure environment
bash jetson_setup.sh

# Step 2: Export TensorRT engine (run once, takes ~5-8 min)
python3 export_tensorrt.py --weights yolov8s.pt --precision fp16 --benchmark

# Step 3a: Standalone detection
python3 main.py --source 0 --model yolov8s.engine --tracker

# Step 3b: ROS2 integration
source /opt/ros/humble/setup.bash
python3 ros2_detector_node.py
```

### Jetson Performance (YOLOv8s + TensorRT FP16)

| Model | Precision | Latency | FPS |
|---|---|---|---|
| YOLOv8n (PyTorch) | FP32 | ~25ms | ~40 |
| YOLOv8s (TensorRT) | FP16 | ~8ms | **~120** |
| YOLOv8m (TensorRT) | FP16 | ~15ms | ~65 |

---

## ROS2 Topics

| Topic | Type | Description |
|---|---|---|
| `/pedestrian_detections` | `BoundingBox2DArray` | Detection boxes (for obstacle avoidance) |
| `/pedestrian_count` | `Int32` | Pedestrian count per frame |
| `/pedestrian_alert` | `String` (JSON) | Safety alert when pedestrians detected |
| `/pedestrian_image` | `Image` | Annotated visualization stream |

---

## Recommended Nighttime Datasets

| Dataset | Description | Link |
|---|---|---|
| **NightOwls** | Specifically designed for nighttime pedestrian detection | [nightowls-dataset.org](https://www.nightowls-dataset.org/) |
| **ECP (EuroCity Persons)** | European city nighttime scenes, high-quality annotations | [eurocitypersons.mpi-inf.mpg.de](https://eurocitypersons.mpi-inf.mpg.de/) |
| **BDD100K** | Large-scale driving dataset, filter by `timeofday=night` | [bdd100k.com](https://www.bdd100k.com/) |
| **KAIST Multispectral** | RGB + thermal pairs for multi-modal research | [Link](https://soonminhwang.github.io/rgbt-ped-detection/) |

---

## Fine-tuning

```bash
# Prepare dataset in YOLO format, copy data.yaml.example → data.yaml and edit paths
cp data.yaml.example data.yaml

# Fine-tune from pretrained weights
python train.py --data data.yaml --weights yolov8s.pt --epochs 50 --batch 16
```

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{highbeam-pedestrian-detection,
  author = {Aaron Xu},
  title  = {Human Recognition Under Extreme Environment},
  year   = {2026},
  url    = {https://github.com/Jaaron0414/Human-recognization-under-extreme-environment}
}
```

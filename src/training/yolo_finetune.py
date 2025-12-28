# -*- coding: utf-8 -*-
"""
YOLO v11 Fine-tuning for PvZ object detection
Run this in Google Colab with GPU

Usage:
    1. Upload to Colab
    2. Run cells sequentially
    3. Download the exported model
"""

# ============================================
# CELL 1: Install dependencies
# ============================================
# !pip install roboflow ultralytics

# ============================================
# CELL 2: Download dataset from Roboflow
# ============================================
"""
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("pvz-bot-vision")
version = project.version(2)
dataset = version.download("yolov11")
"""

# ============================================
# CELL 3: Train YOLOv11 Nano
# ============================================
"""
from ultralytics import YOLO

# Load YOLOv11 nano (lightweight)
model = YOLO('yolo11n.pt')

# Train
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name='pvz_yolo11_run1'
)
"""

# ============================================
# CELL 4: Export to OpenVINO
# ============================================
"""
from ultralytics import YOLO

# Load trained model
model = YOLO('/content/runs/detect/pvz_yolo11_run1/weights/best.pt')

# Export to OpenVINO (optimized for Intel CPU)
model.export(format='openvino')

# Zip for download
!zip -r pvz_model_openvino.zip /content/runs/detect/pvz_yolo11_run1/weights/best_openvino_model/
"""

# ============================================
# Local usage after training
# ============================================
def load_trained_model(model_path: str):
    """Load trained YOLO model for inference"""
    from ultralytics import YOLO
    return YOLO(model_path)


def export_to_openvino(model_path: str, output_dir: str = "models/yolo"):
    """Export YOLO model to OpenVINO format"""
    from ultralytics import YOLO
    import os
    
    model = YOLO(model_path)
    model.export(format='openvino')
    
    print(f"Model exported to OpenVINO format")
    print(f"Copy the *_openvino folder to: {output_dir}/pvz_openvino/")


if __name__ == "__main__":
    print("This script is designed for Google Colab")
    print("Run the cells in Colab with GPU for training")

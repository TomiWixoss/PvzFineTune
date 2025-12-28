# -*- coding: utf-8 -*-
"""
YOLO detector for PvZ using OpenVINO
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    YOLO_MODEL_PATH, CLASS_NAMES, CLASS_COLORS,
    DEFAULT_CONFIDENCE, DEFAULT_IOU_THRESHOLD, INPUT_SIZE
)

try:
    from openvino.runtime import Core
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False


class YOLODetector:
    """YOLO object detector optimized with OpenVINO"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or YOLO_MODEL_PATH
        self.model = None
        self.input_layer = None
        self.output_layer = None
        self.input_size = INPUT_SIZE
        self.class_names = CLASS_NAMES
        self.class_colors = CLASS_COLORS
    
    def load(self) -> bool:
        """Load OpenVINO model"""
        if not HAS_OPENVINO:
            print("✗ OpenVINO not installed. Run: pip install openvino")
            return False
        
        try:
            print(f"Loading model: {self.model_path}")
            ie = Core()
            self.model = ie.compile_model(model=self.model_path, device_name="CPU")
            self.input_layer = self.model.input(0)
            self.output_layer = self.model.output(0)
            
            # Update class names from model if available
            num_classes = self.output_layer.shape[1] - 4
            print(f"✓ Model loaded! Classes: {num_classes}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        input_img = cv2.resize(image, self.input_size)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))  # HWC -> CHW
        input_img = np.expand_dims(input_img, axis=0)   # Add batch
        return input_img
    
    def postprocess(self, output, orig_shape, 
                    conf_threshold: float = None, 
                    iou_threshold: float = None) -> list:
        """Process model output to get detections"""
        conf_threshold = conf_threshold or DEFAULT_CONFIDENCE
        iou_threshold = iou_threshold or DEFAULT_IOU_THRESHOLD
        
        predictions = output[0].T
        
        boxes = []
        scores = []
        class_ids = []
        detections = []
        
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        
        for pred in predictions:
            x, y, w, h = pred[:4]
            class_confs = pred[4:]
            class_id = np.argmax(class_confs)
            confidence = class_confs[class_id]
            
            if confidence > conf_threshold:
                x1 = int((x - w / 2) * scale_x)
                y1 = int((y - h / 2) * scale_y)
                x2 = int((x + w / 2) * scale_x)
                y2 = int((y + h / 2) * scale_y)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))
                class_ids.append(int(class_id))
        
        # NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                for i in indices:
                    box = boxes[i]
                    class_name = self.class_names.get(class_ids[i], f"class_{class_ids[i]}")
                    detections.append({
                        "class_id": class_ids[i],
                        "class_name": class_name,
                        "confidence": scores[i],
                        "x": (box[0] + box[2]) // 2,
                        "y": (box[1] + box[3]) // 2,
                        "box": box
                    })
        
        return detections
    
    def detect(self, frame: np.ndarray, conf_threshold: float = None) -> list:
        """Run detection on frame"""
        input_tensor = self.preprocess(frame)
        output = self.model([input_tensor])[self.output_layer]
        return self.postprocess(output, frame.shape, conf_threshold)
    
    def detect_grouped(self, frame: np.ndarray, conf_threshold: float = None) -> dict:
        """Run detection and group by class name"""
        detections = self.detect(frame, conf_threshold)
        results = {name: [] for name in self.class_names.values()}
        
        for det in detections:
            if det["class_name"] in results:
                results[det["class_name"]].append({
                    "x": det["x"],
                    "y": det["y"],
                    "conf": det["confidence"]
                })
        
        return results
    
    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            if "box" in det:
                x1, y1, x2, y2 = det["box"]
            else:
                x, y = det["x"], det["y"]
                x1, y1, x2, y2 = x - 20, y - 20, x + 20, y + 20
            
            class_name = det["class_name"]
            score = det["confidence"]
            color = self.class_colors.get(class_name, (255, 255, 255))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {score:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame

# -*- coding: utf-8 -*-
"""
YOLO Detector - Object detection với OpenVINO
Tự động load classes từ model metadata
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..core.constants import DEFAULT_CONFIDENCE, DEFAULT_IOU_THRESHOLD, INPUT_SIZE

try:
    from openvino.runtime import Core
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False


def generate_colors(n: int) -> Dict[str, tuple]:
    """Generate distinct colors for n classes"""
    colors = {}
    for i in range(n):
        hue = int(180 * i / n)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors[i] = tuple(int(c) for c in color)
    return colors


class YOLODetector:
    """YOLO object detector với OpenVINO - auto load classes từ metadata"""
    
    def __init__(self, model_path: str = None):
        from ..core.config import Config
        self.model_path = model_path or str(Config.YOLO_MODEL_PATH)
        self.model = None
        self.input_layer = None
        self.output_layer = None
        self.input_size = INPUT_SIZE
        self.class_names = {}
        self.class_colors = {}
    
    def _load_metadata(self):
        """Load class names từ metadata.yaml"""
        model_dir = Path(self.model_path).parent
        metadata_path = model_dir / "metadata.yaml"
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
            
            names = metadata.get("names", {})
            if isinstance(names, dict):
                self.class_names = {int(k): v for k, v in names.items()}
            elif isinstance(names, list):
                self.class_names = {i: name for i, name in enumerate(names)}
            
            print(f"✓ Loaded {len(self.class_names)} classes from metadata")
        else:
            print(f"⚠ No metadata.yaml found, using default classes")
        
        # Generate colors
        colors = generate_colors(len(self.class_names))
        self.class_colors = {self.class_names.get(i, f"class_{i}"): colors.get(i, (255,255,255)) 
                            for i in range(len(self.class_names))}
    
    def load(self) -> bool:
        """Load OpenVINO model"""
        if not HAS_OPENVINO:
            print("✗ OpenVINO not installed")
            return False
        
        try:
            print(f"Loading model: {self.model_path}")
            self._load_metadata()
            
            ie = Core()
            self.model = ie.compile_model(model=self.model_path, device_name="CPU")
            self.input_layer = self.model.input(0)
            self.output_layer = self.model.output(0)
            print("✓ Model loaded!")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        img = cv2.resize(image, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)
    
    def postprocess(
        self, 
        output, 
        orig_shape,
        conf_threshold: float = None,
        iou_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Process model output to get detections"""
        conf_threshold = conf_threshold or DEFAULT_CONFIDENCE
        iou_threshold = iou_threshold or DEFAULT_IOU_THRESHOLD
        
        predictions = output[0].T
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        
        boxes, scores, class_ids = [], [], []
        
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
        detections = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
            if len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    detections.append({
                        "class_id": class_ids[i],
                        "class_name": self.class_names.get(class_ids[i], f"class_{class_ids[i]}"),
                        "confidence": scores[i],
                        "x": (box[0] + box[2]) // 2,
                        "y": (box[1] + box[3]) // 2,
                        "box": box
                    })
        
        return detections
    
    def detect(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        """Run detection on frame"""
        input_tensor = self.preprocess(frame)
        output = self.model([input_tensor])[self.output_layer]
        return self.postprocess(output, frame.shape, conf_threshold)
    
    def detect_grouped(self, frame: np.ndarray, conf_threshold: float = None) -> Dict[str, List]:
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
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            box = det.get("box", [det["x"]-20, det["y"]-20, det["x"]+20, det["y"]+20])
            x1, y1, x2, y2 = box
            
            class_name = det["class_name"]
            color = self.class_colors.get(class_name, (255, 255, 255))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame

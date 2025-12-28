# -*- coding: utf-8 -*-
"""
Realtime PvZ detection using YOLO + OpenVINO
"""

import cv2
import time
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.window_capture import PvZWindowCapture
from inference.yolo_detector import YOLODetector


class PvZRealtimeDetection:
    def __init__(self, model_path="models/yolo/pvz_openvino/best.xml"):
        self.detector = YOLODetector(model_path)
        self.window_capture = PvZWindowCapture()
    
    def start(self):
        """Start realtime detection"""
        print("=" * 60)
        print("PVZ REALTIME DETECTION - YOLO11 + OpenVINO")
        print("=" * 60)
        
        if not self.detector.load():
            return
        
        if not self.window_capture.find_window():
            print("\nPlease:")
            print("1. Open Plants vs Zombies")
            print("2. Run this script again")
            return
        
        print("\n✓ Starting realtime detection!")
        print("Press 'q' on display window to quit\n")
        
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        
        try:
            while True:
                frame = self.window_capture.capture()
                if frame is None:
                    continue
                
                # Detect
                detections = self.detector.detect(frame)
                
                # Draw
                frame = self.detector.draw_detections(frame, detections)
                
                # FPS
                fps_counter += 1
                if time.time() - fps_time > 1:
                    fps = fps_counter
                    fps_counter = 0
                    fps_time = time.time()
                
                cv2.putText(frame, f"FPS: {fps} | Objects: {len(detections)}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("PvZ Detection", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n\nStopped!")
        finally:
            cv2.destroyAllWindows()


def main():
    detector = PvZRealtimeDetection(model_path="models/yolo/pvz_openvino/best.xml")
    detector.start()


if __name__ == "__main__":
    main()

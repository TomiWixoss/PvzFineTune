# -*- coding: utf-8 -*-
"""
Realtime Detection - Detection từ game window
"""

import cv2
import time

from ..core.config import Config
from ..utils.window_capture import WindowCapture
from ..inference.yolo_detector import YOLODetector


class RealtimeDetection:
    """Realtime detection từ game window"""
    
    def __init__(self, model_path: str = None):
        self.detector = YOLODetector(model_path)
        self.window = WindowCapture()
    
    def start(self):
        print("=" * 60)
        print("PVZ REALTIME DETECTION - YOLO11 + OpenVINO")
        print("=" * 60)
        
        if not self.detector.load():
            return
        
        if not self.window.find_window():
            print("\nPlease open Plants vs Zombies first!")
            return
        
        print("\n✓ Starting! Press 'q' to quit\n")
        
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        
        try:
            while True:
                frame = self.window.capture()
                if frame is None:
                    continue
                
                detections = self.detector.detect(frame)
                frame = self.detector.draw_detections(frame, detections)
                
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
            print("\nStopped!")
        finally:
            cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PvZ Realtime Detection')
    parser.add_argument('-m', '--model', help='YOLO model path')
    args = parser.parse_args()
    
    detector = RealtimeDetection(model_path=args.model)
    detector.start()


if __name__ == "__main__":
    main()

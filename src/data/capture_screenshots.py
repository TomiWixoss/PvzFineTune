# -*- coding: utf-8 -*-
"""
Auto capture PvZ screenshots for YOLO dataset
"""

import cv2
import os
import time
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.window_capture import PvZWindowCapture


class PvZScreenCapture:
    def __init__(self, output_dir="data/raw/pvz_screenshots", interval=0.5):
        """
        Initialize screenshot capture tool
        
        Args:
            output_dir: Output directory for images
            interval: Capture interval in seconds
        """
        self.output_dir = output_dir
        self.interval = interval
        self.capture_count = 0
        self.window_capture = PvZWindowCapture()
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    def save_screenshot(self, frame):
        """Save captured frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pvz_frame_{self.capture_count:04d}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(filepath, frame)
        self.capture_count += 1
        print(f"Saved: {filename} (Total: {self.capture_count})")
    
    def start(self):
        """Start auto capture"""
        print("=" * 50)
        print("PVZ SCREENSHOT CAPTURE FOR YOLOV11")
        print("=" * 50)
        
        if not self.window_capture.find_window():
            print("\nPlease:")
            print("1. Open Plants vs Zombies")
            print("2. Run this script again")
            return
        
        print(f"\nCapturing every {self.interval} seconds...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                frame = self.window_capture.capture()
                if frame is not None:
                    self.save_screenshot(frame)
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print(f"\n\nStopped! Total captured: {self.capture_count} images")
            print(f"Saved at: {os.path.abspath(self.output_dir)}")


def main():
    capturer = PvZScreenCapture(
        output_dir="data/raw/pvz_dataset",
        interval=0.5
    )
    capturer.start()


if __name__ == "__main__":
    main()

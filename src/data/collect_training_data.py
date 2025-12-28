# -*- coding: utf-8 -*-
"""
Collect training data for FunctionGemma from YOLO detection

Logic:
- Sun on screen → collect_sun (click sun position)
- Zombie + pea_shooter_pack ready → plant_pea_shooter at zombie row
- Nothing to do → do_nothing
"""

import cv2
import numpy as np
import json
import os
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.window_capture import PvZWindowCapture
from inference.yolo_detector import YOLODetector


class PvZDataCollector:
    def __init__(self, model_path="models/yolo/pvz_openvino/best.xml", 
                 output_file="data/processed/training_data.json"):
        self.model_path = model_path
        self.output_file = output_file
        self.window_capture = PvZWindowCapture()
        self.detector = None
        self.data = []
        
        # Load existing data if available
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✓ Loaded {len(self.data)} samples from {output_file}")
        
        # PvZ 800x600 - Grid 5 rows
        self.row_boundaries = [109, 189, 269, 349, 429, 509]
    
    def get_row_from_y(self, y):
        """Calculate row (1-5) from pixel y"""
        for i in range(5):
            if y < self.row_boundaries[i + 1]:
                return i + 1
        return 5
    
    def generate_game_state(self, detections):
        """Generate game state from detections"""
        suns = []
        zombies = []
        pea_shooters = []
        can_plant = False
        
        for det in detections:
            name = det["class_name"]
            x, y = det["x"], det["y"]
            
            if name == "sun":
                suns.append({"x": x, "y": y})
            elif name == "zombie":
                row = self.get_row_from_y(y)
                zombies.append({"row": row, "x": x, "y": y})
            elif name == "pea_shooter":
                row = self.get_row_from_y(y)
                pea_shooters.append({"row": row})
            elif name == "pea_shooter_pack_yes":
                can_plant = True
        
        # Create description
        parts = []
        
        if suns:
            parts.append(f"Sun at ({suns[0]['x']}, {suns[0]['y']})")
        else:
            parts.append("No sun")
        
        if zombies:
            zombie_rows = [z['row'] for z in zombies]
            parts.append(f"Zombie row {zombie_rows}")
        else:
            parts.append("No zombie")
        
        if pea_shooters:
            pea_rows = list(set([p['row'] for p in pea_shooters]))
            parts.append(f"Pea at row {pea_rows}")
        
        parts.append(f"Can plant: {can_plant}")
        
        return ". ".join(parts), suns, zombies, can_plant
    
    def save_data(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(self.data)} samples to {self.output_file}")
    
    def start(self, interval=2.0):
        print("=" * 60)
        print("TRAINING DATA COLLECTION")
        print("=" * 60)
        
        self.detector = YOLODetector(self.model_path)
        if not self.detector.load():
            return
        
        if not self.window_capture.find_window():
            return
        
        print(f"\nCollecting every {interval} seconds")
        print("'c' - Capture now")
        print("'s' - Skip")
        print("'q' - Stop and save\n")
        
        last_capture = 0
        
        try:
            while True:
                frame = self.window_capture.capture()
                if frame is None:
                    continue
                
                # Detect
                detections = self.detector.detect(frame)
                
                # Draw detections
                for det in detections:
                    x, y = det["x"], det["y"]
                    color = (0, 0, 255) if det["class_name"] == "zombie" else (0, 255, 0)
                    if det["class_name"] == "sun":
                        color = (0, 255, 255)
                    cv2.circle(frame, (x, y), 20, color, 2)
                    cv2.putText(frame, det["class_name"], (x-30, y-25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.putText(frame, f"Samples: {len(self.data)} | 'c' capture, 'q' quit",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("Data Collector", frame)
                
                key = cv2.waitKey(1) & 0xFF
                current_time = time.time()
                should_capture = (current_time - last_capture >= interval) or (key == ord('c'))
                
                if should_capture and key != ord('s'):
                    game_state, suns, zombies, can_plant = self.generate_game_state(detections)
                    
                    sample = {
                        "id": len(self.data) + 1,
                        "game_state": game_state,
                        "suns": suns,
                        "zombies": zombies,
                        "can_plant": can_plant,
                        "action": "",
                        "arguments": {}
                    }
                    
                    self.data.append(sample)
                    last_capture = current_time
                    print(f"[{len(self.data)}] {game_state}")
                
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.save_data()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            print("LABELING INSTRUCTIONS")
            print("=" * 60)
            print(f"""
Open '{self.output_file}' and fill:

1. Has sun → action: "collect_sun", arguments: {{"x": ..., "y": ...}}
2. Has zombie + can_plant=true → action: "plant_pea_shooter", arguments: {{"row": ...}}
3. Nothing to do → action: "do_nothing", arguments: {{}}
""")


def main():
    collector = PvZDataCollector(
        model_path="models/yolo/pvz_openvino/best.xml",
        output_file="data/processed/training_data.json"
    )
    collector.start(interval=2.0)


if __name__ == "__main__":
    main()

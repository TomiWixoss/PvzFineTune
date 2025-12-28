# -*- coding: utf-8 -*-
"""
PvZ Auto Play Bot
- Rule-based: Collect sun (instant)
- AI: Plant decisions (optional)
"""

import cv2
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    YOLO_MODEL_PATH, GEMMA_MODEL_PATH,
    GRID_ROW_Y, GRID_COLUMNS_X,
    SUN_COLLECT_COOLDOWN, PLANT_COOLDOWN, TARGET_FPS
)
from utils.window_capture import PvZWindowCapture
from inference.yolo_detector import YOLODetector
from inference.gemma_inference import GemmaInference


class PvZController:
    """Game controller for clicking and tracking"""
    
    def __init__(self, window_capture: PvZWindowCapture):
        self.window = window_capture
        self.collected_suns = []  # [(x, y, time), ...]
    
    def is_sun_collected(self, x: int, y: int) -> bool:
        """Check if sun was recently clicked"""
        now = time.time()
        self.collected_suns = [(sx, sy, t) for sx, sy, t in self.collected_suns if now - t < 1.5]
        
        for sx, sy, t in self.collected_suns:
            if abs(sx - x) < 60 and abs(sy - y) < 60:
                return True
        return False
    
    def collect_sun(self, x: int, y: int) -> bool:
        """Collect sun with tracking"""
        if self.is_sun_collected(x, y):
            return False
        self.window.click(x, y)
        self.collected_suns.append((x, y, time.time()))
        return True
    
    def plant_at_empty_slot(self, seed_pos: tuple, existing_plants: list) -> bool:
        """Plant at first empty slot"""
        self.window.click(seed_pos[0], seed_pos[1])
        time.sleep(0.15)
        
        for col_x in GRID_COLUMNS_X:
            is_occupied = any(abs(p["x"] - col_x) < 40 for p in existing_plants)
            if not is_occupied:
                print(f"  → Planting at ({col_x}, {GRID_ROW_Y})")
                self.window.click(col_x, GRID_ROW_Y)
                return True
        
        print("  → All columns occupied!")
        return False


class PvZAutoPlay:
    """Main auto-play bot"""
    
    def __init__(self, yolo_path: str = None, gemma_path: str = None):
        self.window_capture = PvZWindowCapture()
        self.detector = YOLODetector(yolo_path or YOLO_MODEL_PATH)
        self.ai = GemmaInference(gemma_path or GEMMA_MODEL_PATH)
        self.controller = None
        
        self.last_sun_collect = 0
        self.last_plant_time = 0
    
    def run(self):
        print("=" * 50)
        print("PVZ AUTO PLAY BOT")
        print("=" * 50)
        
        # Load YOLO model
        if not self.detector.load():
            return
        
        # Load AI model (optional)
        if not self.ai.load():
            print("⚠ AI model not loaded, using rule-based only")
        
        # Find game window
        if not self.window_capture.find_window():
            print("\n✗ Open PvZ game first!")
            return
        
        self.controller = PvZController(self.window_capture)
        
        print("\n✓ Running! Press 'q' to quit\n")
        
        fps_counter, fps_time, fps = 0, time.time(), 0
        frame_time = 1.0 / TARGET_FPS
        
        try:
            while True:
                loop_start = time.time()
                frame = self.window_capture.capture()
                if frame is None:
                    continue
                
                # Detection
                det = self.detector.detect_grouped(frame)
                
                suns = det.get("sun", [])
                zombies = det.get("zombie", [])
                plants = det.get("pea_shooter", [])
                seed_ready = det.get("pea_shooter_ready", [])
                sunflower_reward = det.get("sunflower_reward", [])
                
                now = time.time()
                
                # === Collect sun ===
                if suns and now - self.last_sun_collect > SUN_COLLECT_COOLDOWN:
                    sun = suns[0]
                    if self.controller.collect_sun(sun["x"], sun["y"]):
                        self.last_sun_collect = now
                        print(f"[SUN] Collected at ({sun['x']}, {sun['y']})")
                
                # === Plant decision ===
                can_plant = len(seed_ready) > 0
                has_zombie = len(zombies) > 0
                
                if can_plant and now - self.last_plant_time > PLANT_COOLDOWN:
                    should_plant = True
                    if self.ai.model is not None:
                        should_plant = self.ai.should_plant(has_zombie, can_plant)
                    
                    if should_plant:
                        seed_pos = (seed_ready[0]["x"], seed_ready[0]["y"])
                        self.controller.plant_at_empty_slot(seed_pos, plants)
                        self.last_plant_time = now
                
                # === Click sunflower reward if visible ===
                if sunflower_reward:
                    reward = sunflower_reward[0]
                    self.window_capture.click(reward["x"], reward["y"])
                    print("[REWARD] Clicked sunflower reward")
                
                # Draw & display
                self._draw_frame(frame, det, fps)
                cv2.imshow("PvZ Auto", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # FPS control
                fps_counter += 1
                if time.time() - fps_time > 1:
                    fps = fps_counter
                    fps_counter = 0
                    fps_time = time.time()
                
                elapsed = time.time() - loop_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                    
        finally:
            cv2.destroyAllWindows()
    
    def _draw_frame(self, frame, det, fps):
        """Draw detections on frame"""
        colors = {
            "sun": (0, 255, 255),
            "zombie": (0, 0, 255),
            "pea_shooter": (0, 255, 0),
            "pea_shooter_ready": (0, 255, 0),
        }
        
        for class_name, items in det.items():
            color = colors.get(class_name, (255, 255, 255))
            for item in items:
                cv2.circle(frame, (item["x"], item["y"]), 15, color, 2)
                cv2.putText(frame, class_name.split("_")[0].upper(), 
                           (item["x"]-20, item["y"]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Status bar
        suns = len(det.get("sun", []))
        zombies = len(det.get("zombie", []))
        plants = len(det.get("pea_shooter", []))
        ready = len(det.get("pea_shooter_ready", []))
        
        status = f"FPS:{fps} | Sun:{suns} | Zombie:{zombies} | Plant:{plants} | Ready:{ready}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PvZ Auto Play Bot')
    parser.add_argument('-m', '--model', help='YOLO model path')
    parser.add_argument('-g', '--gemma', help='Gemma model path')
    args = parser.parse_args()
    
    bot = PvZAutoPlay(yolo_path=args.model, gemma_path=args.gemma)
    bot.run()


if __name__ == "__main__":
    main()

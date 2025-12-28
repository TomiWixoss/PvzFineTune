# -*- coding: utf-8 -*-
"""
PvZ Auto Play - Hybrid Mode
- Rule-based: Collect sun (fast)
- AI: Plant decisions
"""

import cv2
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.window_capture import PvZWindowCapture
from inference.yolo_detector import YOLODetector
from inference.gemma_inference import GemmaInference


class PvZController:
    """Game controller for clicking and tracking"""
    
    # Level 1 grid config
    ROW_Y = 355
    COLUMNS_X = [75, 154, 229, 312, 393, 476, 557, 638, 732]
    
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
        # Click seed packet
        self.window.click(seed_pos[0], seed_pos[1])
        time.sleep(0.15)
        
        # Find empty column
        for col_x in self.COLUMNS_X:
            is_occupied = any(abs(p["x"] - col_x) < 40 for p in existing_plants)
            if not is_occupied:
                print(f"  → Planting at ({col_x}, {self.ROW_Y})")
                self.window.click(col_x, self.ROW_Y)
                return True
        
        print("  → All columns occupied!")
        return False


class PvZAutoPlay:
    """Main auto-play bot"""
    
    def __init__(self, 
                 yolo_path="models/yolo/pvz_openvino/best.xml",
                 gemma_path="models/gemma/pvz_functiongemma_final"):
        self.window_capture = PvZWindowCapture()
        self.detector = YOLODetector(yolo_path)
        self.ai = GemmaInference(gemma_path)
        self.controller = None
        
        self.last_sun_collect = 0
        self.last_plant_time = 0
    
    def run(self):
        print("=" * 50)
        print("PVZ AUTO PLAY - HYBRID MODE")
        print("  Sun: Rule-based (instant)")
        print("  Plant: AI decision")
        print("=" * 50)
        
        # Load models
        if not self.detector.load():
            return
        if not self.ai.load():
            print("Warning: AI model not loaded, using rule-based only")
        
        # Find game window
        if not self.window_capture.find_window():
            print("Open PvZ first!")
            return
        
        self.controller = PvZController(self.window_capture)
        
        print("\n✓ Running! Press 'q' to quit\n")
        
        fps_counter, fps_time, fps = 0, time.time(), 0
        target_fps = 30
        frame_time = 1.0 / target_fps
        
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
                seed_yes = det.get("pea_shooter_pack_yes", [])
                
                now = time.time()
                
                # === RULE-BASED: Collect sun immediately ===
                if suns and now - self.last_sun_collect > 0.1:
                    sun = suns[0]
                    if self.controller.collect_sun(sun["x"], sun["y"]):
                        self.last_sun_collect = now
                        print(f"[SUN] Collected at ({sun['x']}, {sun['y']})")
                
                # === AI: Plant decision ===
                can_plant = len(seed_yes) > 0
                has_zombie = len(zombies) > 0
                
                if can_plant and now - self.last_plant_time > 1.5:
                    should_plant = True
                    if self.ai.model is not None:
                        should_plant = self.ai.should_plant(has_zombie, can_plant)
                    
                    if should_plant:
                        seed_pos = (seed_yes[0]["x"], seed_yes[0]["y"])
                        self.controller.plant_at_empty_slot(seed_pos, plants)
                        self.last_plant_time = now
                
                # Draw
                self._draw_frame(frame, suns, zombies, plants, seed_yes, fps, can_plant)
                
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
    
    def _draw_frame(self, frame, suns, zombies, plants, seed_yes, fps, can_plant):
        """Draw detections on frame"""
        for sun in suns:
            cv2.circle(frame, (sun["x"], sun["y"]), 15, (0, 255, 255), 2)
            cv2.putText(frame, "SUN", (sun["x"]-15, sun["y"]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        for z in zombies:
            cv2.circle(frame, (z["x"], z["y"]), 15, (0, 0, 255), 2)
            cv2.putText(frame, "ZOMBIE", (z["x"]-25, z["y"]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        for p in plants:
            cv2.circle(frame, (p["x"], p["y"]), 10, (0, 255, 0), 2)
        
        for s in seed_yes:
            cv2.rectangle(frame, (s["x"]-20, s["y"]-20), (s["x"]+20, s["y"]+20), (0, 255, 0), 2)
            cv2.putText(frame, "READY", (s["x"]-20, s["y"]-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        
        status = f"FPS:{fps} | Suns:{len(suns)} | Zombies:{len(zombies)} | Plants:{len(plants)} | CanPlant:{can_plant}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    bot = PvZAutoPlay(
        yolo_path="models/yolo/pvz_openvino/best.xml",
        gemma_path="models/gemma/pvz_functiongemma_final"
    )
    bot.run()


if __name__ == "__main__":
    main()

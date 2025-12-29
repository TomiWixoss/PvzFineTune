# -*- coding: utf-8 -*-
"""
PvZ Auto Play Bot - AI-driven with OpenVINO
Actions: plant(plant_type, row, col) or wait()
Sun collection handled by rule-based logic
"""

import cv2
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    YOLO_MODEL_PATH, GEMMA_MODEL_PATH,
    GRID_ROWS_Y, GRID_COLUMNS_X,
    TARGET_FPS
)
from utils.window_capture import PvZWindowCapture
from inference.yolo_detector import YOLODetector
from inference.gemma_inference import GemmaInference


class PvZController:
    """Game controller for clicking"""
    
    def __init__(self, window_capture: PvZWindowCapture):
        self.window = window_capture
    
    def collect_sun(self, x: int, y: int):
        """Collect sun at pixel position (rule-based)"""
        self.window.click(x, y)
        print(f"[RULE] Collect sun at ({x}, {y})")
    
    def plant_at_grid(self, seed_pos: tuple, row: int, col: int, plant_type: str = "pea_shooter"):
        """Plant at grid position (row, col)"""
        if row < 0 or row >= len(GRID_ROWS_Y) or col < 0 or col >= len(GRID_COLUMNS_X):
            print(f"[AI] Invalid grid position: row={row}, col={col}")
            return False
        
        # Click seed packet
        self.window.click(seed_pos[0], seed_pos[1])
        time.sleep(0.15)
        
        # Click grid position
        x = GRID_COLUMNS_X[col]
        y = GRID_ROWS_Y[row]
        self.window.click(x, y)
        print(f"[AI] Plant {plant_type} at row={row}, col={col}")
        return True


class PvZAutoPlay:
    """Main auto-play bot - AI driven"""
    
    def __init__(self, yolo_path: str = None, gemma_path: str = None):
        self.window_capture = PvZWindowCapture()
        self.detector = YOLODetector(yolo_path or YOLO_MODEL_PATH)
        self.ai = GemmaInference(gemma_path or GEMMA_MODEL_PATH)
        self.controller = None
        

    
    def _build_game_state(self, det: dict) -> tuple:
        """Build game state from detections for AI"""
        plants_raw = det.get("pea_shooter", [])
        zombies_raw = det.get("zombie", [])
        seed_ready = det.get("pea_shooter_ready", [])
        seed_cooldown = det.get("pea_shooter_cooldown", [])
        
        # Build plants list with row/col
        plants = []
        for p in plants_raw:
            plants.append({
                "type": "pea_shooter",
                "row": self._get_row(p["y"]),
                "col": self._get_col(p["x"]),
                "x": p["x"], "y": p["y"]
            })
        
        # Build zombies list with row/col
        zombies = []
        for z in zombies_raw:
            zombies.append({
                "type": "zombie",
                "row": self._get_row(z["y"]),
                "col": self._get_col(z["x"]),
                "x": z["x"], "y": z["y"]
            })
        
        # Build seeds list with status
        seeds = []
        for s in seed_ready:
            seeds.append({"type": "pea_shooter", "status": "ready", "x": s["x"], "y": s["y"]})
        for s in seed_cooldown:
            seeds.append({"type": "pea_shooter", "status": "cooldown", "x": s["x"], "y": s["y"]})
        
        return plants, zombies, seeds
    
    def _get_row(self, y: int) -> int:
        """Y coordinate → row index (0-4)"""
        min_dist = float('inf')
        row = 0
        for i, row_y in enumerate(GRID_ROWS_Y):
            if abs(y - row_y) < min_dist:
                min_dist = abs(y - row_y)
                row = i
        return row
    
    def _get_col(self, x: int) -> int:
        """X coordinate → col index (0-8)"""
        min_dist = float('inf')
        col = 0
        for i, col_x in enumerate(GRID_COLUMNS_X):
            if abs(x - col_x) < min_dist:
                min_dist = abs(x - col_x)
                col = i
        return col
    
    def run(self):
        print("=" * 50)
        print("PVZ AUTO PLAY BOT - AI MODE (OpenVINO)")
        print("=" * 50)
        
        # Load YOLO model
        if not self.detector.load():
            return
        
        # Load AI model
        if not self.ai.load():
            print("✗ AI model required!")
            return
        
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
                sunflower_reward = det.get("sunflower_reward", [])
                seed_ready = det.get("pea_shooter_ready", [])
                
                # === RULE-BASED: Auto collect sun ===
                for sun in suns:
                    self.controller.collect_sun(sun["x"], sun["y"])
                    time.sleep(0.05)  # Small delay between clicks
                
                # === RULE-BASED: Click sunflower reward ===
                if sunflower_reward:
                    reward = sunflower_reward[0]
                    self.window_capture.click(reward["x"], reward["y"])
                    print("[RULE] Clicked sunflower reward")
                
                # === AI Decision (no delay, runs every frame) ===
                plants, zombies, seeds = self._build_game_state(det)
                game_state = GemmaInference.create_game_state(plants, zombies, seeds)
                
                action, args = self.ai.get_action(game_state)
                
                if action == "plant" and seed_ready:
                    plant_type = args.get("plant_type", "pea_shooter")
                    row = args.get("row", 2)
                    col = args.get("col", 0)
                    seed_pos = (seed_ready[0]["x"], seed_ready[0]["y"])
                    self.controller.plant_at_grid(seed_pos, row, col, plant_type)
                
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
            "pea_shooter_ready": (0, 200, 0),
            "pea_shooter_cooldown": (128, 128, 128),
        }
        
        for class_name, items in det.items():
            color = colors.get(class_name, (255, 255, 255))
            for item in items:
                cv2.circle(frame, (item["x"], item["y"]), 15, color, 2)
                label = class_name.replace("_", " ").upper()[:8]
                cv2.putText(frame, label, (item["x"]-20, item["y"]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Status bar
        suns = len(det.get("sun", []))
        zombies = len(det.get("zombie", []))
        plants = len(det.get("pea_shooter", []))
        ready = len(det.get("pea_shooter_ready", []))
        cooldown = len(det.get("pea_shooter_cooldown", []))
        
        status = f"FPS:{fps} | Sun:{suns} | Zombie:{zombies} | Plant:{plants} | Seed:{'R' if ready else 'C'}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PvZ Auto Play Bot - AI Mode')
    parser.add_argument('-m', '--model', help='YOLO model path')
    parser.add_argument('-g', '--gemma', help='Gemma model path')
    args = parser.parse_args()
    
    bot = PvZAutoPlay(yolo_path=args.model, gemma_path=args.gemma)
    bot.run()


if __name__ == "__main__":
    main()

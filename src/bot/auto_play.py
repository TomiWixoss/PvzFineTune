# -*- coding: utf-8 -*-
"""
PvZ Auto Play Bot - AI-driven (no rule-based logic)
All decisions made by FunctionGemma
"""

import cv2
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    YOLO_MODEL_PATH, GEMMA_MODEL_PATH,
    GRID_ROWS_Y, GRID_COLUMNS_X,
    AI_INFERENCE_DELAY, SUN_FALL_DELAY, PLANT_COOLDOWN, TARGET_FPS
)
from utils.window_capture import PvZWindowCapture
from inference.yolo_detector import YOLODetector
from inference.gemma_inference import GemmaInference


class PvZController:
    """Game controller for clicking"""
    
    def __init__(self, window_capture: PvZWindowCapture):
        self.window = window_capture
    
    def collect_sun(self, x: int, y: int, pos_type: str = ""):
        """Collect sun at pixel position"""
        self.window.click(x, y)
        print(f"[AI] Collect sun at ({x}, {y}) [{pos_type}]")
    
    def plant_at_grid(self, seed_pos: tuple, row: int, col: int):
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
        print(f"[AI] Plant at row={row}, col={col} ({x}, {y})")
        return True


class PvZAutoPlay:
    """Main auto-play bot - AI driven"""
    
    def __init__(self, yolo_path: str = None, gemma_path: str = None):
        self.window_capture = PvZWindowCapture()
        self.detector = YOLODetector(yolo_path or YOLO_MODEL_PATH)
        self.ai = GemmaInference(gemma_path or GEMMA_MODEL_PATH)
        self.controller = None
        
        self.last_action_time = 0
        self.last_plant_time = 0
    
    def run(self):
        print("=" * 50)
        print("PVZ AUTO PLAY BOT - AI MODE")
        print("=" * 50)
        
        # Load YOLO model
        if not self.detector.load():
            return
        
        # Load AI model (required)
        if not self.ai.load():
            print("✗ AI model required! Cannot run without Gemma.")
            return
        
        # Find game window
        if not self.window_capture.find_window():
            print("\n✗ Open PvZ game first!")
            return
        
        self.controller = PvZController(self.window_capture)
        
        print("\n✓ Running! Press 'q' to quit\n")
        print(f"  AI Inference Delay: {AI_INFERENCE_DELAY}s")
        print(f"  Sun Fall Delay: {SUN_FALL_DELAY}s")
        print()
        
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
                
                # === AI Decision (with delay for inference time) ===
                if now - self.last_action_time > AI_INFERENCE_DELAY:
                    game_state = GemmaInference.create_game_state(
                        suns=suns,
                        zombies=zombies,
                        plants=plants,
                        can_plant=len(seed_ready) > 0
                    )
                    
                    action, args = self.ai.get_action(game_state)
                    
                    if action == "collect_sun" and suns:
                        # Click vị trí gốc (lúc detect)
                        x_orig = args.get("x", suns[0]["x"])
                        y_orig = args.get("y", suns[0]["y"])
                        self.controller.collect_sun(x_orig, y_orig, "original")
                        
                        # Wait for sun to fall
                        time.sleep(SUN_FALL_DELAY)
                        
                        # Click vị trí sau delay (sun đã rơi)
                        # Re-detect để lấy vị trí mới
                        frame_new = self.window_capture.capture()
                        if frame_new is not None:
                            det_new = self.detector.detect_grouped(frame_new)
                            suns_new = det_new.get("sun", [])
                            if suns_new:
                                self.controller.collect_sun(suns_new[0]["x"], suns_new[0]["y"], "delayed")
                        
                        self.last_action_time = now
                    
                    elif action == "plant_pea_shooter" and seed_ready:
                        if now - self.last_plant_time > PLANT_COOLDOWN:
                            row = args.get("row", 0)
                            col = args.get("col", 0)
                            seed_pos = (seed_ready[0]["x"], seed_ready[0]["y"])
                            self.controller.plant_at_grid(seed_pos, row, col)
                            self.last_plant_time = now
                            self.last_action_time = now
                    
                    elif action == "do_nothing":
                        pass  # AI chose to wait
                
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
    parser = argparse.ArgumentParser(description='PvZ Auto Play Bot - AI Mode')
    parser.add_argument('-m', '--model', help='YOLO model path')
    parser.add_argument('-g', '--gemma', help='Gemma model path')
    args = parser.parse_args()
    
    bot = PvZAutoPlay(yolo_path=args.model, gemma_path=args.gemma)
    bot.run()


if __name__ == "__main__":
    main()

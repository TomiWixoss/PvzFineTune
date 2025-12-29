# -*- coding: utf-8 -*-
"""
Auto Play Bot - AI-driven với OpenVINO
Priority: AI action > Sun collection
"""

import cv2
import time
from typing import Dict, List, Tuple

from ..core.config import Config
from ..core.constants import TARGET_FPS, SUN_CLICK_DELAY
from ..utils.window_capture import WindowCapture
from ..utils.grid_utils import get_row, get_col
from ..inference.yolo_detector import YOLODetector
from ..inference.gemma_inference import GemmaInference
from .controller import GameController


class AutoPlayBot:
    """Main auto-play bot - AI driven"""
    
    def __init__(self, yolo_path: str = None, gemma_path: str = None):
        self.window = WindowCapture()
        self.detector = YOLODetector(yolo_path)
        self.ai = GemmaInference(gemma_path)
        self.controller = None
    
    def _build_game_state(self, det: Dict) -> Tuple[List, List, List]:
        """Build game state from detections"""
        grid_rows = Config.GRID_ROWS_Y
        grid_cols = Config.GRID_COLUMNS_X
        
        plants = [{"type": "pea_shooter", "row": get_row(p["y"], grid_rows), "col": get_col(p["x"], grid_cols)} 
                  for p in det.get("pea_shooter", [])]
        
        zombies = [{"type": "zombie", "row": get_row(z["y"], grid_rows), "col": get_col(z["x"], grid_cols)} 
                   for z in det.get("zombie", [])]
        
        seeds = []
        for s in det.get("pea_shooter_ready", []):
            seeds.append({"type": "pea_shooter", "status": "ready", "x": s["x"], "y": s["y"]})
        for s in det.get("pea_shooter_cooldown", []):
            seeds.append({"type": "pea_shooter", "status": "cooldown", "x": s["x"], "y": s["y"]})
        
        return plants, zombies, seeds
    
    def _collect_suns(self, suns: List[Dict]):
        """Thu thập tất cả sun"""
        for sun in suns:
            self.controller.collect_sun(sun["x"], sun["y"])
            time.sleep(SUN_CLICK_DELAY)
    
    def run(self):
        print("=" * 50)
        print("PVZ AUTO PLAY BOT - AI MODE (OpenVINO)")
        print("=" * 50)
        
        if not self.detector.load():
            return
        
        if not self.ai.load():
            print("✗ AI model required!")
            return
        
        if not self.window.find_window():
            print("\n✗ Open PvZ game first!")
            return
        
        self.controller = GameController(self.window)
        print("\n✓ Running! Press 'q' to quit\n")
        
        fps_counter, fps_time, fps = 0, time.time(), 0
        frame_time = 1.0 / TARGET_FPS
        ai_calls = 0
        
        try:
            while True:
                loop_start = time.time()
                frame = self.window.capture()
                if frame is None:
                    continue
                
                det = self.detector.detect_grouped(frame)
                ai_acted = False
                
                # AI Decision TRƯỚC - ưu tiên cao nhất
                seed_ready = det.get("pea_shooter_ready", [])
                if seed_ready:
                    plants, zombies, seeds = self._build_game_state(det)
                    game_state = GemmaInference.create_game_state(plants, zombies, seeds)
                    action, args = self.ai.get_action(game_state)
                    ai_calls += 1
                    
                    if action == "plant":
                        seed = seed_ready[0]
                        self.controller.plant_at_grid(
                            (seed["x"], seed["y"]),
                            args.get("row", 2),
                            args.get("col", 0),
                            args.get("plant_type", "pea_shooter")
                        )
                        ai_acted = True
                
                # Thu sun CHỈ KHI AI không có action
                if not ai_acted and det.get("sun"):
                    self._collect_suns(det["sun"])
                
                # Click sunflower reward
                if det.get("sunflower_reward"):
                    self.window.click(det["sunflower_reward"][0]["x"], det["sunflower_reward"][0]["y"])
                    print("[RULE] Clicked sunflower reward")
                
                # Draw & display
                self._draw_frame(frame, det, fps, ai_calls)
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
            print(f"\n📊 Total AI calls: {ai_calls}")
    
    def _draw_frame(self, frame, det: Dict, fps: int, ai_calls: int):
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
        
        seed_status = "READY" if det.get("pea_shooter_ready") else "CD"
        status = f"FPS:{fps} | AI:{ai_calls} | Seed:{seed_status} | Z:{len(det.get('zombie', []))}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PvZ Auto Play Bot')
    parser.add_argument('-m', '--model', help='YOLO model path')
    parser.add_argument('-g', '--gemma', help='Gemma model path')
    args = parser.parse_args()
    
    bot = AutoPlayBot(yolo_path=args.model, gemma_path=args.gemma)
    bot.run()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
PvZ Auto Play - Hybrid Mode
- Rule-based: Nhặt sun (nhanh)
- AI: Quyết định trồng cây
"""

import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time
import torch
import re
import threading
from queue import Queue
from openvino.runtime import Core
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import get_json_schema

# ============================================
# CONFIG
# ============================================
YOLO_MODEL_PATH = "pvz_openvino/best.xml"
LLM_MODEL_PATH = "pvz_functiongemma_final"
DETECTION_CONF = 0.5

# ============================================
# YOLO DETECTOR
# ============================================
class YOLODetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.class_names = {0: "pea_shooter", 1: "pea_shooter_pack_no", 
                           2: "pea_shooter_pack_yes", 3: "sun", 4: "zombie"}
        self.input_size = (640, 640)
        
    def load(self):
        ie = Core()
        self.model = ie.compile_model(model=self.model_path, device_name="CPU")
        self.input_layer = self.model.input(0)
        self.output_layer = self.model.output(0)
        print("✓ YOLO loaded!")
        
    def detect(self, frame):
        input_img = cv2.resize(frame, self.input_size)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_img = np.expand_dims(np.transpose(input_img, (2, 0, 1)), 0)
        
        output = self.model([input_img])[self.output_layer][0].T
        orig_h, orig_w = frame.shape[:2]
        scale_x, scale_y = orig_w / 640, orig_h / 640
        
        results = {name: [] for name in self.class_names.values()}
        for pred in output:
            x, y, w, h = pred[:4]
            class_id = np.argmax(pred[4:])
            conf = pred[4 + class_id]
            if conf > DETECTION_CONF:
                results[self.class_names[class_id]].append({
                    "x": int(x * scale_x), "y": int(y * scale_y), "conf": float(conf)
                })
        return results

# ============================================
# FUNCTIONGEMMA AI (cho quyết định plant)
# ============================================
class PlantAI:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.request_queue = Queue()
        self.result_queue = Queue()
        self.running = False
        self.last_decision = None
        self.last_decision_time = 0
        
    def load(self):
        print(f"Loading FunctionGemma on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)
        self._setup_tools()
        print("✓ FunctionGemma loaded!")
        
    def _setup_tools(self):
        def plant_pea_shooter() -> str:
            """Plant pea shooter."""
            return "ok"
        def do_nothing() -> str:
            """Wait."""
            return "ok"
        self.tools = [get_json_schema(plant_pea_shooter), get_json_schema(do_nothing)]
        self.system_msg = "You are a PvZ bot. Decide: plant_pea_shooter or do_nothing."
    
    def should_plant(self, has_zombie, can_plant):
        """Hỏi AI có nên plant không - luôn plant nếu có thể"""
        if not can_plant:
            return False
            
        # Cache decision 2 giây
        now = time.time()
        if self.last_decision is not None and now - self.last_decision_time < 2:
            return self.last_decision
        
        # Nếu có thể plant thì plant (không cần đợi zombie)
        game_state = f"NO_SUN. {'HAS_ZOMBIE' if has_zombie else 'NO_ZOMBIE'}. CAN_PLANT"
        
        messages = [
            {"role": "developer", "content": self.system_msg},
            {"role": "user", "content": game_state},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tools=self.tools, add_generation_prompt=True, 
            return_dict=True, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=32)
        output = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)
        
        # Luôn plant nếu có thể (override AI nếu cần)
        decision = True  # Luôn plant khi có thể
        self.last_decision = decision
        self.last_decision_time = now
        
        print(f"  [AI] {game_state} -> PLANT!")
        return decision

# ============================================
# GAME CONTROLLER
# ============================================
class PvZController:
    def __init__(self):
        self.window = None
        self.offset_x = 0
        self.offset_y = 0
        self.planted_positions = set()
        self.collected_suns = []  # Track sun đã nhặt [(x, y, time), ...]
        
    def find_window(self):
        for title in gw.getAllTitles():
            if any(n.lower() in title.lower() for n in ["Plants vs. Zombies", "Plants vs Zombies", "popcapgame1"]):
                self.window = gw.getWindowsWithTitle(title)[0]
                self.offset_x, self.offset_y = self.window.left, self.window.top
                print(f"✓ Found: {title}")
                return True
        return False
    
    def capture(self):
        try:
            shot = pyautogui.screenshot(region=(self.window.left, self.window.top, self.window.width, self.window.height))
            return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
        except:
            return None
    
    def click(self, x, y):
        pyautogui.click(self.offset_x + x, self.offset_y + y)
    
    def is_sun_already_collected(self, x, y):
        """Kiểm tra sun này đã click chưa (trong 1.5s gần đây)"""
        now = time.time()
        # Xóa sun cũ
        self.collected_suns = [(sx, sy, t) for sx, sy, t in self.collected_suns if now - t < 1.5]
        # Kiểm tra
        for sx, sy, t in self.collected_suns:
            if abs(sx - x) < 60 and abs(sy - y) < 60:
                return True
        return False
        
    def collect_sun(self, x, y):
        """Nhặt sun - có tracking tránh click lại"""
        if self.is_sun_already_collected(x, y):
            return False
        self.click(x, y)
        self.collected_suns.append((x, y, time.time()))
        return True
        
    def plant_at_zombie_row(self, seed_pack_pos, zombie_y, existing_plants):
        """
        Trồng cây ở row có zombie
        Màn 1 chỉ có 1 row, 9 cột
        """
        # Click seed packet
        self.click(seed_pack_pos[0], seed_pack_pos[1])
        time.sleep(0.15)
        
        # Tọa độ từ calibration
        ROW_Y = 355  # Y trung bình của các slot
        COLUMNS_X = [75, 154, 229, 312, 393, 476, 557, 638, 732]  # 9 cột
        
        # Tìm cột trống (không có plant)
        for col_x in COLUMNS_X:
            is_occupied = False
            for plant in existing_plants:
                # Nếu plant gần cột này (trong 40px)
                if abs(plant["x"] - col_x) < 40:
                    is_occupied = True
                    break
            
            if not is_occupied:
                print(f"  → Planting at ({col_x}, {ROW_Y})")
                self.click(col_x, ROW_Y)
                return True
        
        print("  → All columns occupied!")
        return False

# ============================================
# MAIN - HYBRID MODE
# ============================================
class PvZAutoPlay:
    def __init__(self):
        self.detector = YOLODetector(YOLO_MODEL_PATH)
        self.ai = PlantAI(LLM_MODEL_PATH)
        self.controller = PvZController()
        self.last_sun_collect = 0
        self.last_plant_time = 0
        
    def run(self):
        print("=" * 50)
        print("PVZ AUTO PLAY - HYBRID MODE")
        print("  Sun: Rule-based (instant)")
        print("  Plant: AI decision")
        print("=" * 50)
        
        self.detector.load()
        self.ai.load()
        
        if not self.controller.find_window():
            print("Open PvZ first!")
            return
        
        print("\n✓ Running! Press 'q' to quit\n")
        
        fps_counter, fps_time, fps = 0, time.time(), 0
        target_fps = 30  # Giảm xuống 30fps
        frame_time = 1.0 / target_fps
        
        try:
            while True:
                loop_start = time.time()
                frame = self.controller.capture()
                if frame is None:
                    continue
                
                # Detection
                det = self.detector.detect(frame)
                
                suns = det.get("sun", [])
                zombies = det.get("zombie", [])
                plants = det.get("pea_shooter", [])
                seed_yes = det.get("pea_shooter_pack_yes", [])
                seed_no = det.get("pea_shooter_pack_no", [])
                
                now = time.time()
                
                # === RULE-BASED: Nhặt sun ngay lập tức ===
                if suns and now - self.last_sun_collect > 0.1:
                    sun = suns[0]
                    if self.controller.collect_sun(sun["x"], sun["y"]):
                        self.last_sun_collect = now
                        print(f"[SUN] Collected at ({sun['x']}, {sun['y']})")
                
                # === AI: Quyết định plant ===
                can_plant = len(seed_yes) > 0
                has_zombie = len(zombies) > 0
                
                # Plant nếu có thể (không cần đợi zombie)
                if can_plant and now - self.last_plant_time > 1.5:
                    # Hỏi AI
                    if self.ai.should_plant(has_zombie, can_plant):
                        seed_pos = (seed_yes[0]["x"], seed_yes[0]["y"])
                        # Nếu có zombie thì plant ở row zombie, không thì plant ở giữa
                        plant_y = zombies[0]["y"] if zombies else 295
                        self.controller.plant_at_zombie_row(seed_pos, plant_y, plants)
                        self.last_plant_time = now
                
                # Draw
                for sun in suns:
                    cv2.circle(frame, (sun["x"], sun["y"]), 15, (0, 255, 255), 2)
                    cv2.putText(frame, "SUN", (sun["x"]-15, sun["y"]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                for z in zombies:
                    cv2.circle(frame, (z["x"], z["y"]), 15, (0, 0, 255), 2)
                    cv2.putText(frame, "ZOMBIE", (z["x"]-25, z["y"]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                for p in plants:
                    cv2.circle(frame, (p["x"], p["y"]), 10, (0, 255, 0), 2)
                for s in seed_yes:
                    cv2.rectangle(frame, (s["x"]-20, s["y"]-20), (s["x"]+20, s["y"]+20), (0, 255, 0), 2)
                    cv2.putText(frame, "READY", (s["x"]-20, s["y"]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                
                # FPS
                fps_counter += 1
                if time.time() - fps_time > 1:
                    fps = fps_counter
                    fps_counter = 0
                    fps_time = time.time()
                
                status = f"FPS:{fps} | Suns:{len(suns)} | Zombies:{len(zombies)} | Plants:{len(plants)} | CanPlant:{can_plant}"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("PvZ Auto", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Giữ 30fps
                elapsed = time.time() - loop_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                    
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    bot = PvZAutoPlay()
    bot.run()

"""
Script thu thập training data cho FunctionGemma từ YOLO detection.

Logic đơn giản:
- Có sun trên màn hình → nhặt sun (click vào vị trí sun)
- Có zombie + pea_shooter_pack sáng → trồng pea_shooter ở row có zombie
- Không có gì → do_nothing
"""

import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time
import json
import os
from datetime import datetime
from openvino.runtime import Core

class PvZDataCollector:
    def __init__(self, model_path="pvz_openvino/best.xml", output_file="training_data.json"):
        self.model_path = model_path
        self.output_file = output_file
        self.window = None
        self.model = None
        self.data = []
        
        # Load existing data nếu có
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✓ Đã load {len(self.data)} samples từ {output_file}")
        
        # Class names
        self.class_names = {
            0: "pea_shooter",
            1: "pea_shooter_pack_no",
            2: "pea_shooter_pack_yes", 
            3: "sun",
            4: "zombie"
        }
        
        self.input_size = (640, 640)
        
        # PvZ 800x600 - Grid 5 rows
        self.row_boundaries = [109, 189, 269, 349, 429, 509]  # y pixels cho row 1-5
    
    def get_row_from_y(self, y):
        """Tính row (1-5) từ pixel y"""
        for i in range(5):
            if y < self.row_boundaries[i + 1]:
                return i + 1
        return 5
    
    def load_model(self):
        try:
            print(f"Đang load model từ: {self.model_path}")
            ie = Core()
            self.model = ie.compile_model(model=self.model_path, device_name="CPU")
            self.input_layer = self.model.input(0)
            self.output_layer = self.model.output(0)
            print("✓ Đã load model!")
            return True
        except Exception as e:
            print(f"✗ Lỗi: {e}")
            return False
    
    def find_pvz_window(self):
        possible_names = ["Plants vs. Zombies", "Plants vs Zombies", "PlantsVsZombies", "popcapgame1"]
        all_windows = gw.getAllTitles()
        
        for window_title in all_windows:
            for name in possible_names:
                if name.lower() in window_title.lower():
                    self.window = gw.getWindowsWithTitle(window_title)[0]
                    print(f"✓ Tìm thấy: {window_title}")
                    return True
        
        print("✗ Không tìm thấy cửa sổ PvZ!")
        return False
    
    def capture_window(self):
        try:
            left, top = self.window.left, self.window.top
            width, height = self.window.width, self.window.height
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except:
            return None
    
    def preprocess(self, image):
        input_img = cv2.resize(image, self.input_size)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)
        return input_img
    
    def postprocess(self, output, orig_shape, conf_threshold=0.3):
        predictions = output[0].T
        detections = []
        
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        
        for pred in predictions:
            x, y, w, h = pred[:4]
            class_confs = pred[4:]
            class_id = np.argmax(class_confs)
            confidence = class_confs[class_id]
            
            if confidence > conf_threshold:
                cx = int(x * scale_x)
                cy = int(y * scale_y)
                
                detections.append({
                    "class_id": int(class_id),
                    "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                    "confidence": float(confidence),
                    "x": cx,
                    "y": cy
                })
        
        # NMS đơn giản
        if len(detections) > 1:
            boxes = [[d["x"]-20, d["y"]-20, d["x"]+20, d["y"]+20] for d in detections]
            scores = [d["confidence"] for d in detections]
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)
            if len(indices) > 0:
                indices = indices.flatten()
                detections = [detections[i] for i in indices]
        
        return detections
    
    def generate_game_state(self, detections):
        """Tạo game state đơn giản từ detections"""
        
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
        
        # Tạo description ngắn gọn
        parts = []
        
        # Sun
        if suns:
            parts.append(f"Sun at ({suns[0]['x']}, {suns[0]['y']})")
        else:
            parts.append("No sun")
        
        # Zombies
        if zombies:
            zombie_rows = [z['row'] for z in zombies]
            parts.append(f"Zombie row {zombie_rows}")
        else:
            parts.append("No zombie")
        
        # Pea shooters đã trồng
        if pea_shooters:
            pea_rows = list(set([p['row'] for p in pea_shooters]))
            parts.append(f"Pea at row {pea_rows}")
        
        # Có thể trồng không
        parts.append(f"Can plant: {can_plant}")
        
        return ". ".join(parts), suns, zombies, can_plant
    
    def save_data(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"✓ Đã lưu {len(self.data)} samples vào {self.output_file}")
    
    def start_collecting(self, interval=2.0):
        print("=" * 60)
        print("THU THẬP TRAINING DATA")
        print("=" * 60)
        
        if not self.load_model():
            return
        
        if not self.find_pvz_window():
            return
        
        print(f"\nThu thập mỗi {interval} giây")
        print("'c' - Capture ngay")
        print("'s' - Skip")
        print("'q' - Dừng và lưu\n")
        
        last_capture = 0
        
        try:
            while True:
                frame = self.capture_window()
                if frame is None:
                    continue
                
                # Detect
                input_tensor = self.preprocess(frame)
                output = self.model([input_tensor])[self.output_layer]
                detections = self.postprocess(output, frame.shape)
                
                # Vẽ detections
                for det in detections:
                    x, y = det["x"], det["y"]
                    color = (0, 0, 255) if det["class_name"] == "zombie" else (0, 255, 0)
                    if det["class_name"] == "sun":
                        color = (0, 255, 255)
                    cv2.circle(frame, (x, y), 20, color, 2)
                    cv2.putText(frame, det["class_name"], (x-30, y-25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Info
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
                        "action": "",      # ĐIỀN: collect_sun / plant_pea_shooter / do_nothing
                        "arguments": {}    # ĐIỀN: {"x": ..., "y": ...} hoặc {"row": ...}
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
            print("HƯỚNG DẪN ĐIỀN DATA")
            print("=" * 60)
            print(f"""
Mở file '{self.output_file}' và điền:

1. Có sun → action: "collect_sun", arguments: {{"x": ..., "y": ...}}
2. Có zombie + can_plant=true → action: "plant_pea_shooter", arguments: {{"row": ...}}
3. Không có gì làm → action: "do_nothing", arguments: {{}}
""")

if __name__ == "__main__":
    collector = PvZDataCollector(
        model_path="pvz_openvino/best.xml",
        output_file="training_data.json"
    )
    collector.start_collecting(interval=2.0)

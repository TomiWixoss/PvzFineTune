import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time
from openvino.runtime import Core

class PvZRealtimeDetection:
    def __init__(self, model_path="pvz_openvino/best.xml"):
        """
        Khởi tạo tool nhận diện realtime cho PvZ với OpenVINO
        
        Args:
            model_path: Đường dẫn đến model OpenVINO (.xml)
        """
        self.model_path = model_path
        self.window = None
        self.model = None
        self.input_layer = None
        self.output_layer = None
        
        # Class names từ metadata
        self.class_names = {
            0: "pea_shooter",
            1: "pea_shooter_pack_no",
            2: "pea_shooter_pack_yes",
            3: "sun",
            4: "zombie"
        }
        
        # Màu sắc cho từng class (BGR format)
        self.colors = {
            0: (0, 255, 0),      # pea_shooter - Xanh lá
            1: (0, 165, 255),    # pea_shooter_pack_no - Cam
            2: (0, 255, 255),    # pea_shooter_pack_yes - Vàng
            3: (255, 255, 0),    # sun - Cyan
            4: (0, 0, 255)       # zombie - Đỏ
        }
        
        self.input_size = (640, 640)
    
    def load_model(self):
        """Load model OpenVINO"""
        try:
            print(f"Đang load model từ: {self.model_path}")
            
            # Khởi tạo OpenVINO Core
            ie = Core()
            
            # Load model
            self.model = ie.compile_model(model=self.model_path, device_name="CPU")
            
            # Lấy input và output layer
            self.input_layer = self.model.input(0)
            self.output_layer = self.model.output(0)
            
            print("✓ Đã load model thành công!")
            print(f"  Input shape: {self.input_layer.shape}")
            print(f"  Output shape: {self.output_layer.shape}")
            return True
        except Exception as e:
            print(f"✗ Lỗi khi load model: {e}")
            return False
    
    def find_pvz_window(self):
        """Tìm cửa sổ game PvZ"""
        possible_names = [
            "Plants vs. Zombies",
            "Plants vs Zombies",
            "PlantsVsZombies",
            "popcapgame1"
        ]
        
        all_windows = gw.getAllTitles()
        
        for window_title in all_windows:
            for name in possible_names:
                if name.lower() in window_title.lower():
                    self.window = gw.getWindowsWithTitle(window_title)[0]
                    print(f"✓ Đã tìm thấy cửa sổ: {window_title}")
                    return True
        
        print("✗ Không tìm thấy cửa sổ PvZ!")
        print("Các cửa sổ đang mở:")
        for title in all_windows:
            if title.strip():
                print(f"  - {title}")
        return False
    
    def capture_window(self):
        """Chụp màn hình cửa sổ game"""
        try:
            left, top = self.window.left, self.window.top
            width, height = self.window.width, self.window.height
            
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
        except Exception as e:
            print(f"Lỗi khi chụp màn hình: {e}")
            return None
    
    def preprocess(self, image):
        """Tiền xử lý ảnh cho model"""
        # Resize về kích thước input
        input_img = cv2.resize(image, self.input_size)
        
        # Chuyển từ BGR sang RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Normalize về [0, 1] và chuyển sang NCHW format
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))  # HWC -> CHW
        input_img = np.expand_dims(input_img, axis=0)   # Add batch dimension
        
        return input_img
    
    def postprocess(self, output, orig_shape, conf_threshold=0.25, iou_threshold=0.45):
        """Xử lý output từ model"""
        # Output shape: [1, 84, 8400] cho YOLO11
        # 84 = 4 (bbox) + 80 (classes) nhưng model này có 5 classes
        # Nên sẽ là [1, 9, 8400] = 4 (bbox) + 5 (classes)
        
        predictions = output[0]  # [9, 8400] hoặc [84, 8400]
        
        # Transpose để dễ xử lý: [8400, 9]
        predictions = predictions.T
        
        boxes = []
        scores = []
        class_ids = []
        
        # Scale factors
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        
        for pred in predictions:
            # pred = [x, y, w, h, class0_conf, class1_conf, ...]
            x, y, w, h = pred[:4]
            class_confs = pred[4:]
            
            # Lấy class có confidence cao nhất
            class_id = np.argmax(class_confs)
            confidence = class_confs[class_id]
            
            if confidence > conf_threshold:
                # Chuyển từ center format sang corner format
                x1 = int((x - w / 2) * scale_x)
                y1 = int((y - h / 2) * scale_y)
                x2 = int((x + w / 2) * scale_x)
                y2 = int((y + h / 2) * scale_y)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))
                class_ids.append(int(class_id))
        
        # NMS (Non-Maximum Suppression)
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                scores = [scores[i] for i in indices]
                class_ids = [class_ids[i] for i in indices]
        
        return boxes, scores, class_ids
    
    def draw_detections(self, frame, boxes, scores, class_ids):
        """Vẽ bounding box và label lên frame"""
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            
            # Lấy tên class và màu
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            color = self.colors.get(class_id, (255, 255, 255))
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Tạo label
            label = f"{class_name}: {score:.2f}"
            
            # Vẽ background cho text
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Vẽ text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return frame
    
    def start_detection(self):
        """Bắt đầu nhận diện realtime"""
        print("=" * 60)
        print("PVZ REALTIME DETECTION - YOLO11 + OpenVINO")
        print("=" * 60)
        
        # Load model
        if not self.load_model():
            return
        
        # Tìm cửa sổ game
        if not self.find_pvz_window():
            print("\nVui lòng:")
            print("1. Mở game Plants vs Zombies")
            print("2. Chạy lại script này")
            return
        
        print("\n✓ Bắt đầu nhận diện realtime!")
        print("Nhấn 'q' trên cửa sổ hiển thị để thoát\n")
        
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        
        try:
            while True:
                # Chụp màn hình
                frame = self.capture_window()
                
                if frame is None:
                    continue
                
                # Tiền xử lý
                input_tensor = self.preprocess(frame)
                
                # Chạy inference
                output = self.model([input_tensor])[self.output_layer]
                
                # Hậu xử lý
                boxes, scores, class_ids = self.postprocess(output, frame.shape)
                
                # Vẽ kết quả lên frame
                frame = self.draw_detections(frame, boxes, scores, class_ids)
                
                # Tính FPS
                fps_counter += 1
                if time.time() - fps_time > 1:
                    fps = fps_counter
                    fps_counter = 0
                    fps_time = time.time()
                
                # Hiển thị FPS và số object
                cv2.putText(
                    frame,
                    f"FPS: {fps} | Objects: {len(boxes)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Hiển thị frame
                cv2.imshow("PvZ Detection", frame)
                
                # Nhấn 'q' để thoát
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n\nĐã dừng!")
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Đường dẫn đến model OpenVINO
    MODEL_PATH = "pvz_openvino/best.xml"
    
    detector = PvZRealtimeDetection(model_path=MODEL_PATH)
    detector.start_detection()

import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time
import os
from datetime import datetime

class PvZScreenCapture:
    def __init__(self, output_dir="pvz_screenshots", interval=2):
        """
        Khởi tạo tool chụp màn hình PvZ
        
        Args:
            output_dir: Thư mục lưu ảnh
            interval: Khoảng thời gian giữa các lần chụp (giây)
        """
        self.output_dir = output_dir
        self.interval = interval
        self.window = None
        self.capture_count = 0
        
        # Tạo thư mục nếu chưa có
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Đã tạo thư mục: {output_dir}")
    
    def find_pvz_window(self):
        """Tìm cửa sổ game PvZ"""
        # Các tên cửa sổ có thể có của PvZ
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
                    print(f"Đã tìm thấy cửa sổ: {window_title}")
                    return True
        
        print("Không tìm thấy cửa sổ PvZ!")
        print("Các cửa sổ đang mở:")
        for title in all_windows:
            if title.strip():
                print(f"  - {title}")
        return False
    
    def capture_window(self):
        """Chụp màn hình cửa sổ game"""
        try:
            # Lấy vị trí và kích thước cửa sổ
            left, top = self.window.left, self.window.top
            width, height = self.window.width, self.window.height
            
            # Chụp màn hình vùng cửa sổ
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            
            # Chuyển đổi sang numpy array (OpenCV format)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
        except Exception as e:
            print(f"Lỗi khi chụp màn hình: {e}")
            return None
    
    def save_screenshot(self, frame):
        """Lưu ảnh chụp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pvz_frame_{self.capture_count:04d}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(filepath, frame)
        self.capture_count += 1
        print(f"Đã lưu: {filename} (Tổng: {self.capture_count} ảnh)")
    
    def start_capture(self):
        """Bắt đầu chụp màn hình tự động"""
        print("=" * 50)
        print("TOOL CHỤP MÀN HÌNH PVZ CHO YOLOV11")
        print("=" * 50)
        
        # Tìm cửa sổ game
        if not self.find_pvz_window():
            print("\nVui lòng:")
            print("1. Mở game Plants vs Zombies")
            print("2. Chạy lại script này")
            return
        
        print(f"\nBắt đầu chụp màn hình mỗi {self.interval} giây...")
        print("Nhấn Ctrl+C để dừng\n")
        
        try:
            while True:
                # Kiểm tra cửa sổ vẫn còn tồn tại
                if not self.window.isActive and not self.window.isMinimized:
                    try:
                        self.window.activate()
                    except:
                        pass
                
                # Chụp màn hình
                frame = self.capture_window()
                
                if frame is not None:
                    self.save_screenshot(frame)
                
                # Đợi trước khi chụp tiếp
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print(f"\n\nĐã dừng! Tổng cộng đã chụp {self.capture_count} ảnh")
            print(f"Ảnh được lưu tại: {os.path.abspath(self.output_dir)}")

if __name__ == "__main__":
    # Cấu hình
    OUTPUT_DIR = "pvz_dataset"  # Thư mục lưu ảnh
    CAPTURE_INTERVAL = 0.5  # Chụp mỗi 0.5 giây
    
    # Khởi tạo và chạy
    capturer = PvZScreenCapture(output_dir=OUTPUT_DIR, interval=CAPTURE_INTERVAL)
    capturer.start_capture()

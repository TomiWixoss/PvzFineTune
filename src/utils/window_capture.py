# -*- coding: utf-8 -*-
"""
Utility for capturing PvZ game window - FAST version using mss/dxcam
"""

import cv2
import numpy as np
import pygetwindow as gw
import pyautogui

# Try fast capture methods
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import dxcam
    HAS_DXCAM = True
except ImportError:
    HAS_DXCAM = False


class PvZWindowCapture:
    """Utility class for finding and capturing PvZ game window"""
    
    POSSIBLE_WINDOW_NAMES = [
        "Plants vs. Zombies",
        "Plants vs Zombies",
        "PlantsVsZombies",
        "popcapgame1"
    ]
    
    def __init__(self, method: str = "auto"):
        """
        method: "auto", "dxcam", "mss", "pyautogui"
        - dxcam: Fastest (GPU-based, Windows only, ~120+ FPS)
        - mss: Fast (CPU-based, cross-platform, ~60 FPS)
        - pyautogui: Slow (fallback, ~15 FPS)
        """
        self.window = None
        self.offset_x = 0
        self.offset_y = 0
        self.method = method
        self._camera = None
        self._mss = None
        self._region = None
    
    def find_window(self) -> bool:
        """Find PvZ game window"""
        all_windows = gw.getAllTitles()
        
        for window_title in all_windows:
            for name in self.POSSIBLE_WINDOW_NAMES:
                if name.lower() in window_title.lower():
                    self.window = gw.getWindowsWithTitle(window_title)[0]
                    self.offset_x = self.window.left
                    self.offset_y = self.window.top
                    self._region = {
                        "left": self.window.left,
                        "top": self.window.top,
                        "width": self.window.width,
                        "height": self.window.height
                    }
                    print(f"✓ Found window: {window_title}")
                    self._init_capture()
                    return True
        
        print("✗ PvZ window not found!")
        print("Available windows:")
        for title in all_windows:
            if title.strip():
                print(f"  - {title}")
        return False
    
    def _init_capture(self):
        """Initialize capture method"""
        if self.method == "auto":
            if HAS_DXCAM:
                self.method = "dxcam"
            elif HAS_MSS:
                self.method = "mss"
            else:
                self.method = "pyautogui"
        
        if self.method == "dxcam" and HAS_DXCAM:
            try:
                self._camera = dxcam.create(output_color="BGR")
                print(f"✓ Using dxcam (fastest)")
            except Exception as e:
                print(f"⚠ dxcam failed: {e}, falling back to mss")
                self.method = "mss" if HAS_MSS else "pyautogui"
        
        if self.method == "mss" and HAS_MSS:
            self._mss = mss.mss()
            print(f"✓ Using mss (fast)")
        
        if self.method == "pyautogui":
            print(f"⚠ Using pyautogui (slow). Install mss or dxcam for better FPS:")
            print(f"  pip install mss")
            print(f"  pip install dxcam")
    
    def capture(self) -> np.ndarray:
        """Capture game window screenshot"""
        try:
            # Update region in case window moved
            self._region = {
                "left": self.window.left,
                "top": self.window.top,
                "width": self.window.width,
                "height": self.window.height
            }
            self.offset_x = self.window.left
            self.offset_y = self.window.top
            
            if self.method == "dxcam" and self._camera:
                frame = self._camera.grab(region=(
                    self._region["left"],
                    self._region["top"],
                    self._region["left"] + self._region["width"],
                    self._region["top"] + self._region["height"]
                ))
                if frame is None:
                    return self._capture_fallback()
                return frame
            
            elif self.method == "mss" and self._mss:
                screenshot = self._mss.grab(self._region)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                return frame
            
            else:
                return self._capture_fallback()
                
        except Exception as e:
            print(f"Capture error: {e}")
            return self._capture_fallback()
    
    def _capture_fallback(self) -> np.ndarray:
        """Fallback to pyautogui"""
        try:
            screenshot = pyautogui.screenshot(region=(
                self._region["left"],
                self._region["top"],
                self._region["width"],
                self._region["height"]
            ))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except:
            return None
    
    def click(self, x: int, y: int):
        """Click at relative position in game window"""
        pyautogui.click(self.offset_x + x, self.offset_y + y)
    
    def __del__(self):
        """Cleanup"""
        if self._camera:
            del self._camera
        if self._mss:
            self._mss.close()

# -*- coding: utf-8 -*-
"""
Window Capture - Capture game window với multiple backends
"""

import cv2
import numpy as np
import pygetwindow as gw
import pyautogui

from ..core.constants import PVZ_WINDOW_NAMES

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


class WindowCapture:
    """Capture game window với auto-select backend"""
    
    def __init__(self, method: str = "auto"):
        """
        Args:
            method: "auto", "dxcam", "mss", "pyautogui"
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
            for name in PVZ_WINDOW_NAMES:
                if name.lower() in window_title.lower():
                    self.window = gw.getWindowsWithTitle(window_title)[0]
                    self._update_region()
                    print(f"✓ Found window: {window_title}")
                    self._init_capture()
                    return True
        
        print("✗ PvZ window not found!")
        return False
    
    def _update_region(self):
        """Update capture region from window position"""
        self.offset_x = self.window.left
        self.offset_y = self.window.top
        self._region = {
            "left": self.window.left,
            "top": self.window.top,
            "width": self.window.width,
            "height": self.window.height
        }
    
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
                print("✓ Using dxcam (fastest)")
                return
            except Exception as e:
                print(f"⚠ dxcam failed: {e}")
                self.method = "mss" if HAS_MSS else "pyautogui"
        
        if self.method == "mss" and HAS_MSS:
            self._mss = mss.mss()
            print("✓ Using mss (fast)")
            return
        
        print("⚠ Using pyautogui (slow)")
    
    def capture(self) -> np.ndarray:
        """Capture game window screenshot"""
        try:
            self._update_region()
            
            if self.method == "dxcam" and self._camera:
                frame = self._camera.grab(region=(
                    self._region["left"],
                    self._region["top"],
                    self._region["left"] + self._region["width"],
                    self._region["top"] + self._region["height"]
                ))
                if frame is not None:
                    return frame
            
            elif self.method == "mss" and self._mss:
                screenshot = self._mss.grab(self._region)
                frame = np.array(screenshot)
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            return self._capture_fallback()
        except Exception:
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
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except:
            return None
    
    def click(self, x: int, y: int):
        """Click at relative position in game window"""
        pyautogui.click(self.offset_x + x, self.offset_y + y)
    
    def __del__(self):
        if self._camera:
            del self._camera
        if self._mss:
            self._mss.close()

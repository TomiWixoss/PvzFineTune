# -*- coding: utf-8 -*-
"""
Shared utilities for PvZ window capture
"""

import cv2
import numpy as np
import pygetwindow as gw
import pyautogui


class PvZWindowCapture:
    """Utility class for finding and capturing PvZ game window"""
    
    POSSIBLE_WINDOW_NAMES = [
        "Plants vs. Zombies",
        "Plants vs Zombies",
        "PlantsVsZombies",
        "popcapgame1"
    ]
    
    def __init__(self):
        self.window = None
        self.offset_x = 0
        self.offset_y = 0
    
    def find_window(self) -> bool:
        """Find PvZ game window"""
        all_windows = gw.getAllTitles()
        
        for window_title in all_windows:
            for name in self.POSSIBLE_WINDOW_NAMES:
                if name.lower() in window_title.lower():
                    self.window = gw.getWindowsWithTitle(window_title)[0]
                    self.offset_x = self.window.left
                    self.offset_y = self.window.top
                    print(f"✓ Found window: {window_title}")
                    return True
        
        print("✗ PvZ window not found!")
        print("Available windows:")
        for title in all_windows:
            if title.strip():
                print(f"  - {title}")
        return False
    
    def capture(self) -> np.ndarray:
        """Capture game window screenshot"""
        try:
            left, top = self.window.left, self.window.top
            width, height = self.window.width, self.window.height
            
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            print(f"Capture error: {e}")
            return None
    
    def click(self, x: int, y: int):
        """Click at relative position in game window"""
        pyautogui.click(self.offset_x + x, self.offset_y + y)
    
    @property
    def is_active(self) -> bool:
        """Check if window is still active"""
        return self.window is not None and not self.window.isMinimized

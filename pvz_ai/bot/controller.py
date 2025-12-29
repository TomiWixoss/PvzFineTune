# -*- coding: utf-8 -*-
"""
Game Controller - Điều khiển click trong game
"""

import time
from typing import Tuple

from ..core.config import Config
from ..utils.window_capture import WindowCapture


class GameController:
    """Game controller for clicking actions"""
    
    def __init__(self, window: WindowCapture):
        self.window = window
    
    def cancel_selection(self):
        """Cancel seed selection bằng right click"""
        import pyautogui
        pyautogui.rightClick()
    
    def collect_sun(self, x: int, y: int):
        """Collect sun at pixel position"""
        # Luôn cancel trước để đảm bảo không đang chọn seed
        self.cancel_selection()
        self.window.click(x, y)
    
    def plant_at_grid(
        self, 
        seed_pos: Tuple[int, int], 
        row: int, 
        col: int, 
        plant_type: str = "pea_shooter"
    ) -> bool:
        """Plant at grid position (row, col)"""
        grid_rows = Config.GRID_ROWS_Y
        grid_cols = Config.GRID_COLUMNS_X
        
        if row < 0 or row >= len(grid_rows) or col < 0 or col >= len(grid_cols):
            print(f"[AI] Invalid grid position: row={row}, col={col}")
            return False
        
        # Cancel trước để đảm bảo không đang chọn seed khác
        self.cancel_selection()
        
        # Click seed packet
        self.window.click(seed_pos[0], seed_pos[1])
        time.sleep(0.15)
        
        # Click grid position
        x = grid_cols[col]
        y = grid_rows[row]
        self.window.click(x, y)
        print(f"[AI] Plant {plant_type} at row={row}, col={col}")
        return True

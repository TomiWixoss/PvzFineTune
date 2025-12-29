# -*- coding: utf-8 -*-
"""
Config - Cấu hình có thể thay đổi
"""

from pathlib import Path
from .constants import (
    GRID_ROWS_Y_800, GRID_COLUMNS_X_800,
    GRID_ROWS_Y_1080, GRID_COLUMNS_X_1080,
    DEFAULT_CONFIDENCE, DEFAULT_IOU_THRESHOLD
)


class Config:
    """Configuration class"""
    
    # Paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    MODELS_DIR = ROOT_DIR / "models"
    DATA_DIR = ROOT_DIR / "data"
    
    # Model paths
    YOLO_MODEL_PATH = MODELS_DIR / "yolo" / "best.xml"
    GEMMA_MODEL_PATH = MODELS_DIR / "gemma"
    
    # Data paths
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    AI_LABELER_DIR = DATA_DIR / "ai_labeler"
    
    # Detection
    CONFIDENCE = DEFAULT_CONFIDENCE
    IOU_THRESHOLD = DEFAULT_IOU_THRESHOLD
    
    # Grid (default 800x600)
    GRID_ROWS_Y = GRID_ROWS_Y_800
    GRID_COLUMNS_X = GRID_COLUMNS_X_800
    
    @classmethod
    def use_1080p_grid(cls):
        """Switch to 1080p grid"""
        cls.GRID_ROWS_Y = GRID_ROWS_Y_1080
        cls.GRID_COLUMNS_X = GRID_COLUMNS_X_1080
    
    @classmethod
    def use_800_grid(cls):
        """Switch to 800x600 grid"""
        cls.GRID_ROWS_Y = GRID_ROWS_Y_800
        cls.GRID_COLUMNS_X = GRID_COLUMNS_X_800

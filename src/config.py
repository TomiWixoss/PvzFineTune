# -*- coding: utf-8 -*-
"""
Shared configuration for PvZ AI Bot
"""

# ===========================================
# MODEL PATHS
# ===========================================
YOLO_MODEL_PATH = "models/yolo/best.xml"
GEMMA_MODEL_PATH = "models/gemma/pvz_functiongemma_final"

# ===========================================
# YOLO CLASS NAMES (from trained model)
# ===========================================
CLASS_NAMES = {
    0: "pea_shooter",
    1: "pea_shooter_cooldown",
    2: "pea_shooter_ready",
    3: "sun",
    4: "sunflower_reward",
    5: "zombie",
}

# ===========================================
# CLASS COLORS (BGR format for OpenCV)
# ===========================================
CLASS_COLORS = {
    "pea_shooter": (0, 255, 0),           # Green
    "pea_shooter_cooldown": (128, 128, 128),  # Gray
    "pea_shooter_ready": (0, 255, 0),     # Green
    "sun": (0, 255, 255),                 # Yellow
    "sunflower_reward": (0, 200, 255),    # Orange
    "zombie": (0, 0, 255),                # Red
}

# ===========================================
# GAME GRID CONFIG (Level 1)
# ===========================================
GRID_ROW_Y = 355
GRID_COLUMNS_X = [75, 154, 229, 312, 393, 476, 557, 638, 732]

# ===========================================
# DETECTION CONFIG
# ===========================================
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU_THRESHOLD = 0.45
INPUT_SIZE = (640, 640)

# ===========================================
# BOT CONFIG
# ===========================================
SUN_COLLECT_COOLDOWN = 0.1  # seconds
PLANT_COOLDOWN = 1.5  # seconds
TARGET_FPS = 30

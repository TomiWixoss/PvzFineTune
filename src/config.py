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
# GAME GRID CONFIG (5 rows x 9 cols)
# ===========================================
GRID_ROWS_Y = [161, 258, 359, 447, 553]  # Row 0-4 (top to bottom)
GRID_COLUMNS_X = [76, 152, 229, 316, 398, 477, 558, 633, 709]  # Col 0-8 (left to right)

# ===========================================
# DETECTION CONFIG
# ===========================================
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU_THRESHOLD = 0.45
INPUT_SIZE = (640, 640)

# ===========================================
# BOT CONFIG
# ===========================================
AI_INFERENCE_DELAY = 0.3  # seconds - time for AI to process
SUN_FALL_DELAY = 0.5  # seconds - wait for sun to land before collecting
PLANT_COOLDOWN = 1.5  # seconds
TARGET_FPS = 30

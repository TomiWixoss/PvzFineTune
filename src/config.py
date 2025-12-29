# -*- coding: utf-8 -*-
"""
Shared configuration for PvZ AI Bot
"""

# ===========================================
# MODEL PATHS
# ===========================================
YOLO_MODEL_PATH = "models/yolo/best.xml"
GEMMA_MODEL_PATH = "models/gemma"

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
# For 800x600 resolution (windowed)
GRID_ROWS_Y_800 = [161, 258, 359, 447, 553]
GRID_COLUMNS_X_800 = [76, 152, 229, 316, 398, 477, 558, 633, 709]

# For 1920x1080 resolution (fullscreen/video)
# Calibrated from actual video: zombie y=547-716 should be row=2 (middle)
GRID_ROWS_Y_1080 = [380, 480, 580, 680, 780]  # Row 0-4 (top to bottom)
GRID_COLUMNS_X_1080 = [182, 365, 550, 758, 955, 1145, 1339, 1519, 1702]  # Col 0-8

# Default (use 800x600 for game window, 1080 for video)
GRID_ROWS_Y = GRID_ROWS_Y_800
GRID_COLUMNS_X = GRID_COLUMNS_X_800

# ===========================================
# DETECTION CONFIG
# ===========================================
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU_THRESHOLD = 0.45
INPUT_SIZE = (640, 640)

# ===========================================
# BOT CONFIG
# ===========================================
TARGET_FPS = 30

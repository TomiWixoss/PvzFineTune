# -*- coding: utf-8 -*-
"""
Constants - Các hằng số không thay đổi
"""

# ===========================================
# YOLO CLASS NAMES
# ===========================================
CLASS_NAMES = {
    0: "cherry_bomb_reward",
    1: "pea_shooter",
    2: "pea_shooter_cooldown",
    3: "pea_shooter_ready",
    4: "sun",
    5: "sunflower",
    6: "sunflower_cooldown",
    7: "sunflower_ready",
    8: "sunflower_reward",
    9: "zombie",
}

# ===========================================
# CLASS COLORS (BGR format for OpenCV)
# ===========================================
CLASS_COLORS = {
    "cherry_bomb_reward": (0, 0, 255),
    "pea_shooter": (0, 255, 0),
    "pea_shooter_cooldown": (128, 128, 128),
    "pea_shooter_ready": (0, 255, 0),
    "sun": (0, 255, 255),
    "sunflower": (0, 200, 255),
    "sunflower_cooldown": (128, 128, 128),
    "sunflower_ready": (0, 200, 255),
    "sunflower_reward": (0, 200, 255),
    "zombie": (0, 0, 255),
}

# ===========================================
# GAME GRID CONFIG (5 rows x 9 cols)
# ===========================================
GRID_ROWS = 5
GRID_COLS = 9

# For 800x600 resolution (windowed)
GRID_ROWS_Y_800 = [161, 258, 359, 447, 553]
GRID_COLUMNS_X_800 = [76, 152, 229, 316, 398, 477, 558, 633, 709]

# For 1920x1080 resolution (fullscreen/video)
GRID_ROWS_Y_1080 = [380, 480, 580, 680, 780]
GRID_COLUMNS_X_1080 = [182, 365, 550, 758, 955, 1145, 1339, 1519, 1702]

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
SUN_CLICK_DELAY = 0.03  # Delay giữa các click sun trong cùng frame

# ===========================================
# AI LABELER CONFIG
# ===========================================
GEMINI_MODEL_NAME = "gemini-3-flash-preview"
VIDEO_FPS = 24

# ===========================================
# VALID ACTIONS
# ===========================================
VALID_ACTIONS = ["plant", "wait"]

# ===========================================
# WINDOW NAMES
# ===========================================
PVZ_WINDOW_NAMES = [
    "Plants vs. Zombies",
    "Plants vs Zombies",
    "PlantsVsZombies",
    "popcapgame1"
]

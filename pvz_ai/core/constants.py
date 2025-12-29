# -*- coding: utf-8 -*-
"""
Constants - Các hằng số không thay đổi
"""

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

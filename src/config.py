# -*- coding: utf-8 -*-
"""
Shared configuration for PvZ AI Bot
"""

# ===========================================
# MODEL PATHS
# ===========================================
YOLO_MODEL_PATH = "models/yolo/pvz_openvino/best.xml"
GEMMA_MODEL_PATH = "models/gemma/pvz_functiongemma_final"

# ===========================================
# YOLO CLASS NAMES & COLORS (BGR)
# ===========================================
CLASS_NAMES = {
    0: "sun",
    1: "zombie",
    2: "pea_shooter",
    3: "pea_shooter_ready",
    4: "pea_shooter_cooldown",
    5: "sunflower",
    6: "sunflower_ready",
    7: "sunflower_cooldown",
    8: "wall_nut",
    9: "wall_nut_ready",
    10: "wall_nut_cooldown",
    11: "cherry_bomb_ready",
    12: "cherry_bomb_cooldown",
    13: "potato_mine",
    14: "potato_mine_ready",
    15: "potato_mine_cooldown",
    16: "sunflower_reward",
    17: "button_continue",
    18: "button_start",
}

CLASS_COLORS = {
    "sun": (0, 255, 255),              # Yellow
    "zombie": (0, 0, 255),             # Red
    "pea_shooter": (0, 255, 0),        # Green
    "pea_shooter_ready": (0, 255, 0),  # Green
    "pea_shooter_cooldown": (128, 128, 128),  # Gray
    "sunflower": (0, 200, 255),        # Orange
    "sunflower_ready": (0, 200, 255),
    "sunflower_cooldown": (128, 128, 128),
    "wall_nut": (0, 165, 255),         # Orange
    "wall_nut_ready": (0, 165, 255),
    "wall_nut_cooldown": (128, 128, 128),
    "cherry_bomb_ready": (0, 0, 200),  # Dark red
    "cherry_bomb_cooldown": (128, 128, 128),
    "potato_mine": (50, 100, 150),
    "potato_mine_ready": (50, 100, 150),
    "potato_mine_cooldown": (128, 128, 128),
    "sunflower_reward": (0, 200, 255),
    "button_continue": (255, 200, 0),  # Cyan
    "button_start": (255, 200, 0),
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

# -*- coding: utf-8 -*-
"""
Cân bằng lại training data - tăng samples cho do_nothing và plant_pea_shooter
"""

import json

# Tạo data mới cân bằng hơn
data = []

# ========== COLLECT_SUN (20 samples) ==========
sun_positions = [
    (200, 100), (300, 150), (400, 180), (500, 120), (600, 200),
    (150, 90), (250, 130), (350, 160), (450, 140), (550, 170),
    (180, 110), (280, 145), (380, 175), (480, 125), (580, 195),
    (220, 105), (320, 155), (420, 185), (520, 115), (620, 205),
]
for i, (x, y) in enumerate(sun_positions):
    zombie = "Zombie row [3]" if i % 3 == 0 else "No zombie"
    can_plant = "True" if i % 2 == 0 else "False"
    data.append({
        "game_state": f"Sun at ({x}, {y}). {zombie}. Can plant: {can_plant}",
        "action": "collect_sun",
        "arguments": {"x": x, "y": y}
    })

# ========== PLANT_PEA_SHOOTER (20 samples) ==========
for i in range(20):
    data.append({
        "game_state": "No sun. Zombie row [3]. Can plant: True",
        "action": "plant_pea_shooter",
        "arguments": {}
    })

# ========== DO_NOTHING (20 samples) ==========
# Case 1: Không sun, không zombie, không thể plant
for i in range(7):
    data.append({
        "game_state": "No sun. No zombie. Can plant: False",
        "action": "do_nothing",
        "arguments": {}
    })

# Case 2: Không sun, có zombie, không thể plant
for i in range(7):
    data.append({
        "game_state": "No sun. Zombie row [3]. Can plant: False",
        "action": "do_nothing",
        "arguments": {}
    })

# Case 3: Không sun, không zombie, có thể plant (chờ zombie)
for i in range(6):
    data.append({
        "game_state": "No sun. No zombie. Can plant: True",
        "action": "do_nothing",
        "arguments": {}
    })

# Thêm id
for i, sample in enumerate(data):
    sample["id"] = i + 1

# Lưu
with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Thống kê
from collections import Counter
actions = Counter(s["action"] for s in data)
print(f"✓ Đã tạo training_data.json cân bằng")
print(f"  Tổng: {len(data)} samples")
for action, count in actions.items():
    print(f"  - {action}: {count}")

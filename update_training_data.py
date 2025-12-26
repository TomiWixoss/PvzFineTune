# -*- coding: utf-8 -*-
"""
Training data với format rõ ràng hơn - dùng HAS_SUN/NO_SUN prefix
"""

import json

data = []

# ========== COLLECT_SUN (25 samples) ==========
sun_positions = [
    (200, 100), (300, 150), (400, 180), (500, 120), (600, 200),
    (150, 90), (250, 130), (350, 160), (450, 140), (550, 170),
    (180, 110), (280, 145), (380, 175), (480, 125), (580, 195),
    (220, 105), (320, 155), (420, 185), (520, 115), (620, 205),
    (190, 95), (290, 135), (390, 165), (490, 145), (590, 185),
]
for i, (x, y) in enumerate(sun_positions):
    zombie = "HAS_ZOMBIE" if i % 3 == 0 else "NO_ZOMBIE"
    can_plant = "CAN_PLANT" if i % 2 == 0 else "CANNOT_PLANT"
    data.append({
        "game_state": f"HAS_SUN x={x} y={y}. {zombie}. {can_plant}",
        "action": "collect_sun",
        "arguments": {"x": x, "y": y}
    })

# ========== PLANT_PEA_SHOOTER (25 samples) ==========
for i in range(25):
    data.append({
        "game_state": "NO_SUN. HAS_ZOMBIE. CAN_PLANT",
        "action": "plant_pea_shooter",
        "arguments": {}
    })

# ========== DO_NOTHING (25 samples) ==========
# Không sun, không thể plant
for i in range(10):
    data.append({
        "game_state": "NO_SUN. NO_ZOMBIE. CANNOT_PLANT",
        "action": "do_nothing",
        "arguments": {}
    })

for i in range(10):
    data.append({
        "game_state": "NO_SUN. HAS_ZOMBIE. CANNOT_PLANT",
        "action": "do_nothing",
        "arguments": {}
    })

# Không sun, không zombie, có thể plant -> chờ
for i in range(5):
    data.append({
        "game_state": "NO_SUN. NO_ZOMBIE. CAN_PLANT",
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
print(f"✓ Đã tạo training_data.json")
print(f"  Tổng: {len(data)} samples")
for action, count in actions.items():
    print(f"  - {action}: {count}")

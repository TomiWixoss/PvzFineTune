# -*- coding: utf-8 -*-
"""
Tạo 100 samples cân bằng cho PvZ bot
"""

import json
import random

data = []

# ========== COLLECT_SUN (34 samples) ==========
for i in range(34):
    x = random.randint(100, 700)
    y = random.randint(80, 250)
    zombie = random.choice(["HAS_ZOMBIE", "NO_ZOMBIE"])
    can_plant = random.choice(["CAN_PLANT", "CANNOT_PLANT"])
    data.append({
        "game_state": f"HAS_SUN x={x} y={y}. {zombie}. {can_plant}",
        "action": "collect_sun",
        "arguments": {"x": x, "y": y}
    })

# ========== PLANT_PEA_SHOOTER (33 samples) ==========
for i in range(33):
    data.append({
        "game_state": "NO_SUN. HAS_ZOMBIE. CAN_PLANT",
        "action": "plant_pea_shooter",
        "arguments": {}
    })

# ========== DO_NOTHING (33 samples) ==========
# Không sun, không thể plant
for i in range(11):
    data.append({
        "game_state": "NO_SUN. NO_ZOMBIE. CANNOT_PLANT",
        "action": "do_nothing",
        "arguments": {}
    })

for i in range(11):
    data.append({
        "game_state": "NO_SUN. HAS_ZOMBIE. CANNOT_PLANT",
        "action": "do_nothing",
        "arguments": {}
    })

# Không sun, không zombie, có thể plant -> chờ
for i in range(11):
    data.append({
        "game_state": "NO_SUN. NO_ZOMBIE. CAN_PLANT",
        "action": "do_nothing",
        "arguments": {}
    })

# Shuffle
random.shuffle(data)

# Lưu
with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Thống kê
from collections import Counter
actions = Counter(s["action"] for s in data)
print(f"✓ Tạo training_data.json: {len(data)} samples")
for action, count in actions.items():
    print(f"  - {action}: {count}")

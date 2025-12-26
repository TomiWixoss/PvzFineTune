# -*- coding: utf-8 -*-
"""
Tạo 100,000 samples training data
"""

import json
import random

random.seed(42)
data = []

# Tạo 100,000 samples cân bằng: ~33k mỗi action
for i in range(100000):
    action_type = i % 3  # 0, 1, 2 luân phiên
    
    if action_type == 0:
        # COLLECT_SUN - có sun
        x = random.randint(50, 700)
        y = random.randint(50, 250)
        zombie = random.choice(["HAS_ZOMBIE", "NO_ZOMBIE"])
        can_plant = random.choice(["CAN_PLANT", "CANNOT_PLANT"])
        data.append({
            "game_state": f"HAS_SUN x={x} y={y}. {zombie}. {can_plant}",
            "action": "collect_sun",
            "arguments": {"x": x, "y": y}
        })
    
    elif action_type == 1:
        # PLANT_PEA_SHOOTER - không sun, có zombie, có thể plant
        data.append({
            "game_state": "NO_SUN. HAS_ZOMBIE. CAN_PLANT",
            "action": "plant_pea_shooter",
            "arguments": {}
        })
    
    else:
        # DO_NOTHING - không sun, không thể làm gì
        zombie = random.choice(["HAS_ZOMBIE", "NO_ZOMBIE"])
        can_plant = random.choice(["CANNOT_PLANT", "CAN_PLANT"]) if zombie == "NO_ZOMBIE" else "CANNOT_PLANT"
        data.append({
            "game_state": f"NO_SUN. {zombie}. {can_plant}",
            "action": "do_nothing",
            "arguments": {}
        })

# Shuffle
random.shuffle(data)

# Thêm id
for i, sample in enumerate(data):
    sample["id"] = i + 1

# Lưu
with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

# Thống kê
from collections import Counter
actions = Counter(s["action"] for s in data)
print(f"✓ Đã tạo training_data.json")
print(f"  Tổng: {len(data):,} samples")
for action, count in actions.items():
    print(f"  - {action}: {count:,}")

# -*- coding: utf-8 -*-
"""
Script cập nhật training_data.json cho màn 1 PvZ
- Xóa argument row khỏi plant_pea_shooter (màn 1 chỉ có 1 dòng)
- Thêm nhiều samples đa dạng hơn
"""

import json

# Load data hiện tại
with open("training_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Cập nhật: xóa row argument khỏi plant_pea_shooter
for sample in data:
    if sample["action"] == "plant_pea_shooter":
        sample["arguments"] = {}  # Màn 1 không cần row

# Thêm samples mới đa dạng hơn
new_samples = [
    # Có sun, không zombie -> collect_sun
    {"game_state": "Sun at (200, 100). No zombie. Can plant: True", "action": "collect_sun", "arguments": {"x": 200, "y": 100}},
    {"game_state": "Sun at (350, 180). No zombie. Can plant: False", "action": "collect_sun", "arguments": {"x": 350, "y": 180}},
    {"game_state": "Sun at (500, 120). No zombie. Can plant: True", "action": "collect_sun", "arguments": {"x": 500, "y": 120}},
    {"game_state": "Sun at (150, 90). No zombie. Can plant: False", "action": "collect_sun", "arguments": {"x": 150, "y": 90}},
    {"game_state": "Sun at (600, 200). No zombie. Can plant: True", "action": "collect_sun", "arguments": {"x": 600, "y": 200}},
    
    # Có sun, có zombie -> collect_sun (ưu tiên nhặt sun)
    {"game_state": "Sun at (250, 130). Zombie row [3]. Can plant: True", "action": "collect_sun", "arguments": {"x": 250, "y": 130}},
    {"game_state": "Sun at (400, 160). Zombie row [3]. Can plant: False", "action": "collect_sun", "arguments": {"x": 400, "y": 160}},
    {"game_state": "Sun at (550, 140). Zombie row [3]. Can plant: True", "action": "collect_sun", "arguments": {"x": 550, "y": 140}},
    
    # Không sun, có zombie, có thể plant -> plant_pea_shooter
    {"game_state": "No sun. Zombie row [3]. Can plant: True", "action": "plant_pea_shooter", "arguments": {}},
    {"game_state": "No sun. Zombie row [3]. Can plant: True", "action": "plant_pea_shooter", "arguments": {}},
    {"game_state": "No sun. Zombie row [3]. Can plant: True", "action": "plant_pea_shooter", "arguments": {}},
    
    # Không sun, không thể plant -> do_nothing
    {"game_state": "No sun. No zombie. Can plant: False", "action": "do_nothing", "arguments": {}},
    {"game_state": "No sun. Zombie row [3]. Can plant: False", "action": "do_nothing", "arguments": {}},
    {"game_state": "No sun. No zombie. Can plant: False", "action": "do_nothing", "arguments": {}},
    {"game_state": "No sun. Zombie row [3]. Can plant: False", "action": "do_nothing", "arguments": {}},
    {"game_state": "No sun. No zombie. Can plant: False", "action": "do_nothing", "arguments": {}},
    
    # Không sun, không zombie, có thể plant -> do_nothing (chờ zombie)
    {"game_state": "No sun. No zombie. Can plant: True", "action": "do_nothing", "arguments": {}},
    {"game_state": "No sun. No zombie. Can plant: True", "action": "do_nothing", "arguments": {}},
    {"game_state": "No sun. No zombie. Can plant: True", "action": "do_nothing", "arguments": {}},
]

# Thêm id cho samples mới
max_id = max(s.get("id", 0) for s in data)
for i, sample in enumerate(new_samples):
    sample["id"] = max_id + i + 1
    sample["suns"] = []
    sample["zombies"] = []
    sample["can_plant"] = "Can plant: True" in sample["game_state"]

data.extend(new_samples)

# Lưu lại
with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✓ Đã cập nhật training_data.json")
print(f"  - Tổng samples: {len(data)}")
print(f"  - Đã xóa row argument khỏi plant_pea_shooter")
print(f"  - Đã thêm {len(new_samples)} samples mới")

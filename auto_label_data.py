"""
Tự động điền action và arguments cho training data
+ Cân bằng data bằng cách OVERSAMPLE (nhân bản class ít lên)
"""

import json
import random

# Đọc data gốc
with open("training_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Tự động điền
for sample in data:
    suns = sample.get("suns", [])
    zombies = sample.get("zombies", [])
    can_plant = sample.get("can_plant", False)
    
    if suns:
        sample["action"] = "collect_sun"
        sample["arguments"] = {"x": suns[0]["x"], "y": suns[0]["y"]}
    elif zombies and can_plant:
        sample["action"] = "plant_pea_shooter"
        sample["arguments"] = {"row": zombies[0]["row"]}
    else:
        sample["action"] = "do_nothing"
        sample["arguments"] = {}

# Phân loại theo action
by_action = {}
for sample in data:
    action = sample["action"]
    if action not in by_action:
        by_action[action] = []
    by_action[action].append(sample)

print("Trước khi cân bằng:")
for action, samples in by_action.items():
    print(f"  {action}: {len(samples)}")

# Tìm số lượng lớn nhất
max_count = max(len(s) for s in by_action.values())

# OVERSAMPLE: nhân bản class ít lên bằng class nhiều nhất
balanced_data = []
for action, samples in by_action.items():
    if len(samples) == 0:
        continue
    
    # Nhân bản lên đủ max_count
    oversampled = []
    while len(oversampled) < max_count:
        oversampled.extend(samples)
    
    # Cắt lấy đúng max_count
    oversampled = oversampled[:max_count]
    balanced_data.extend(oversampled)

# Shuffle
random.shuffle(balanced_data)

# Đánh lại ID
for i, sample in enumerate(balanced_data):
    sample["id"] = i + 1

# Lưu lại
with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(balanced_data, f, indent=2, ensure_ascii=False)

print(f"\nSau khi cân bằng (oversample): {len(balanced_data)} samples")

# Thống kê lại
actions = {}
for sample in balanced_data:
    action = sample["action"]
    actions[action] = actions.get(action, 0) + 1

print("\nThống kê:")
for action, count in actions.items():
    print(f"  {action}: {count}")

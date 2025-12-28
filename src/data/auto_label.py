# -*- coding: utf-8 -*-
"""
Auto-label training data and balance dataset using oversampling
"""

import json
import random
import os


def auto_label_and_balance(input_file="data/processed/training_data.json"):
    """Auto-fill action and arguments, then balance dataset"""
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Auto-fill labels
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

    # Group by action
    by_action = {}
    for sample in data:
        action = sample["action"]
        if action not in by_action:
            by_action[action] = []
        by_action[action].append(sample)

    print("Before balancing:")
    for action, samples in by_action.items():
        print(f"  {action}: {len(samples)}")

    # Find max count
    max_count = max(len(s) for s in by_action.values())

    # Oversample to balance
    balanced_data = []
    for action, samples in by_action.items():
        if len(samples) == 0:
            continue
        
        oversampled = []
        while len(oversampled) < max_count:
            oversampled.extend(samples)
        
        oversampled = oversampled[:max_count]
        balanced_data.extend(oversampled)

    # Shuffle and re-index
    random.shuffle(balanced_data)
    for i, sample in enumerate(balanced_data):
        sample["id"] = i + 1

    # Save
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(balanced_data, f, indent=2, ensure_ascii=False)

    print(f"\nAfter balancing (oversample): {len(balanced_data)} samples")

    # Stats
    actions = {}
    for sample in balanced_data:
        action = sample["action"]
        actions[action] = actions.get(action, 0) + 1

    print("\nStats:")
    for action, count in actions.items():
        print(f"  {action}: {count}")


def generate_synthetic_data(output_file="data/processed/training_data.json", count=100):
    """Generate synthetic balanced training data"""
    
    data = []
    samples_per_class = count // 3

    # COLLECT_SUN
    for i in range(samples_per_class + 1):
        x = random.randint(100, 700)
        y = random.randint(80, 250)
        zombie = random.choice(["HAS_ZOMBIE", "NO_ZOMBIE"])
        can_plant = random.choice(["CAN_PLANT", "CANNOT_PLANT"])
        data.append({
            "game_state": f"HAS_SUN x={x} y={y}. {zombie}. {can_plant}",
            "action": "collect_sun",
            "arguments": {"x": x, "y": y}
        })

    # PLANT_PEA_SHOOTER
    for i in range(samples_per_class):
        data.append({
            "game_state": "NO_SUN. HAS_ZOMBIE. CAN_PLANT",
            "action": "plant_pea_shooter",
            "arguments": {}
        })

    # DO_NOTHING
    do_nothing_per_type = samples_per_class // 3
    for i in range(do_nothing_per_type):
        data.append({
            "game_state": "NO_SUN. NO_ZOMBIE. CANNOT_PLANT",
            "action": "do_nothing",
            "arguments": {}
        })

    for i in range(do_nothing_per_type):
        data.append({
            "game_state": "NO_SUN. HAS_ZOMBIE. CANNOT_PLANT",
            "action": "do_nothing",
            "arguments": {}
        })

    for i in range(do_nothing_per_type + 1):
        data.append({
            "game_state": "NO_SUN. NO_ZOMBIE. CAN_PLANT",
            "action": "do_nothing",
            "arguments": {}
        })

    random.shuffle(data)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Stats
    from collections import Counter
    actions = Counter(s["action"] for s in data)
    print(f"✓ Generated {output_file}: {len(data)} samples")
    for action, count in actions.items():
        print(f"  - {action}: {count}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate_synthetic_data()
    else:
        auto_label_and_balance()

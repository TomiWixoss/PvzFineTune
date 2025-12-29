# -*- coding: utf-8 -*-
"""
Data Augmentor - Sinh thêm training data từ samples có sẵn
"""

import json
import random
import copy
from pathlib import Path
from typing import List, Dict


def get_valid_rows(game_state: str) -> set:
    """Lấy các row đang được sử dụng trong game_state (chỉ từ PLANTS)"""
    import re
    rows = set()
    # Chỉ lấy row từ PLANTS (không lấy từ zombies vì zombie có thể ở row khác)
    plants_match = re.search(r'PLANTS:\[([^\]]*)\]', game_state)
    if plants_match:
        for match in re.finditer(r'\(\w+,(\d),\d\)', plants_match.group(1)):
            rows.add(int(match.group(1)))
    return rows if rows else {2}  # Default row 2 nếu không có gì


def augment_game_state(game_state: str, valid_rows: set = None) -> List[str]:
    """
    Tạo variations của game_state bằng cách:
    - Thay đổi vị trí zombie (col) - giữ nguyên row trong valid_rows
    - Mirror rows CHỈ KHI có nhiều row
    """
    variations = [game_state]  # Giữ original
    
    if valid_rows is None:
        valid_rows = get_valid_rows(game_state)
    
    # Variation 1: Shift zombie columns (giữ nguyên row)
    for shift in [-1, 1, -2, 2]:
        new_state = shift_zombie_cols(game_state, shift)
        if new_state and new_state != game_state:
            # Kiểm tra zombie vẫn ở valid rows
            if is_zombies_in_valid_rows(new_state, valid_rows):
                variations.append(new_state)
    
    # Variation 2: Mirror rows CHỈ KHI có nhiều row
    if len(valid_rows) > 1:
        mirrored = mirror_rows(game_state)
        if mirrored and mirrored != game_state:
            variations.append(mirrored)
    
    return variations


def is_zombies_in_valid_rows(game_state: str, valid_rows: set) -> bool:
    """Kiểm tra tất cả zombie có ở trong valid_rows không"""
    import re
    zombies_match = re.search(r'ZOMBIES:\[([^\]]*)\]', game_state)
    if not zombies_match or not zombies_match.group(1):
        return True  # Không có zombie thì OK
    
    for match in re.finditer(r'\(\w+,(\d),\d\)', zombies_match.group(1)):
        if int(match.group(1)) not in valid_rows:
            return False
    return True


def fix_zombie_rows(game_state: str, valid_rows: set) -> str:
    """Sửa zombie về valid rows"""
    import re
    
    if not valid_rows:
        valid_rows = {2}
    
    valid_list = list(valid_rows)
    
    def fix_zombie(match):
        zombie_type = match.group(1)
        row = int(match.group(2))
        col = match.group(3)
        # Nếu row không hợp lệ, chọn row gần nhất trong valid_rows
        if row not in valid_rows:
            row = min(valid_list, key=lambda r: abs(r - row))
        return f"({zombie_type},{row},{col})"
    
    # Chỉ fix trong phần ZOMBIES
    parts = game_state.split("ZOMBIES:")
    if len(parts) != 2:
        return game_state
    
    before = parts[0] + "ZOMBIES:"
    after_parts = parts[1].split("].", 1)
    zombies_part = after_parts[0] + "]"
    rest = "." + after_parts[1] if len(after_parts) > 1 else ""
    
    pattern = r'\((\w+),(\d),(\d)\)'
    new_zombies = re.sub(pattern, fix_zombie, zombies_part)
    return before + new_zombies + rest


def shift_zombie_cols(game_state: str, shift: int) -> str:
    """Shift zombie columns by amount"""
    import re
    
    def shift_col(match):
        zombie_type = match.group(1)
        row = int(match.group(2))
        col = int(match.group(3))
        new_col = max(5, min(8, col + shift))  # Keep in range 5-8
        return f"({zombie_type},{row},{new_col})"
    
    # Match zombie tuples in ZOMBIES section
    pattern = r'\((\w+),(\d),(\d)\)'
    
    # Find ZOMBIES section
    if "ZOMBIES:[" not in game_state:
        return None
    
    parts = game_state.split("ZOMBIES:")
    if len(parts) != 2:
        return None
    
    before = parts[0] + "ZOMBIES:"
    after_parts = parts[1].split("].", 1)
    zombies_part = after_parts[0] + "]"
    rest = "." + after_parts[1] if len(after_parts) > 1 else ""
    
    new_zombies = re.sub(pattern, shift_col, zombies_part)
    return before + new_zombies + rest


def mirror_rows(game_state: str) -> str:
    """Mirror rows: 0↔4, 1↔3, 2 stays"""
    mirror_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    
    import re
    
    def mirror_tuple(match):
        item_type = match.group(1)
        row = int(match.group(2))
        col = match.group(3)
        new_row = mirror_map.get(row, row)
        return f"({item_type},{new_row},{col})"
    
    pattern = r'\((\w+),(\d),(\d)\)'
    return re.sub(pattern, mirror_tuple, game_state)


def add_random_zombie(game_state: str) -> str:
    """Add a random zombie to empty row"""
    import re
    
    # Find existing zombie rows
    zombie_rows = set()
    for match in re.finditer(r'ZOMBIES:\[([^\]]*)\]', game_state):
        for m in re.finditer(r'\(\w+,(\d),\d\)', match.group(1)):
            zombie_rows.add(int(m.group(1)))
    
    # Find empty rows
    empty_rows = [r for r in range(5) if r not in zombie_rows]
    if not empty_rows:
        return None
    
    new_row = random.choice(empty_rows)
    new_col = random.randint(6, 8)
    new_zombie = f"(zombie,{new_row},{new_col})"
    
    # Insert new zombie
    if "ZOMBIES:[]" in game_state:
        return game_state.replace("ZOMBIES:[]", f"ZOMBIES:[{new_zombie}]")
    else:
        return game_state.replace("ZOMBIES:[", f"ZOMBIES:[{new_zombie},")


def augment_sample(sample: Dict, mirror_action: bool = True) -> List[Dict]:
    """Augment một sample thành nhiều variations"""
    results = []
    
    game_state = sample.get("game_state", "")
    action = sample.get("action", "")
    args = sample.get("arguments", {})
    
    valid_rows = get_valid_rows(game_state)
    
    # Fix zombie rows trong data gốc trước
    game_state = fix_zombie_rows(game_state, valid_rows)
    sample = copy.deepcopy(sample)
    sample["game_state"] = game_state
    
    # Chỉ giữ original và shift zombie variations
    # KHÔNG mirror vì dễ tạo invalid data
    variations = [game_state]
    
    for shift in [-1, 1]:
        new_state = shift_zombie_cols(game_state, shift)
        if new_state and new_state != game_state:
            if is_zombies_in_valid_rows(new_state, valid_rows):
                variations.append(new_state)
    
    for var_state in variations:
        new_sample = copy.deepcopy(sample)
        new_sample["game_state"] = var_state
        results.append(new_sample)
    
    return results


def generate_synthetic_samples(base_data: List[Dict], target_count: int) -> List[Dict]:
    """
    Sinh thêm samples synthetic dựa trên pattern từ data gốc
    Chỉ cho level 1-1 (row 2 only)
    """
    synthetic = []
    
    # Lấy valid rows từ data gốc
    valid_rows = {2}  # Level 1-1 chỉ có row 2
    
    # Pattern 1: Các trạng thái plant progression (col 0 -> 1 -> 2 -> ...)
    for num_plants in range(9):  # 0 đến 8 plants
        plants_list = [f"(pea_shooter,2,{c})" for c in range(num_plants)]
        plants_str = ",".join(plants_list)
        
        for zombie_col in [5, 6, 7, 8]:  # Zombie ở các vị trí khác nhau
            for has_zombie in [True, False]:
                zombie_str = f"(zombie,2,{zombie_col})" if has_zombie else ""
                
                for seed_status in ["ready", "cooldown"]:
                    game_state = f"PLANTS:[{plants_str}]. ZOMBIES:[{zombie_str}]. SEEDS:[(pea_shooter,{seed_status})]"
                    
                    if seed_status == "ready" and num_plants < 9:
                        # Plant vào col tiếp theo
                        action = "plant"
                        arguments = {"plant_type": "pea_shooter", "row": 2, "col": num_plants}
                    else:
                        # Wait khi cooldown hoặc đã full
                        action = "wait"
                        arguments = {}
                    
                    synthetic.append({
                        "game_state": game_state,
                        "action": action,
                        "arguments": arguments
                    })
    
    # Deduplicate
    seen = set()
    unique = []
    for s in synthetic:
        key = (s["game_state"], s["action"], str(s.get("arguments", {})))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    
    # Shuffle và limit
    random.shuffle(unique)
    return unique[:target_count]


def augment_dataset(
    input_path: str,
    output_path: str = None,
    target_count: int = 50
) -> List[Dict]:
    """
    Augment toàn bộ dataset
    
    Args:
        input_path: Path to training_data.json
        output_path: Output path (default: input_augmented.json)
        target_count: Target number of samples
    
    Returns:
        Augmented dataset
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original: {len(data)} samples")
    
    # Augment từ data gốc
    augmented = []
    for sample in data:
        variations = augment_sample(sample)
        augmented.extend(variations)
    
    # Sinh thêm synthetic nếu chưa đủ
    if len(augmented) < target_count:
        synthetic = generate_synthetic_samples(data, target_count - len(augmented))
        augmented.extend(synthetic)
    
    # Deduplicate by game_state + action
    seen = set()
    unique = []
    for s in augmented:
        key = (s["game_state"], s["action"], str(s.get("arguments", {})))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    
    # Shuffle
    random.shuffle(unique)
    
    # Limit
    if len(unique) > target_count:
        unique = unique[:target_count]
    
    print(f"Augmented: {len(unique)} samples ({len(unique)/len(data):.1f}x)")
    
    # Save
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_augmented.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    
    print(f"Saved: {output_path}")
    return unique


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Augment training data')
    parser.add_argument('input', help='Input training_data.json')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('-n', '--count', type=int, default=100, help='Target sample count')
    args = parser.parse_args()
    
    augment_dataset(args.input, args.output, args.count)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Dataset Converter - Convert dataset sang format training cho Gemma
"""

import json
from pathlib import Path
from typing import List, Dict


def convert_dataset(input_path: str, output_path: str = None) -> List[Dict]:
    """
    Convert dataset từ video_dataset_builder sang format training
    
    Output format:
    [
        {
            "game_state": "PLANTS:[(pea_shooter,2,0)]. ZOMBIES:[]. SEEDS:[(pea_shooter,ready)]",
            "action": "plant",
            "arguments": {"plant_type": "pea_shooter", "row": 1, "col": 0}
        }
    ]
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"✗ File not found: {input_path}")
        return []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    samples = dataset.get("samples", [])
    print(f"✓ Loaded {len(samples)} samples")
    
    training_data = []
    stats = {"plant": 0, "wait": 0}
    
    for sample in samples:
        game_state = sample.get("game_state", {})
        action = sample.get("action", {})
        
        if not action:
            continue
        
        action_type = action.get("type", "wait")
        action_args = action.get("args", {})
        
        training_sample = {
            "game_state": game_state.get("text", ""),
            "action": action_type,
            "arguments": {}
        }
        
        if action_type == "plant" and action_args:
            training_sample["arguments"] = {
                "plant_type": action_args.get("plant_type", "pea_shooter"),
                "row": action_args.get("row", 2),
                "col": action_args.get("col", 0)
            }
        
        training_data.append(training_sample)
        stats[action_type] = stats.get(action_type, 0) + 1
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_training.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Converted {len(training_data)} samples")
    print(f"  Actions: {stats}")
    print(f"  Output: {output_path}")
    
    return training_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert dataset to training format')
    parser.add_argument('input', help='Input dataset.json')
    parser.add_argument('-o', '--output', help='Output training_data.json')
    args = parser.parse_args()
    
    convert_dataset(args.input, args.output)


if __name__ == "__main__":
    main()

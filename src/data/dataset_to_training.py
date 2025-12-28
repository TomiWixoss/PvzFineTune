# -*- coding: utf-8 -*-
"""
Convert video dataset (từ video_dataset_builder) sang format training cho Gemma

INPUT: dataset.json từ video_dataset_builder
OUTPUT: training_data.json cho Gemma finetune

USAGE:
    python -m src.data.dataset_to_training data/processed/dataset_final.json -o data/processed/training_data.json
"""

import json
import argparse
from pathlib import Path


def convert_dataset(input_path: str, output_path: str = None):
    """
    Convert dataset từ video_dataset_builder sang format training
    
    Input format (từ video_dataset_builder):
    {
        "samples": [
            {
                "game_state": {
                    "text": "HAS_SUN x=524 y=169. NO_ZOMBIE. CANNOT_PLANT",
                    "suns": [...],
                    ...
                },
                "action": {
                    "type": "collect_sun",
                    "args": {}
                }
            }
        ]
    }
    
    Output format (cho Gemma training):
    [
        {
            "game_state": "HAS_SUN x=524 y=169. NO_ZOMBIE. CANNOT_PLANT",
            "action": "collect_sun",
            "arguments": {"x": 524, "y": 169}
        }
    ]
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"✗ File not found: {input_path}")
        return
    
    # Load dataset
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    samples = dataset.get("samples", [])
    print(f"✓ Loaded {len(samples)} samples from {input_path.name}")
    
    # Convert
    training_data = []
    stats = {"collect_sun": 0, "plant_pea_shooter": 0, "do_nothing": 0}
    
    for sample in samples:
        game_state = sample.get("game_state", {})
        action = sample.get("action", {})
        
        if not action:
            continue
        
        action_type = action.get("type", "do_nothing")
        action_args = action.get("args", {})
        
        # Build training sample
        training_sample = {
            "game_state": game_state.get("text", ""),
            "action": action_type,
            "arguments": {}
        }
        
        # Add arguments based on action type
        if action_type == "collect_sun":
            # Lấy x, y từ game_state.suns nếu có
            suns = game_state.get("suns", [])
            if suns:
                training_sample["arguments"] = {
                    "x": suns[0].get("x", 0),
                    "y": suns[0].get("y", 0)
                }
            elif action_args:
                training_sample["arguments"] = action_args
        
        elif action_type == "plant_pea_shooter":
            # Lấy row, col từ action args
            if action_args:
                training_sample["arguments"] = {
                    "row": action_args.get("row", 2),
                    "col": action_args.get("col", 0)
                }
        
        training_data.append(training_sample)
        stats[action_type] = stats.get(action_type, 0) + 1
    
    # Output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_training.json"
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Converted {len(training_data)} samples")
    print(f"  Actions: {stats}")
    print(f"  Output: {output_path}")
    
    # Show sample
    if training_data:
        print(f"\n📋 Sample:")
        print(json.dumps(training_data[0], indent=2))
    
    return training_data


def main():
    parser = argparse.ArgumentParser(description='Convert dataset to Gemma training format')
    parser.add_argument('input', help='Input dataset.json from video_dataset_builder')
    parser.add_argument('-o', '--output', help='Output training_data.json')
    
    args = parser.parse_args()
    convert_dataset(args.input, args.output)


if __name__ == "__main__":
    main()

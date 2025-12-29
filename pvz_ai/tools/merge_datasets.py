# -*- coding: utf-8 -*-
"""
Merge Datasets - Gộp training data từ tất cả pvz_level folders
Lấy file mới nhất (timestamp cao nhất) từ mỗi folder
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


def find_latest_training_file(folder: Path) -> Optional[Path]:
    """Tìm file training_data mới nhất trong folder"""
    pattern = re.compile(r"training_data.*_(\d{8}_\d{6})\.json$")
    
    files = []
    for f in folder.iterdir():
        if f.is_file() and f.name.startswith("training_data") and f.suffix == ".json":
            match = pattern.search(f.name)
            if match:
                timestamp = match.group(1)
                files.append((timestamp, f))
    
    if not files:
        return None
    
    # Sort by timestamp descending, lấy file mới nhất
    files.sort(key=lambda x: x[0], reverse=True)
    return files[0][1]


def find_all_level_folders(base_dir: Path) -> List[Path]:
    """Tìm tất cả folders pvz_level*"""
    folders = []
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith("pvz_level"):
            folders.append(f)
    
    # Sort by level number
    def get_level_num(p: Path) -> int:
        match = re.search(r"pvz_level(\d+)", p.name)
        return int(match.group(1)) if match else 0
    
    folders.sort(key=get_level_num)
    return folders


def merge_datasets(base_dir: str = "data/ai_labeler", output_path: str = None) -> Dict:
    """
    Merge training data từ tất cả pvz_level folders
    
    Args:
        base_dir: Thư mục chứa các pvz_level folders
        output_path: Path output (default: data/ai_labeler/training_data_final.json)
    """
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return {}
    
    # Tìm tất cả level folders
    level_folders = find_all_level_folders(base_dir)
    print(f"📁 Found {len(level_folders)} level folders")
    
    merged_data = []
    sources = []
    
    for folder in level_folders:
        latest_file = find_latest_training_file(folder)
        
        if latest_file:
            print(f"  ✓ {folder.name}: {latest_file.name}")
            
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                merged_data.extend(data)
                sources.append({
                    "folder": folder.name,
                    "file": latest_file.name,
                    "samples": len(data)
                })
            else:
                merged_data.append(data)
                sources.append({
                    "folder": folder.name,
                    "file": latest_file.name,
                    "samples": 1
                })
        else:
            print(f"  ✗ {folder.name}: No training_data found")
    
    if not merged_data:
        print("❌ No data to merge")
        return {}
    
    # Output path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_path is None:
        output_path = base_dir / f"training_data_final_{timestamp}.json"
    else:
        output_path = Path(output_path)
    
    # Save merged data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    # Stats
    action_stats = {}
    for sample in merged_data:
        action = sample.get("action", "unknown")
        action_stats[action] = action_stats.get(action, 0) + 1
    
    print(f"\n{'='*50}")
    print(f"✅ Merged {len(merged_data)} samples from {len(sources)} folders")
    print(f"📊 Actions: {action_stats}")
    print(f"💾 Output: {output_path}")
    print(f"{'='*50}")
    
    return {
        "output": str(output_path),
        "total_samples": len(merged_data),
        "sources": sources,
        "action_stats": action_stats
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge training datasets from all pvz_level folders")
    parser.add_argument("-d", "--dir", default="data/ai_labeler", help="Base directory")
    parser.add_argument("-o", "--output", help="Output file path")
    args = parser.parse_args()
    
    merge_datasets(args.dir, args.output)


if __name__ == "__main__":
    main()

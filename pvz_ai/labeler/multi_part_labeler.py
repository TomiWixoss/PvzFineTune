# -*- coding: utf-8 -*-
"""
Multi-Part Video Labeler - Xử lý video có nhiều parts
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from ..utils.time_utils import parse_time, format_time
from .ai_labeler import AIVideoLabeler


class MultiPartLabeler:
    """Xử lý video với nhiều parts tuần tự"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.labeler = AIVideoLabeler(api_key)
    
    def find_parts(self, video_path: str) -> List[str]:
        """Tìm tất cả parts: video_2.mp4 -> video_2_part1.mp4, video_2_part2.mp4..."""
        video_path = Path(video_path)
        pattern = re.compile(rf"^{re.escape(video_path.stem)}_part(\d+){re.escape(video_path.suffix)}$")
        
        parts = []
        for f in video_path.parent.iterdir():
            if f.is_file() and (match := pattern.match(f.name)):
                parts.append((int(match.group(1)), str(f)))
        
        parts.sort(key=lambda x: x[0])
        return [p[1] for p in parts]
    
    def adjust_times(self, actions: List, offset_sec: int) -> List:
        """Điều chỉnh timestamp theo offset"""
        result = []
        for a in actions:
            new_a = a.copy()
            new_a["time"] = format_time(parse_time(a.get("time", "0:00")) + offset_sec)
            result.append(new_a)
        return result
    
    def merge_training(self, files: List[str], output: str) -> str:
        """Merge training data files"""
        merged = []
        for f in files:
            if Path(f).exists():
                data = json.loads(Path(f).read_text(encoding="utf-8"))
                merged.extend(data if isinstance(data, list) else [data])
        
        Path(output).write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✅ Merged {len(files)} files -> {output} ({len(merged)} samples)")
        return output
    
    def process(self, video_path: str, output_path: Optional[str] = None, part_duration: int = 60) -> Dict:
        """Xử lý video với tất cả parts"""
        parts = self.find_parts(video_path)
        
        if not parts:
            print(f"📹 No parts, processing: {video_path}")
            return self.labeler.process_video(video_path, output_path)
        
        print(f"\n{'='*50}\n🎬 Found {len(parts)} parts\n{'='*50}")
        
        video_name = Path(video_path).stem
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path(output_path).parent if output_path else Path(f"data/ai_labeler/{video_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        all_actions, training_files, results = [], [], []
        
        for i, part in enumerate(parts):
            print(f"\n📹 Part {i+1}/{len(parts)}: {Path(part).name}")
            
            result = self.labeler.process_video(part, str(out_dir / f"part{i+1}.json"))
            results.append(result)
            
            if result["validation"]["passed"]:
                adjusted = self.adjust_times(result["actions"], i * part_duration)
                all_actions.extend(adjusted)
                if "training_data" in result:
                    training_files.append(result["training_data"])
            
            # Save progress sau mỗi part
            progress = {"done": i + 1, "total": len(parts), "actions": len(all_actions), "training_files": training_files}
            Path(out_dir / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(f"💾 Saved progress: {i+1}/{len(parts)} parts")
        
        # Merge training
        merged_path = None
        if training_files:
            merged_path = str(out_dir / f"training_data_{ts}.json")
            self.merge_training(training_files, merged_path)
        
        passed = sum(1 for r in results if r["validation"]["passed"])
        combined = {
            "video": video_path,
            "parts": len(parts),
            "passed": passed,
            "actions": all_actions,
            "training_data": merged_path,
        }
        
        Path(out_dir / f"combined_{ts}.json").write_text(
            json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        
        print(f"\n🎉 Done! {passed}/{len(parts)} parts | {len(all_actions)} actions")
        return combined


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Video path (sẽ tìm parts tự động)")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("-d", "--duration", type=int, default=60, help="Part duration (s)")
    args = parser.parse_args()
    
    MultiPartLabeler().process(args.video, args.output, args.duration)


if __name__ == "__main__":
    main()

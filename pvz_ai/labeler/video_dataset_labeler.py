# -*- coding: utf-8 -*-
"""
Video Dataset Labeler - Gemini xem video tạo dataset với thinking
Giữ nguyên cấu hình Gemini cũ, output thêm thinking field
Hỗ trợ multi-part videos
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from google.genai import types

from ..core.constants import GEMINI_MODEL_NAME, VIDEO_FPS
from .gemini_client import GeminiKeyManager, is_rate_limit_error, is_retryable_error
from .validator import ActionValidator


def load_yolo_labels(metadata_path: str = "models/yolo/metadata.yaml") -> dict:
    """Load labels từ YOLO metadata.yaml"""
    import yaml
    path = Path(metadata_path)
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('names', {})


def get_plant_types(labels: dict) -> list:
    """Lọc ra các loại plant từ labels"""
    plant_types = []
    exclude_suffixes = ['_cooldown', '_ready', '_reward']
    exclude_names = ['sun', 'zombie', 'zombies']
    
    for idx, name in labels.items():
        if name in exclude_names:
            continue
        if any(name.endswith(suffix) for suffix in exclude_suffixes):
            continue
        plant_types.append(name)
    return plant_types


def build_system_prompt() -> str:
    """Build system prompt cho dataset generation với thinking"""
    labels = load_yolo_labels()
    plant_types = get_plant_types(labels)
    plant_list = ", ".join(plant_types) if plant_types else "pea_shooter, sunflower"
    
    return f"""You are an AI trainer for Plants vs Zombies. Watch the gameplay video and create training dataset.

## TASK:
Watch video, analyze each important moment, create dataset with:
1. **game_state**: Game state at that moment
2. **thinking**: Reasoning behind the decision (1-2 sentences, SHORT)
3. **action**: Action (plant or wait)
4. **arguments**: Action parameters

## GAME STATE FORMAT:
`PLANTS:[(type,row,col),...]. ZOMBIES:[(type,row,col),...]. SEEDS:[(type,status),...]`
- status: "ready" or "cooldown"
- row: 0-4 (0=top), col: 0-8 (0=left)

## PLANT TYPES (from YOLO):
{plant_list}

## OUTPUT FORMAT (JSON array):
```json
[
  {{
    "game_state": "PLANTS:[]. ZOMBIES:[]. SEEDS:[(pea_shooter,ready),(sunflower,cooldown)]",
    "thinking": "No zombie. Pea_shooter ready. Plant at row 2 col 0 to prepare defense.",
    "action": "plant",
    "arguments": {{"plant_type": "pea_shooter", "row": 2, "col": 0}}
  }},
  {{
    "game_state": "PLANTS:[(pea_shooter,2,0)]. ZOMBIES:[(zombie,2,8)]. SEEDS:[(pea_shooter,cooldown),(sunflower,ready)]",
    "thinking": "Zombie at row 2. Already have defense. Seed cooldown. Wait.",
    "action": "wait",
    "arguments": {{}}
  }},
  {{
    "game_state": "PLANTS:[(pea_shooter,2,0)]. ZOMBIES:[(zombie,1,7)]. SEEDS:[(pea_shooter,ready),(sunflower,ready)]",
    "thinking": "Zombie at row 1, no defense there. Pea_shooter ready. Plant at row 1.",
    "action": "plant",
    "arguments": {{"plant_type": "pea_shooter", "row": 1, "col": 0}}
  }}
]
```

## RULES:
1. **thinking** MUST explain logic: why this action, where is zombie, seed status
2. **game_state** must reflect state AT THE MOMENT of decision
3. If zombie at row X, prioritize defending row X
4. Only plant when seed "ready", wait when "cooldown"
5. Don't plant on occupied cells
6. Create MANY diverse samples from video (early game, mid game, emergency, economy)

⚠️ Return ONLY JSON array, no other text.
"""


class VideoDatasetLabeler:
    """Gemini xem video tạo dataset với thinking"""
    
    def __init__(self, api_key: Optional[str] = None):
        keys = [api_key] if api_key else None
        self.key_manager = GeminiKeyManager(keys)
        
        self.config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            response_mime_type="application/json",
            system_instruction=[types.Part.from_text(text=build_system_prompt())],
        )
        
        self.all_samples: List[Dict] = []
    
    def _load_video(self, video_path: str) -> tuple:
        """Load video bytes"""
        print(f"📦 Loading video: {video_path}")
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        print(f"   Size: {len(video_bytes) / (1024 * 1024):.1f} MB")
        return video_bytes, "video/mp4"
    
    def _call_gemini(self, video_bytes: bytes, mime_type: str, prompt: str) -> List[Dict]:
        """Gọi Gemini API"""
        while self.key_manager.has_available_key():
            try:
                client = self.key_manager.get_client()
                print(f"🤖 Calling Gemini with key {self.key_manager.get_current_key_info()}...")
                
                parts = [
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type=mime_type),
                        video_metadata=types.VideoMetadata(fps=VIDEO_FPS),
                    ),
                    types.Part.from_text(text=prompt),
                ]
                
                contents = [types.Content(role="user", parts=parts)]
                
                full_text = ""
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_MODEL_NAME,
                    contents=contents,
                    config=self.config,
                ):
                    if chunk.text:
                        full_text += chunk.text
                        print(".", end="", flush=True)
                print()
                
                samples = json.loads(full_text)
                print(f"📋 Gemini returned {len(samples)} samples")
                return samples
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                if not self.key_manager.rotate_key():
                    break
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted")
                return []
            except Exception as e:
                print(f"⚠️ Error: {e}")
                if is_rate_limit_error(e):
                    if not self.key_manager.rotate_key():
                        break
                else:
                    if not self.key_manager.rotate_key():
                        break
        
        print("❌ All keys exhausted")
        return []
    
    def _validate_sample(self, sample: Dict) -> bool:
        """Validate single sample"""
        required = ["game_state", "thinking", "action", "arguments"]
        if not all(k in sample for k in required):
            return False
        
        if sample["action"] not in ["plant", "wait"]:
            return False
        
        if sample["action"] == "plant":
            args = sample.get("arguments", {})
            if not all(k in args for k in ["plant_type", "row", "col"]):
                return False
            try:
                row, col = int(args["row"]), int(args["col"])
                if not (0 <= row <= 4 and 0 <= col <= 8):
                    return False
            except:
                return False
        
        if not sample.get("thinking"):
            return False
        
        return True
    
    def _validate_with_game_logic(self, samples: List[Dict]) -> Dict:
        """Validate samples với game logic"""
        return ActionValidator.validate_training_data(samples)
    
    def process_video(self, video_path: str, output_dir: str = "data/training") -> Dict:
        """Process video và tạo dataset"""
        print(f"\n{'='*60}")
        print(f"🎬 VIDEO DATASET LABELER")
        print(f"{'='*60}")
        print(f"Video: {video_path}")
        print(f"Model: {GEMINI_MODEL_NAME} | Thinking: HIGH")
        
        # Setup output
        video_name = Path(video_path).stem
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Load video
        video_bytes, mime_type = self._load_video(video_path)
        
        # Call Gemini
        prompt = "Watch this gameplay video and create training dataset with thinking for each decision."
        samples = self._call_gemini(video_bytes, mime_type, prompt)
        
        if not samples:
            return {"success": False, "error": "No samples generated"}
        
        # Save raw
        raw_path = output_path / f"{video_name}_raw_{timestamp}.json"
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"💾 Raw saved: {raw_path}")
        
        # Validate format
        print("\n🔍 Validating samples...")
        valid_samples = [s for s in samples if self._validate_sample(s)]
        print(f"   Format valid: {len(valid_samples)}/{len(samples)}")
        
        # Validate game logic
        validation = self._validate_with_game_logic(valid_samples)
        print(f"   Game logic: {validation['score']:.1f}%")
        
        if validation.get('errors'):
            print(f"   Errors: {len(validation['errors'])}")
            for err in validation['errors'][:5]:
                print(f"      {err}")
        
        # Get clean samples
        clean_samples = validation.get('valid_samples', valid_samples)
        
        # Save clean dataset
        clean_path = output_path / f"{video_name}_dataset_{timestamp}.json"
        with open(clean_path, 'w', encoding='utf-8') as f:
            json.dump(clean_samples, f, indent=2, ensure_ascii=False)
        print(f"✅ Dataset saved: {clean_path}")
        
        # Stats
        stats = {
            "total_raw": len(samples),
            "format_valid": len(valid_samples),
            "game_logic_valid": len(clean_samples),
            "actions": {}
        }
        for s in clean_samples:
            action = s.get("action", "unknown")
            stats["actions"][action] = stats["actions"].get(action, 0) + 1
        
        print(f"\n📊 Stats:")
        print(f"   Total: {stats['game_logic_valid']} samples")
        print(f"   Actions: {stats['actions']}")
        
        self.all_samples = clean_samples
        
        return {
            "success": True,
            "video": video_path,
            "output": str(clean_path),
            "stats": stats,
            "validation_score": validation['score']
        }
    
    def find_video_parts(self, video_path: str) -> List[str]:
        """Tìm tất cả parts của video: video.mp4 -> video_part1.mp4, video_part2.mp4..."""
        video_path = Path(video_path)
        base_name = video_path.stem
        suffix = video_path.suffix
        
        # Pattern: basename_part1.mp4, basename_part2.mp4, ...
        pattern = re.compile(rf"^{re.escape(base_name)}_part(\d+){re.escape(suffix)}$")
        
        parts = []
        for f in video_path.parent.iterdir():
            if f.is_file():
                match = pattern.match(f.name)
                if match:
                    parts.append((int(match.group(1)), str(f)))
        
        parts.sort(key=lambda x: x[0])
        return [p[1] for p in parts]
    
    def process_multiple_videos(self, video_paths: List[str], output_dir: str = "data/training") -> Dict:
        """Process nhiều videos và merge dataset"""
        print(f"\n{'='*60}")
        print(f"🎬 MULTI-VIDEO DATASET LABELER")
        print(f"{'='*60}")
        print(f"Videos: {len(video_paths)}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        all_samples = []
        results = []
        
        for i, video_path in enumerate(video_paths):
            print(f"\n📹 [{i+1}/{len(video_paths)}] {Path(video_path).name}")
            
            result = self.process_video(video_path, output_dir)
            results.append(result)
            
            if result["success"]:
                all_samples.extend(self.all_samples)
                print(f"   ✅ Added {len(self.all_samples)} samples")
            else:
                print(f"   ❌ Failed: {result.get('error')}")
        
        # Merge all samples
        merged_path = output_path / f"merged_dataset_{timestamp}.json"
        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
        # Stats
        total_stats = {"total": len(all_samples), "actions": {}}
        for s in all_samples:
            action = s.get("action", "unknown")
            total_stats["actions"][action] = total_stats["actions"].get(action, 0) + 1
        
        print(f"\n{'='*60}")
        print(f"📊 MERGED STATS:")
        print(f"   Videos processed: {len(results)}")
        print(f"   Successful: {sum(1 for r in results if r['success'])}")
        print(f"   Total samples: {total_stats['total']}")
        print(f"   Actions: {total_stats['actions']}")
        print(f"✅ Merged dataset: {merged_path}")
        
        return {
            "success": True,
            "output": str(merged_path),
            "videos_processed": len(results),
            "total_samples": len(all_samples),
            "stats": total_stats
        }
    
    def process_with_parts(self, video_path: str, output_dir: str = "data/training") -> Dict:
        """Process video và tự động tìm parts"""
        parts = self.find_video_parts(video_path)
        
        if not parts:
            # Không có parts, process single video
            print(f"📹 No parts found, processing single video: {video_path}")
            return self.process_video(video_path, output_dir)
        
        print(f"📹 Found {len(parts)} parts for {Path(video_path).name}")
        return self.process_multiple_videos(parts, output_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video Dataset Labeler với Thinking")
    parser.add_argument("video", nargs='+', help="Path to video(s)")
    parser.add_argument("-o", "--output", default="data/training", help="Output directory")
    parser.add_argument("-k", "--api-key", help="Gemini API key")
    parser.add_argument("--find-parts", action="store_true", help="Auto find video parts")
    args = parser.parse_args()
    
    labeler = VideoDatasetLabeler(api_key=args.api_key)
    
    if len(args.video) == 1:
        if args.find_parts:
            result = labeler.process_with_parts(args.video[0], args.output)
        else:
            result = labeler.process_video(args.video[0], args.output)
    else:
        result = labeler.process_multiple_videos(args.video, args.output)
    
    if result["success"]:
        print(f"\n✅ Done! Dataset: {result['output']}")
    else:
        print(f"\n❌ Failed: {result.get('error')}")


if __name__ == "__main__":
    main()

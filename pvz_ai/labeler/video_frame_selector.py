# -*- coding: utf-8 -*-
"""
Video Frame Selector - Gemini phân tích video và chọn frames đa dạng cho YOLO training
"""

import json
import yaml
import cv2
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from google.genai import types

from ..core.constants import GEMINI_MODEL_NAME, VIDEO_FPS
from .gemini_client import GeminiKeyManager, is_rate_limit_error, is_retryable_error
from ..utils.time_utils import parse_time


def load_yolo_classes(metadata_path: str = "models/yolo/metadata.yaml") -> Dict[int, str]:
    """Load class names từ YOLO metadata"""
    path = Path(metadata_path)
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('names', {})


def build_frame_selector_prompt(yolo_classes: Dict[int, str]) -> str:
    """Build system prompt cho frame selection"""
    class_list = "\n".join([f"  - {idx}: {name}" for idx, name in yolo_classes.items()])
    
    return f"""---
Bạn là chuyên gia phân tích video game Plants vs Zombies để chọn frames tốt nhất cho training YOLO object detection.

## 🎯 MỤC TIÊU:
Chọn các frames ĐA DẠNG và ĐẦY ĐỦ các class để fine-tune YOLO model.

## 📦 CÁC CLASS CẦN DETECT (từ YOLO model):
{class_list}

## 📋 TIÊU CHÍ CHỌN FRAME:

### 1. ĐA DẠNG CLASS:
- Ưu tiên frames có NHIỀU class khác nhau xuất hiện
- Đảm bảo mỗi class đều có ít nhất vài frames
- Đặc biệt chú ý các class hiếm (cherry_bomb_reward, sunflower_reward)

### 2. ĐA DẠNG TRẠNG THÁI:
- Seed packet: ready vs cooldown
- Zombie: đang đi, đang ăn, nhiều zombie
- Sun: đang rơi, đang bay, nhiều sun
- Plant: mới trồng, đang bắn, nhiều plant

### 3. ĐA DẠNG BỐI CẢNH:
- Đầu game (ít plant, ít zombie)
- Giữa game (nhiều plant, zombie bắt đầu)
- Cuối game (nhiều zombie, intense)
- Các góc nhìn khác nhau của grid

### 4. CHẤT LƯỢNG FRAME:
- Không bị blur/motion blur
- Objects rõ ràng, không bị che khuất quá nhiều
- Lighting tốt

## ⏱️ TIMESTAMP FORMAT:
Format: `M:SS.mmm` (phút:giây.miligiây)
Ví dụ: `0:15.500`, `1:02.250`

## 🎬 OUTPUT FORMAT:
```json
[
  {{
    "time": "0:05.250",
    "classes_visible": ["sunflower_ready", "pea_shooter_ready", "sun"],
    "reason": "Đầu game, seed packets ready, sun đang rơi",
    "priority": "high"
  }},
  {{
    "time": "0:18.500",
    "classes_visible": ["sunflower", "pea_shooter", "zombie", "sun"],
    "reason": "Có plant đã trồng và zombie đầu tiên",
    "priority": "high"
  }}
]
```

## 📊 YÊU CẦU:
1. Chọn 20-50 frames đa dạng từ video
2. Đảm bảo coverage tất cả classes
3. Priority: "high" (quan trọng), "medium" (bổ sung), "low" (backup)
4. Khoảng cách giữa các frame ít nhất 1-2 giây để tránh trùng lặp
5. CHỈ trả về JSON array, không text khác
"""


class VideoFrameSelector:
    """Gemini phân tích video và chọn frames cho YOLO training"""
    
    def __init__(self, api_key: Optional[str] = None, metadata_path: str = "models/yolo/metadata.yaml"):
        keys = [api_key] if api_key else None
        self.key_manager = GeminiKeyManager(keys)
        self.yolo_classes = load_yolo_classes(metadata_path)
        
        system_prompt = build_frame_selector_prompt(self.yolo_classes)
        
        self.config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            response_mime_type="application/json",
            system_instruction=[types.Part.from_text(text=system_prompt)],
        )
    
    def _load_video(self, video_path: str) -> tuple:
        """Load video bytes"""
        print(f"📦 Loading video: {video_path}")
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        print(f"   Size: {len(video_bytes) / (1024 * 1024):.1f} MB")
        return video_bytes, "video/mp4"
    
    def _call_ai(self, video_bytes: bytes, mime_type: str) -> List[Dict]:
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
                    types.Part.from_text(text="Phân tích video và chọn các frames đa dạng cho YOLO training."),
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
                
                frames = json.loads(full_text)
                print(f"📋 AI selected {len(frames)} frames")
                return frames
                
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
    
    def analyze_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """Phân tích video và xuất JSON timestamps"""
        print(f"\n{'='*50}")
        print(f"🎬 Analyzing: {video_path}")
        print(f"   Model: {GEMINI_MODEL_NAME}")
        print(f"   YOLO Classes: {len(self.yolo_classes)}")
        print(f"{'='*50}\n")
        
        video_name = Path(video_path).stem
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_path:
            output_file = Path(output_path)
        else:
            output_dir = Path(f"data/yolo_frames/{video_name}")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"frame_selection_{timestamp_str}.json"
        
        video_bytes, mime_type = self._load_video(video_path)
        frames = self._call_ai(video_bytes, mime_type)
        
        if not frames:
            return {"video": video_path, "frames": [], "success": False}
        
        # Thống kê classes
        class_coverage = {}
        for frame in frames:
            for cls in frame.get("classes_visible", []):
                class_coverage[cls] = class_coverage.get(cls, 0) + 1
        
        result = {
            "video": video_path,
            "timestamp": datetime.now().isoformat(),
            "model": GEMINI_MODEL_NAME,
            "yolo_classes": self.yolo_classes,
            "total_frames": len(frames),
            "class_coverage": class_coverage,
            "frames": frames,
            "success": True
        }
        
        # Save JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {output_file}")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Total frames: {len(frames)}")
        print(f"   Class coverage:")
        for cls, count in sorted(class_coverage.items(), key=lambda x: -x[1]):
            print(f"      {cls}: {count} frames")
        
        return result


def extract_selected_frames(
    video_path: str,
    json_path: str,
    output_dir: Optional[str] = None,
    priority_filter: Optional[List[str]] = None
) -> int:
    """Trích xuất frames từ video theo JSON timestamps"""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    frames_info = data.get("frames", [])
    if not frames_info:
        print("❌ No frames in JSON")
        return 0
    
    # Filter by priority
    if priority_filter:
        frames_info = [f for f in frames_info if f.get("priority") in priority_filter]
    
    video_name = Path(video_path).stem
    if output_dir is None:
        output_dir = f"data/yolo_frames/{video_name}/images"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 Video FPS: {video_fps}")
    print(f"📦 Extracting {len(frames_info)} frames...")
    
    extracted = 0
    for i, frame_info in enumerate(frames_info):
        time_str = frame_info.get("time", "0:00.000")
        time_sec = parse_time(time_str)
        frame_num = int(time_sec * video_fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Tên file chứa thông tin classes
            classes = "_".join(frame_info.get("classes_visible", [])[:3])
            filename = f"frame_{i:04d}_{time_str.replace(':', '-').replace('.', '_')}_{classes[:30]}.jpg"
            filepath = Path(output_dir) / filename
            cv2.imwrite(str(filepath), frame)
            extracted += 1
        else:
            print(f"⚠️ Cannot read frame at {time_str}")
    
    cap.release()
    print(f"✅ Extracted {extracted} frames to {output_dir}")
    
    # Save metadata for labeling
    metadata_path = Path(output_dir).parent / "frames_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "video": video_path,
            "output_dir": output_dir,
            "frames": frames_info,
            "extracted_count": extracted
        }, f, indent=2, ensure_ascii=False)
    print(f"💾 Metadata saved: {metadata_path}")
    
    return extracted


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video Frame Selector for YOLO Training")
    parser.add_argument("command", choices=["analyze", "extract"], help="Command to run")
    parser.add_argument("video", help="Path to video")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("-j", "--json", help="JSON file for extract command")
    parser.add_argument("-p", "--priority", nargs="+", default=["high", "medium"], 
                       help="Priority filter for extract")
    parser.add_argument("-k", "--api-key", help="Gemini API key")
    args = parser.parse_args()
    
    if args.command == "analyze":
        selector = VideoFrameSelector(api_key=args.api_key)
        selector.analyze_video(args.video, args.output)
    
    elif args.command == "extract":
        if not args.json:
            print("❌ --json required for extract command")
            return
        extract_selected_frames(args.video, args.json, args.output, args.priority)


if __name__ == "__main__":
    main()

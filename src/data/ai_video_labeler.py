# -*- coding: utf-8 -*-
"""
AI Video Labeler - Gemini xem video PvZ và xuất JSON actions
Có validation logic và self-correction loop
"""

import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env file
load_dotenv()

# ===========================================
# CONFIG
# ===========================================
MODEL_NAME = "gemini-3-flash-preview"
VIDEO_FPS = 24
MAX_INLINE_SIZE_MB = 20  # Dùng base64 nếu video < 20MB

VALID_ACTIONS = ["plant", "wait"]
VALID_PLANT_TYPES = [
    "pea_shooter", "sunflower", "wall_nut", "cherry_bomb",
    "snow_pea", "repeater", "potato_mine", "chomper",
    "squash", "threepeater", "jalapeno", "spikeweed"
]
GRID_ROWS = 5  # 0-4
GRID_COLS = 9  # 0-8

SYSTEM_PROMPT = """---
Bạn là chuyên gia phân tích gameplay Plants vs Zombies. Xem video frame-by-frame và ghi lại hành động TRỒNG CÂY của người chơi.

## ⚠️ LƯU Ý QUAN TRỌNG
- **KHÔNG ghi action thu thập sun** - việc này do code rule tự động xử lý
- **CHỈ ghi 2 loại action**: `plant` (trồng cây) và `wait` (chờ)
- AI sẽ học cách quyết định KHI NÀO và Ở ĐÂU nên trồng cây

## 🎯 2 LOẠI ACTION:

### 1. `plant` - Trồng cây
**KHI NÀO**: Người chơi click seed packet VÀ đặt cây xuống grid
**THAM SỐ**:
- `plant_type`: Loại cây (pea_shooter, sunflower, wall_nut, ...)
- `row`: Hàng (0-4, 0=trên cùng)
- `col`: Cột (0-8, 0=trái nhất)

```json
{"time": "0:18","action": "plant","args": { "plant_type": "pea_shooter", "row": 2, "col": 0 },"note": "trồng pea_shooter hàng giữa, cột đầu"}
```

**GRID**:
```
Row 0 (top)    : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 1          : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 2 (middle) : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 3          : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 4 (bottom) : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Col 0 → → → → → → → → Col 8
```

**PLANT TYPES** (phổ biến):
- `pea_shooter` - Bắn đậu
- `sunflower` - Hoa hướng dương
- `wall_nut` - Hạt óc chó (chắn)
- `cherry_bomb` - Bom cherry
- `snow_pea` - Đậu băng
- `repeater` - Bắn đậu đôi

### 2. `wait` - Chờ
**KHI NÀO**:
- Seed packet đang cooldown (xám)
- Không đủ sun để trồng
- Đang chờ zombie xuất hiện
- Không cần trồng thêm

```json
{"time": "0:25","action": "wait","args": {},"note": "seed cooldown, chờ"}
```

## ✅ VALIDATION CHECKLIST:
| Action  | Điều kiện BẮT BUỘC                                  |
| ------- | --------------------------------------------------- |
| `plant` | Người chơi THỰC SỰ trồng cây tại timestamp đó       |
| `wait`  | Không có hành động trồng cây trong khoảng thời gian |

## ❌ LỖI THƯỜNG GẶP:
```json
// ❌ SAI: Ghi collect_sun (không dùng nữa!)
{"time": "0:15", "action": "collect_sun"}

// ❌ SAI: Thiếu plant_type
{"time": "0:20", "action": "plant", "args": {"row": 2, "col": 1}}

// ❌ SAI: Ghi plant khi chưa thực sự trồng
{"time": "0:20", "action": "plant", "args": {"plant_type": "pea_shooter", "row": 2, "col": 1}}
```

## ✅ VÍ DỤ ĐÚNG:
```json
[
{"time": "0:05","action": "wait","args": {},"note": "game starting, chờ đủ sun"},
{"time": "0:09","action": "plant","args": { "plant_type": "pea_shooter", "row": 2, "col": 0 },"note": "trồng pea_shooter đầu tiên"},
{"time": "0:15","action": "wait","args": {},"note": "seed cooldown"},
{"time": "0:22","action": "plant","args": { "plant_type": "pea_shooter", "row": 2, "col": 1 },"note": "trồng thêm pea_shooter"}
]
```

## 🎬 OUTPUT FORMAT:
```json
[{"time": "M:SS","action": "plant | wait","args": { "plant_type": "...", "row": N, "col": N },"note": "lý do action"}]
```

**Time format**: `M:SS` hoặc `M:SS.S`

---
⚠️ Nhớ:
1. **CHỈ ghi `plant` và `wait`** - KHÔNG ghi collect_sun
2. **`plant` phải có đủ**: plant_type, row, col
3. **Ghi timestamp chính xác** khi người chơi đặt cây xuống
4. **Note** lý do để hiểu context"""


class AIVideoLabeler:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required. Set in .env or pass via --api-key")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Generation config matching reference code
        self.config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            response_mime_type="application/json",
            system_instruction=[types.Part.from_text(text=SYSTEM_PROMPT)],
        )
    
    def _get_video_size_mb(self, video_path: str) -> float:
        return os.path.getsize(video_path) / (1024 * 1024)
    
    def _load_video_inline(self, video_path: str) -> types.Part:
        """Load video as base64 inline data"""
        print(f"📦 Loading video inline (base64)...")
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        
        return types.Part.from_bytes(
            data=video_bytes,
            mime_type="video/mp4",
            video_metadata=types.VideoMetadata(fps=VIDEO_FPS),
        )
    
    def _upload_video(self, video_path: str) -> types.Part:
        """Upload video via Files API"""
        print(f"📤 Uploading video via Files API...")
        video_file = self.client.files.upload(file=video_path)
        
        while video_file.state.name == "PROCESSING":
            print("⏳ Processing...")
            time.sleep(5)
            video_file = self.client.files.get(name=video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError(f"Upload failed: {video_file.state.name}")
        
        print(f"✅ Ready: {video_file.uri}")
        return types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type)
    
    def load_video(self, video_path: str) -> types.Part:
        """Auto-select: inline for small videos, upload for large"""
        size_mb = self._get_video_size_mb(video_path)
        print(f"📁 Video size: {size_mb:.1f} MB")
        
        if size_mb < MAX_INLINE_SIZE_MB:
            return self._load_video_inline(video_path)
        else:
            return self._upload_video(video_path)
    
    def analyze_video(self, video_part: types.Part, prompt: str = "Xem video và tạo danh sách actions JSON.") -> list:
        """Gọi Gemini phân tích video"""
        print(f"🤖 AI đang xem video (thinking=HIGH)...")
        
        contents = [
            types.Content(
                role="user",
                parts=[video_part, types.Part.from_text(text=prompt)],
            )
        ]
        
        # Stream response
        full_text = ""
        for chunk in self.client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=self.config,
        ):
            if chunk.text:
                full_text += chunk.text
                print(".", end="", flush=True)
        print()
        
        try:
            actions = json.loads(full_text)
            print(f"📋 AI trả về {len(actions)} actions")
            return actions
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            print(f"Raw: {full_text[:500]}")
            return []
    
    def validate_actions(self, actions: list) -> dict:
        """Validate logic của actions"""
        errors = []
        warnings = []
        
        grid = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        last_plant_time = {}
        
        for i, action in enumerate(actions):
            action_type = action.get("action")
            time_str = action.get("time", "?")
            args = action.get("args", {})
            
            if action_type not in VALID_ACTIONS:
                errors.append(f"[{i}] time={time_str}: Invalid action '{action_type}'")
                continue
            
            if action_type == "plant":
                plant_type = args.get("plant_type")
                row = args.get("row")
                col = args.get("col")
                
                if not plant_type:
                    errors.append(f"[{i}] time={time_str}: plant thiếu plant_type")
                    continue
                if row is None or col is None:
                    errors.append(f"[{i}] time={time_str}: plant thiếu row/col")
                    continue
                
                if plant_type not in VALID_PLANT_TYPES:
                    warnings.append(f"[{i}] time={time_str}: plant_type '{plant_type}' không phổ biến")
                
                if not (0 <= row < GRID_ROWS):
                    errors.append(f"[{i}] time={time_str}: row={row} ngoài range 0-4")
                    continue
                if not (0 <= col < GRID_COLS):
                    errors.append(f"[{i}] time={time_str}: col={col} ngoài range 0-8")
                    continue
                
                if grid[row][col] is not None:
                    existing = grid[row][col]
                    if not (plant_type == "wall_nut" and existing == "wall_nut"):
                        errors.append(f"[{i}] time={time_str}: Ô ({row},{col}) đã có {existing}")
                        continue
                
                time_seconds = self._parse_time(time_str)
                if plant_type in last_plant_time:
                    diff = time_seconds - last_plant_time[plant_type]
                    if 0 < diff < 3:
                        warnings.append(f"[{i}] time={time_str}: {plant_type} trồng quá nhanh ({diff:.1f}s)")
                
                grid[row][col] = plant_type
                last_plant_time[plant_type] = time_seconds
        
        total = len(actions)
        error_count = len(errors)
        score = ((total - error_count) / total * 100) if total > 0 else 0
        
        return {
            "passed": score >= 90,
            "score": score,
            "total": total,
            "errors": errors,
            "warnings": warnings
        }
    
    def _parse_time(self, time_str: str) -> float:
        try:
            parts = time_str.replace(".", ":").split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + float(parts[2]) / 10
            return 0
        except:
            return 0
    
    def request_correction(self, video_part: types.Part, validation: dict) -> list:
        """Gửi feedback về AI để sửa lỗi"""
        print("🔄 Yêu cầu AI sửa lỗi...")
        
        error_feedback = "\n".join(validation["errors"][:20])
        warning_feedback = "\n".join(validation["warnings"][:10])
        
        prompt = f"""
Kết quả validation KHÔNG ĐẠT (score: {validation['score']:.1f}%).

## LỖI CẦN SỬA:
{error_feedback}

## CẢNH BÁO:
{warning_feedback}

## YÊU CẦU:
1. Xem lại video cẩn thận
2. Sửa các lỗi trên (không trồng chồng, row 0-4, col 0-8)
3. Trả về JSON array đã sửa
"""
        return self.analyze_video(video_part, prompt)
    
    def _save_raw_response(self, actions: list, iteration: int, output_dir: Path):
        """Lưu raw response ngay sau khi nhận từ AI"""
        raw_file = output_dir / f"raw_iter_{iteration}.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(actions, f, indent=2, ensure_ascii=False)
        print(f"💾 Raw saved: {raw_file}")
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     max_iterations: int = 3) -> dict:
        """Main pipeline"""
        print(f"\n{'='*50}")
        print(f"🎬 Processing: {video_path}")
        print(f"   Model: {MODEL_NAME} | Thinking: HIGH")
        print(f"{'='*50}\n")
        
        # Setup output directory
        if output_path is None:
            output_path = f"data/ai_labeled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_part = self.load_video(video_path)
        actions = self.analyze_video(video_part)
        
        # Lưu raw ngay sau khi nhận
        self._save_raw_response(actions, 0, output_dir)
        
        validation = {"score": 0, "passed": False, "errors": [], "warnings": []}
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            validation = self.validate_actions(actions)
            print(f"📊 Score: {validation['score']:.1f}% | Errors: {len(validation['errors'])} | Warnings: {len(validation['warnings'])}")
            
            if validation["passed"]:
                print(f"✅ PASSED!")
                break
            
            if iteration < max_iterations - 1:
                actions = self.request_correction(video_part, validation)
                # Lưu raw sau mỗi lần correction
                self._save_raw_response(actions, iteration + 1, output_dir)
            else:
                print(f"⚠️ Max iterations reached")
        
        result = {
            "video": video_path,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "validation": validation,
            "actions": actions
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Final saved: {output_path}")
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Labeler for PvZ")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("-k", "--api-key", help="Gemini API key")
    parser.add_argument("-i", "--iterations", type=int, default=3)
    args = parser.parse_args()
    
    labeler = AIVideoLabeler(api_key=args.api_key)
    result = labeler.process_video(args.video, args.output, args.iterations)
    
    print(f"\n{'='*50}")
    print(f"Final: {result['validation']['score']:.1f}% | {len(result['actions'])} actions")


if __name__ == "__main__":
    main()

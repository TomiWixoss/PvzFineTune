# -*- coding: utf-8 -*-
"""
AI Video Labeler - Gemini xem video PvZ và xuất JSON actions
GIỮ NGUYÊN LOGIC GEMINI 100% - KHÔNG SỬA
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from google.genai import types

from ..core.constants import GEMINI_MODEL_NAME, VIDEO_FPS
from .gemini_client import GeminiKeyManager, is_rate_limit_error, is_retryable_error
from .validator import ActionValidator
from .auto_fixer import ActionAutoFixer

# ===========================================
# SYSTEM PROMPT - GIỮ NGUYÊN 100%
# ===========================================
SYSTEM_PROMPT = """---
Bạn là chuyên gia phân tích gameplay Plants vs Zombies. Xem video frame-by-frame và ghi lại hành động TRỒNG CÂY của người chơi.

## ⚠️ LƯU Ý QUAN TRỌNG
- **KHÔNG ghi action thu thập sun** - việc này do code rule tự động xử lý
- **CHỈ ghi 2 loại action**: `plant` (trồng cây) và `wait` (chờ)
- **TIMESTAMP CHÍNH XÁC**: Ghi tới millisecond (M:SS.mmm)

## ⏱️ TIMESTAMP FORMAT (BẮT BUỘC):
Format: `M:SS.mmm` (phút:giây.miligiây)
- M = phút (0, 1, 2, ...)
- SS = giây (00-59)
- mmm = miligiây (000-999)

Ví dụ:
- `0:05.250` = 5 giây 250ms
- `0:18.500` = 18 giây 500ms  
- `1:02.750` = 1 phút 2 giây 750ms
- `2:30.125` = 2 phút 30 giây 125ms

⚠️ PHẢI ghi đủ 3 chữ số miligiây!

## 🎯 2 LOẠI ACTION:

### 1. `plant` - Trồng cây
**THAM SỐ**:
- `plant_type`: Loại cây (pea_shooter, sunflower, wall_nut, ...)
- `row`: Hàng (0-4, 0=trên cùng)
- `col`: Cột (0-8, 0=trái nhất)

**GRID**:
```
Row 0 (top)    : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 1          : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 2 (middle) : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 3          : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 4 (bottom) : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Col 0 → → → → → → → → Col 8
```

### 2. `wait` - Chờ (seed cooldown, không đủ sun, ...)

## 🎬 OUTPUT FORMAT:
```json
[
  {"time": "0:18.500", "action": "plant", "args": {"plant_type": "pea_shooter", "row": 2, "col": 0}, "note": "..."},
  {"time": "0:25.250", "action": "wait", "args": {}, "note": "..."}
]
```

⚠️ CHỈ trả về JSON array, không text khác.
⚠️ Timestamp PHẢI có millisecond (M:SS.mmm)
"""


class AIVideoLabeler:
    """AI Video Labeler - GIỮ NGUYÊN LOGIC 100%"""
    
    def __init__(self, api_key: Optional[str] = None):
        keys = [api_key] if api_key else None
        self.key_manager = GeminiKeyManager(keys)
        
        self.config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            response_mime_type="application/json",
            system_instruction=[types.Part.from_text(text=SYSTEM_PROMPT)],
        )
        
        self.history: List[types.Content] = []
    
    def _load_video(self, video_path: str) -> tuple:
        """Load video bytes"""
        print(f"📦 Loading video: {video_path}")
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        
        size_mb = len(video_bytes) / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")
        
        return video_bytes, "video/mp4"
    
    def _call_ai_chat(self, video_bytes: bytes, mime_type: str, prompt: str, is_first: bool = False) -> List:
        """Gọi Gemini API - GIỮ NGUYÊN LOGIC 100%"""
        while self.key_manager.has_available_key():
            try:
                client = self.key_manager.get_client()
                print(f"🤖 Calling AI with key {self.key_manager.get_current_key_info()}...")
                
                if is_first:
                    parts = [
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type=mime_type),
                            video_metadata=types.VideoMetadata(fps=VIDEO_FPS),
                        ),
                        types.Part.from_text(text=prompt),
                    ]
                else:
                    parts = [types.Part.from_text(text=prompt)]
                
                contents = [types.Content(role="user", parts=parts)]
                
                full_text = ""
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_MODEL_NAME,
                    contents=self.history + contents,
                    config=self.config,
                ):
                    if chunk.text:
                        full_text += chunk.text
                        print(".", end="", flush=True)
                print()
                
                actions = json.loads(full_text)
                print(f"📋 AI returned {len(actions)} actions")
                
                self.history.append(types.Content(role="user", parts=parts))
                self.history.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=full_text)],
                ))
                
                return actions
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                if not self.key_manager.rotate_key():
                    break
                continue
            
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted by user")
                return []
                
            except Exception as e:
                print(f"⚠️ Error: {e}")
                
                if is_rate_limit_error(e):
                    print("   → Rate limit, rotating key...")
                    if not self.key_manager.rotate_key():
                        break
                elif is_retryable_error(e):
                    print("   → Overload, retrying...")
                    continue
                else:
                    print("   → Unknown error, rotating key...")
                    if not self.key_manager.rotate_key():
                        break
        
        print("❌ All keys exhausted or blocked")
        return []
    
    def reset_chat(self):
        """Reset chat history"""
        self.history = []
    
    def _save_json(self, data: Any, path: str):
        """Save JSON to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {path}")
    
    def _filter_valid_actions(self, actions: List, validation: Dict) -> List:
        """Lọc chỉ giữ lại các actions không có error"""
        if not validation.get("validated_samples"):
            return actions
        
        valid_actions = []
        for sample in validation["validated_samples"]:
            if sample.get("valid", False):
                idx = sample.get("id", 0) - 1
                if 0 <= idx < len(actions):
                    valid_actions.append(actions[idx])
        
        return valid_actions
    
    def _get_game_states_for_errors(self, video_path: str, actions: List, validation: Dict) -> str:
        """Thu thập game_state tại các timestamp có lỗi"""
        try:
            from ..data.video_dataset_builder import VideoDatasetBuilder
            
            validated_samples = validation.get("validated_samples", [])
            error_samples = [s for s in validated_samples if not s.get("valid", True)]
            
            if not error_samples:
                return "Không có thông tin game_state"
            
            lines = []
            for sample in error_samples[:10]:
                idx = sample.get("id", 0)
                timestamp = sample.get("timestamp", "?")
                game_state = sample.get("game_state", {})
                error = sample.get("error", "")
                action = sample.get("action", {})
                
                state_text = game_state.get("text", "N/A")
                
                lines.append(f"""
### Action [{idx}] tại {timestamp}:
- **Lỗi**: {error}
- **Action**: {action.get('type')} - {action.get('args')}
- **Game State**: {state_text}
""")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Không thể lấy game_state: {e}"
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """Main pipeline - GIỮ NGUYÊN LOGIC 100%"""
        print(f"\n{'='*50}")
        print(f"🎬 Processing: {video_path}")
        print(f"   Model: {GEMINI_MODEL_NAME} | Thinking: HIGH")
        print(f"{'='*50}\n")
        
        self.reset_chat()
        
        video_name = Path(video_path).stem
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_path is None:
            output_dir = Path(f"data/ai_labeler/{video_name}")
        else:
            output_dir = Path(output_path).parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        final_output = output_dir / f"result_{timestamp_str}.json"
        training_output = output_dir / f"training_data_{timestamp_str}.json"
        
        video_bytes, mime_type = self._load_video(video_path)
        auto_fixer = ActionAutoFixer(video_path)
        
        actions = self._call_ai_chat(
            video_bytes, mime_type,
            "Xem video và tạo danh sách actions JSON.",
            is_first=True
        )
        
        if not actions:
            print("❌ AI không trả về actions, dừng.")
            return {"video": video_path, "actions": [], "validation": {"passed": False, "score": 0}}
        
        iteration = 0
        raw_path = output_dir / f"raw_iter_{iteration}.json"
        self._save_json(actions, str(raw_path))
        
        validation = {"score": 0, "passed": False, "errors": [], "warnings": []}
        
        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            print("🔧 Auto-fixing timestamps...")
            fix_result = auto_fixer.fix_actions(actions)
            
            if fix_result["fix_count"] > 0:
                print(f"   ✅ Fixed {fix_result['fix_count']} actions")
                actions = fix_result["fixed_actions"]
                fixed_path = output_dir / f"fixed_iter_{iteration}.json"
                self._save_json(actions, str(fixed_path))
            
            try:
                validation = ActionValidator.validate_with_video(actions, video_path)
            except Exception as e:
                print(f"   ⚠️ Cannot validate with video: {e}")
                validation = ActionValidator.validate_simple(actions)
            
            print(ActionValidator.format_result(validation))
            
            if validation["passed"]:
                print("✅ PASSED!")
                break
            
            unfixable = fix_result.get("unfixable_errors", [])
            if not unfixable:
                unfixable = validation.get("errors", [])
            
            if not unfixable:
                print("✅ No more errors!")
                break
            
            if not self.key_manager.has_available_key():
                print("❌ Hết key, dừng.")
                break
            
            game_states_info = self._get_game_states_for_errors(video_path, actions, validation)
            
            error_feedback = "\n".join(unfixable[:20])
            prompt = f"""
Kết quả validation KHÔNG ĐẠT (score: {validation['score']:.1f}%).

## LỖI CẦN SỬA:
{error_feedback}

## TRẠNG THÁI GAME TẠI CÁC TIMESTAMP LỖI:
{game_states_info}

## YÊU CẦU:
1. Xem lại video (bạn đã xem ở lượt trước)
2. Dựa vào game_state ở trên để hiểu:
   - PLANTS: cây đã trồng ở đâu (không được trồng chồng)
   - SEEDS: seed packet nào ready/cooldown
3. **LƯU Ý**: Có thể bạn đã ghi THỪA action (video chỉ trồng 3 cây mà bạn ghi 4). Hãy xem lại và XÓA action không có thật.
4. Sửa các lỗi:
   - Không trồng chồng lên ô đã có cây
   - row trong range 0-4, col trong range 0-8
   - CHỈ plant khi seed packet READY (không cooldown)
   - Timestamp phải chính xác khi cây THỰC SỰ được đặt xuống
5. **TIMESTAMP FORMAT**: M:SS.mmm (phút:giây.miligiây, VD: 0:18.500)
6. Trả về JSON array đã sửa
"""
            self.key_manager.reset_blocked()
            
            new_actions = self._call_ai_chat(video_bytes, mime_type, prompt, is_first=False)
            
            if not new_actions:
                print("❌ AI không trả về actions, dừng.")
                break
            
            actions = new_actions
            raw_path = output_dir / f"raw_iter_{iteration}.json"
            self._save_json(actions, str(raw_path))
        
        auto_fixer.close()
        
        clean_actions = self._filter_valid_actions(actions, validation)
        print(f"\n📋 Clean actions: {len(clean_actions)}/{len(actions)}")
        
        result = {
            "video": video_path,
            "timestamp": datetime.now().isoformat(),
            "model": GEMINI_MODEL_NAME,
            "iterations": iteration,
            "validation": {
                "passed": validation["passed"],
                "score": validation["score"],
                "total": validation["total"],
                "errors_count": len(validation.get("errors", [])),
                "warnings_count": len(validation.get("warnings", [])),
            },
            "actions": clean_actions,
            "all_actions": actions,
        }
        
        self._save_json(result, str(final_output))
        print(f"\n💾 Final: {final_output}")
        
        if validation["passed"] and validation["score"] >= 100:
            print("\n🎯 Building training data...")
            training_path = self._build_training_data(video_path, clean_actions, output_dir, str(training_output))
            if training_path:
                result["training_data"] = training_path
        
        return result
    
    def _build_training_data(self, video_path: str, actions: List, output_dir: Path, training_path: str) -> Optional[str]:
        """Tự động build training data"""
        try:
            from ..data.video_dataset_builder import VideoDatasetBuilder
            from ..data.dataset_converter import convert_dataset
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            actions_file = output_dir / f"actions_temp_{timestamp}.json"
            
            builder_actions = []
            for action in actions:
                builder_actions.append({
                    "time": action.get("time", "0:00"),
                    "action": action.get("action", "wait"),
                    "args": action.get("args", {})
                })
            
            self._save_json(builder_actions, str(actions_file))
            
            dataset_path = output_dir / f"dataset_temp_{timestamp}.json"
            
            builder = VideoDatasetBuilder(video_path)
            if builder.load():
                builder.process_actions_file(str(actions_file), str(dataset_path), save_frames=False)
                builder.close()
                
                convert_dataset(str(dataset_path), training_path)
                
                actions_file.unlink()
                dataset_path.unlink()
                
                print(f"✅ Training data: {training_path}")
                return training_path
            else:
                print("❌ Cannot load video for training data")
                return None
                
        except Exception as e:
            print(f"❌ Error building training data: {e}")
            return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Labeler for PvZ")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("-k", "--api-key", help="Gemini API key")
    args = parser.parse_args()
    
    labeler = AIVideoLabeler(api_key=args.api_key)
    result = labeler.process_video(args.video, args.output)
    
    print(f"\n{'='*50}")
    print(f"Final: {result['validation']['score']:.1f}% | {len(result['actions'])} actions")


if __name__ == "__main__":
    main()

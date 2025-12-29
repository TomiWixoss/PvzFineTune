# -*- coding: utf-8 -*-
"""
AI Video Labeler - Gemini xem video PvZ và xuất JSON actions
Main orchestrator: load video -> call AI -> validate -> retry loop
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from google.genai import types

from .gemini_key_manager import (
    GeminiKeyManager,
    is_rate_limit_error,
    is_retryable_error,
)
from .action_validator import (
    validate_actions_with_video, 
    validate_actions_simple,
    format_validation_result
)
from .action_auto_fixer import ActionAutoFixer

# ===========================================
# CONFIG
# ===========================================
MODEL_NAME = "gemini-3-flash-preview"
VIDEO_FPS = 24

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
    def __init__(self, api_key: Optional[str] = None):
        # Nếu truyền 1 key thì dùng, không thì load từ env
        keys = [api_key] if api_key else None
        self.key_manager = GeminiKeyManager(keys)
        
        self.config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            response_mime_type="application/json",
            system_instruction=[types.Part.from_text(text=SYSTEM_PROMPT)],
        )
        
        # Chat history để giữ context qua các lượt
        self.history: list[types.Content] = []
    
    def _load_video(self, video_path: str) -> tuple[bytes, str]:
        """Load video bytes"""
        print(f"📦 Loading video: {video_path}")
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        
        size_mb = len(video_bytes) / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")
        
        return video_bytes, "video/mp4"
    
    def _create_chat(self, video_part: types.Part):
        """Tạo chat session mới với video"""
        client = self.key_manager.get_client()
        
        # Tạo chat với history (nếu có)
        chat = client.chats.create(
            model=MODEL_NAME,
            config=self.config,
            history=self.history,
        )
        
        return chat, client
    
    def _call_ai_chat(self, video_bytes: bytes, mime_type: str, prompt: str, is_first: bool = False) -> list:
        """
        Gọi Gemini API qua chat conversation
        - Lượt đầu: gửi video + prompt
        - Lượt sau: chỉ gửi prompt (AI đã có context video từ history)
        """
        while self.key_manager.has_available_key():
            try:
                client = self.key_manager.get_client()
                print(f"🤖 Calling AI with key {self.key_manager.get_current_key_info()}...")
                
                # Build parts theo đúng format reference code
                if is_first:
                    # Lượt đầu: gửi video với video_metadata + prompt
                    parts = [
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type=mime_type),
                            video_metadata=types.VideoMetadata(fps=VIDEO_FPS),
                        ),
                        types.Part.from_text(text=prompt),
                    ]
                else:
                    # Lượt sau: chỉ gửi prompt
                    parts = [types.Part.from_text(text=prompt)]
                
                contents = [types.Content(role="user", parts=parts)]
                
                # Stream response (không dùng chat, dùng generate_content_stream như reference)
                full_text = ""
                for chunk in client.models.generate_content_stream(
                    model=MODEL_NAME,
                    contents=self.history + contents,
                    config=self.config,
                ):
                    if chunk.text:
                        full_text += chunk.text
                        print(".", end="", flush=True)
                print()
                
                # Parse JSON
                actions = json.loads(full_text)
                print(f"📋 AI returned {len(actions)} actions")
                
                # Cập nhật history
                self.history.append(types.Content(role="user", parts=parts))
                self.history.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=full_text)],
                ))
                
                return actions
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                print(f"   Raw: {full_text[:200]}...")
                print("   → Rotating key and retrying...")
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
                    print("   → Overload, retrying immediately...")
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
    
    def _save_json(self, data: any, path: str):
        """Save JSON to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {path}")
    
    def _filter_valid_actions(self, actions: list, validation: dict) -> list:
        """Lọc chỉ giữ lại các actions không có error"""
        if not validation.get("validated_samples"):
            return actions
        
        valid_actions = []
        for sample in validation["validated_samples"]:
            if sample.get("valid", False):
                # Tìm action tương ứng
                idx = sample.get("id", 0) - 1
                if 0 <= idx < len(actions):
                    valid_actions.append(actions[idx])
        
        return valid_actions
    
    def _get_game_states_for_errors(self, video_path: str, actions: list, validation: dict) -> str:
        """
        Thu thập game_state tại các timestamp có lỗi
        Returns: string mô tả game_state cho AI
        """
        try:
            from .video_dataset_builder import VideoDatasetBuilder
            
            # Lấy danh sách samples có lỗi
            validated_samples = validation.get("validated_samples", [])
            error_samples = [s for s in validated_samples if not s.get("valid", True)]
            
            if not error_samples:
                return "Không có thông tin game_state"
            
            lines = []
            for sample in error_samples[:10]:  # Max 10 errors
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
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
    ) -> dict:
        """
        Main pipeline:
        1. Load video
        2. Call AI (chat) -> get actions
        3. Save raw immediately
        4. TỰ FIX: Quét ±2s tìm timestamp seed ready
        5. Validate với video + YOLO
        6. Nếu còn lỗi không fix được → gửi errors về AI
        7. Lặp vô hạn tới khi pass hoặc hết key
        8. Cuối cùng lưu bản sạch (chỉ actions không error)
        """
        print(f"\n{'='*50}")
        print(f"🎬 Processing: {video_path}")
        print(f"   Model: {MODEL_NAME} | Thinking: HIGH")
        print(f"   Mode: Auto-fix + Loop until pass > 90%")
        print(f"{'='*50}\n")
        
        # Reset chat history cho video mới
        self.reset_chat()
        
        # Setup output - gom vào data/ai_labeler/<video_name>/
        video_name = Path(video_path).stem
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_path is None:
            output_dir = Path(f"data/ai_labeler/{video_name}")
        else:
            output_dir = Path(output_path).parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Final output paths
        final_output = output_dir / f"result_{timestamp_str}.json"
        training_output = output_dir / f"training_data_{timestamp_str}.json"
        
        # Load video once
        video_bytes, mime_type = self._load_video(video_path)
        
        # Init auto fixer
        auto_fixer = ActionAutoFixer(video_path)
        
        # Initial call (lượt đầu, gửi video)
        actions = self._call_ai_chat(
            video_bytes, mime_type,
            "Xem video và tạo danh sách actions JSON.",
            is_first=True
        )
        
        if not actions:
            print("❌ AI không trả về actions, dừng.")
            return {"video": video_path, "actions": [], "validation": {"passed": False, "score": 0}}
        
        # Save raw immediately
        iteration = 0
        raw_path = output_dir / f"raw_iter_{iteration}.json"
        self._save_json(actions, str(raw_path))
        
        # Validation loop - lặp vô hạn tới khi pass hoặc hết key
        validation = {"score": 0, "passed": False, "errors": [], "warnings": []}
        
        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # BƯỚC 1: Tự fix trước
            print("🔧 Auto-fixing timestamps...")
            fix_result = auto_fixer.fix_actions(actions)
            
            if fix_result["fix_count"] > 0:
                print(f"   ✅ Fixed {fix_result['fix_count']} actions")
                actions = fix_result["fixed_actions"]
                # Save fixed version
                fixed_path = output_dir / f"fixed_iter_{iteration}.json"
                self._save_json(actions, str(fixed_path))
            
            # BƯỚC 2: Validate
            try:
                validation = validate_actions_with_video(actions, video_path)
                print("   (Validated with video + YOLO)")
            except Exception as e:
                print(f"   ⚠️ Cannot validate with video: {e}")
                print("   (Using simple validation)")
                validation = validate_actions_simple(actions)
            
            print(format_validation_result(validation))
            
            if validation["passed"]:
                print("✅ PASSED!")
                break
            
            # BƯỚC 3: Nếu còn lỗi không fix được → gửi AI
            unfixable = fix_result.get("unfixable_errors", [])
            if not unfixable:
                # Dùng validation errors
                unfixable = validation.get("errors", [])
            
            if not unfixable:
                print("✅ No more errors!")
                break
            
            # Check còn key không
            if not self.key_manager.has_available_key():
                print("❌ Hết key, dừng.")
                break
            
            # Thu thập game_state cho các lỗi
            game_states_info = self._get_game_states_for_errors(video_path, actions, validation)
            
            # Build correction prompt với game_state
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
            # Reset blocked keys for retry
            self.key_manager.reset_blocked()
            
            # Gọi tiếp trong cùng conversation
            new_actions = self._call_ai_chat(video_bytes, mime_type, prompt, is_first=False)
            
            if not new_actions:
                print("❌ AI không trả về actions, dừng.")
                break
            
            actions = new_actions
            
            # Save each iteration
            raw_path = output_dir / f"raw_iter_{iteration}.json"
            self._save_json(actions, str(raw_path))
        
        # Close auto fixer
        auto_fixer.close()
        
        # Lọc chỉ giữ actions không error
        clean_actions = self._filter_valid_actions(actions, validation)
        print(f"\n📋 Clean actions: {len(clean_actions)}/{len(actions)}")
        
        # Final result
        result = {
            "video": video_path,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "iterations": iteration,
            "validation": {
                "passed": validation["passed"],
                "score": validation["score"],
                "total": validation["total"],
                "errors_count": len(validation.get("errors", [])),
                "warnings_count": len(validation.get("warnings", [])),
            },
            "actions": clean_actions,  # Chỉ lưu actions sạch
            "all_actions": actions,    # Lưu cả bản gốc để debug
        }
        
        self._save_json(result, str(final_output))
        print(f"\n💾 Final: {final_output}")
        
        # Nếu pass 100% → tự động build training data
        if validation["passed"] and validation["score"] >= 100:
            print("\n🎯 Building training data...")
            training_path = self._build_training_data(video_path, clean_actions, output_dir, str(training_output))
            if training_path:
                result["training_data"] = training_path
        
        return result
    
    def _build_training_data(self, video_path: str, actions: list, output_dir: Path, training_path: str) -> Optional[str]:
        """
        Tự động build training data từ actions đã validate
        1. Dùng VideoDatasetBuilder để tạo dataset với game_state
        2. Dùng dataset_to_training để convert sang format Gemma
        """
        try:
            from .video_dataset_builder import VideoDatasetBuilder
            from .dataset_to_training import convert_dataset
            
            # Tạo file actions tạm
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            actions_file = output_dir / f"actions_temp_{timestamp}.json"
            
            # Convert format cho VideoDatasetBuilder
            builder_actions = []
            for action in actions:
                builder_actions.append({
                    "time": action.get("time", "0:00"),
                    "action": action.get("action", "wait"),
                    "args": action.get("args", {})
                })
            
            self._save_json(builder_actions, str(actions_file))
            
            # Build dataset (intermediate)
            dataset_path = output_dir / f"dataset_temp_{timestamp}.json"
            
            builder = VideoDatasetBuilder(video_path)
            if builder.load():
                builder.process_actions_file(str(actions_file), str(dataset_path), save_frames=False)
                builder.close()
                
                # Convert sang format Gemma training
                convert_dataset(str(dataset_path), training_path)
                
                # Xóa file tạm
                actions_file.unlink()
                dataset_path.unlink()
                
                print(f"✅ Training data: {training_path}")
                return training_path
            else:
                print("❌ Cannot load video for training data")
                return None
                
        except Exception as e:
            print(f"❌ Error building training data: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Labeler for PvZ")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("-k", "--api-key", help="Gemini API key (optional, uses .env if not provided)")
    args = parser.parse_args()
    
    labeler = AIVideoLabeler(api_key=args.api_key)
    result = labeler.process_video(args.video, args.output)
    
    print(f"\n{'='*50}")
    print(f"Final: {result['validation']['score']:.1f}% | {len(result['actions'])} actions")


if __name__ == "__main__":
    main()

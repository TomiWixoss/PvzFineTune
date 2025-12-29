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
from .action_validator import validate_actions, format_validation_result

# ===========================================
# CONFIG
# ===========================================
MODEL_NAME = "gemini-2.5-flash"
VIDEO_FPS = 24

SYSTEM_PROMPT = """---
Bạn là chuyên gia phân tích gameplay Plants vs Zombies. Xem video frame-by-frame và ghi lại hành động TRỒNG CÂY của người chơi.

## ⚠️ LƯU Ý QUAN TRỌNG
- **KHÔNG ghi action thu thập sun** - việc này do code rule tự động xử lý
- **CHỈ ghi 2 loại action**: `plant` (trồng cây) và `wait` (chờ)

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
  {"time": "M:SS", "action": "plant", "args": {"plant_type": "...", "row": N, "col": N}, "note": "..."},
  {"time": "M:SS", "action": "wait", "args": {}, "note": "..."}
]
```

⚠️ CHỈ trả về JSON array, không text khác.
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
    
    def _load_video(self, video_path: str) -> types.Part:
        """Load video as inline bytes"""
        print(f"📦 Loading video: {video_path}")
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        
        size_mb = len(video_bytes) / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")
        
        return types.Part.from_bytes(
            data=video_bytes,
            mime_type="video/mp4",
            video_metadata=types.VideoMetadata(fps=VIDEO_FPS),
        )
    
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
    
    def _call_ai_chat(self, video_part: types.Part, prompt: str, is_first: bool = False) -> list:
        """
        Gọi Gemini API qua chat conversation
        - Lượt đầu: gửi video + prompt
        - Lượt sau: chỉ gửi prompt (AI đã có context video từ history)
        """
        while self.key_manager.has_available_key():
            try:
                client = self.key_manager.get_client()
                print(f"🤖 Calling AI with key {self.key_manager.get_current_key_info()}...")
                
                # Tạo chat session với history
                chat = client.chats.create(
                    model=MODEL_NAME,
                    config=self.config,
                    history=self.history,
                )
                
                # Build message parts
                if is_first:
                    # Lượt đầu: gửi video + prompt
                    parts = [video_part, types.Part.from_text(text=prompt)]
                else:
                    # Lượt sau: chỉ gửi prompt (video đã trong history)
                    parts = [types.Part.from_text(text=prompt)]
                
                # Stream response
                full_text = ""
                response_stream = chat.send_message_stream(message=parts)
                
                for chunk in response_stream:
                    if chunk.text:
                        full_text += chunk.text
                        print(".", end="", flush=True)
                print()
                
                # Parse JSON
                actions = json.loads(full_text)
                print(f"📋 AI returned {len(actions)} actions")
                
                # Cập nhật history cho lượt sau
                # User message
                self.history.append(types.Content(
                    role="user",
                    parts=parts,
                ))
                # AI response
                self.history.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=full_text)],
                ))
                
                return actions
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                print(f"   Raw: {full_text[:200]}...")
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
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_iterations: int = 3
    ) -> dict:
        """
        Main pipeline:
        1. Load video
        2. Call AI (chat) -> get actions
        3. Save raw immediately
        4. Validate
        5. If not passed, send errors back to AI (same chat) and repeat
        """
        print(f"\n{'='*50}")
        print(f"🎬 Processing: {video_path}")
        print(f"   Model: {MODEL_NAME} | Thinking: HIGH")
        print(f"   Mode: Chat conversation (with history)")
        print(f"{'='*50}\n")
        
        # Reset chat history cho video mới
        self.reset_chat()
        
        # Setup output
        if output_path is None:
            output_path = f"data/ai_labeled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load video once
        video_part = self._load_video(video_path)
        
        # Initial call (lượt đầu, gửi video)
        actions = self._call_ai_chat(
            video_part, 
            "Xem video và tạo danh sách actions JSON.",
            is_first=True
        )
        
        # Save raw immediately
        raw_path = output_dir / f"raw_iter_0.json"
        self._save_json(actions, str(raw_path))
        
        # Validation loop
        validation = {"score": 0, "passed": False, "errors": [], "warnings": []}
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            validation = validate_actions(actions)
            print(format_validation_result(validation))
            
            if validation["passed"]:
                print("✅ PASSED!")
                break
            
            if iteration < max_iterations - 1:
                # Build correction prompt
                error_feedback = "\n".join(validation["errors"][:20])
                prompt = f"""
Kết quả validation KHÔNG ĐẠT (score: {validation['score']:.1f}%).

## LỖI CẦN SỬA:
{error_feedback}

## YÊU CẦU:
1. Xem lại video (bạn đã xem ở lượt trước)
2. Sửa các lỗi (không trồng chồng, row 0-4, col 0-8)
3. Trả về JSON array đã sửa
"""
                # Reset blocked keys for retry
                self.key_manager.reset_blocked()
                
                # Gọi tiếp trong cùng chat (is_first=False, không gửi lại video)
                actions = self._call_ai_chat(video_part, prompt, is_first=False)
                
                # Save each iteration
                raw_path = output_dir / f"raw_iter_{iteration + 1}.json"
                self._save_json(actions, str(raw_path))
            else:
                print("⚠️ Max iterations reached")
        
        # Final result
        result = {
            "video": video_path,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "iterations": min(iteration + 1, max_iterations),
            "validation": validation,
            "actions": actions
        }
        
        self._save_json(result, output_path)
        print(f"\n💾 Final: {output_path}")
        
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Labeler for PvZ")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("-k", "--api-key", help="Gemini API key (optional, uses .env if not provided)")
    parser.add_argument("-i", "--iterations", type=int, default=3, help="Max correction iterations")
    args = parser.parse_args()
    
    labeler = AIVideoLabeler(api_key=args.api_key)
    result = labeler.process_video(args.video, args.output, args.iterations)
    
    print(f"\n{'='*50}")
    print(f"Final: {result['validation']['score']:.1f}% | {len(result['actions'])} actions")


if __name__ == "__main__":
    main()

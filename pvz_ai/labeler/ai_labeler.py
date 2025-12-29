# -*- coding: utf-8 -*-
"""
AI Video Labeler - Gemini xem video PvZ và xuất JSON actions
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
from .prompts import get_system_prompt, CORRECTION_PROMPT_TEMPLATE
from .training_builder import TrainingBuilder


class AIVideoLabeler:
    """AI Video Labeler - Orchestrator chính"""
    
    def __init__(self, api_key: Optional[str] = None):
        keys = [api_key] if api_key else None
        self.key_manager = GeminiKeyManager(keys)
        
        self.config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            response_mime_type="application/json",
            system_instruction=[types.Part.from_text(text=get_system_prompt())],
        )
        
        self.history: List[types.Content] = []
    
    def reset_chat(self):
        """Reset chat history"""
        self.history = []
    
    def _load_video(self, video_path: str) -> tuple:
        """Load video bytes"""
        print(f"📦 Loading video: {video_path}")
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        print(f"   Size: {len(video_bytes) / (1024 * 1024):.1f} MB")
        return video_bytes, "video/mp4"
    
    def _save_json(self, data: Any, path: str):
        """Save JSON to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {path}")
    
    def _call_ai(self, video_bytes: bytes, mime_type: str, prompt: str, is_first: bool = False) -> List:
        """Gọi Gemini API"""
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
                
                # Update history
                self.history.append(types.Content(role="user", parts=parts))
                self.history.append(types.Content(role="model", parts=[types.Part.from_text(text=full_text)]))
                
                return actions
                
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
                elif is_retryable_error(e):
                    continue
                else:
                    if not self.key_manager.rotate_key():
                        break
        
        print("❌ All keys exhausted")
        return []
    
    def _filter_valid_actions(self, actions: List, validation: Dict) -> List:
        """Lọc actions không có error"""
        if not validation.get("validated_samples"):
            return actions
        
        valid_actions = []
        for sample in validation["validated_samples"]:
            if sample.get("valid", False):
                idx = sample.get("id", 0) - 1
                if 0 <= idx < len(actions):
                    valid_actions.append(actions[idx])
        return valid_actions
    
    def _get_error_game_states(self, validation: Dict) -> str:
        """Thu thập game_state tại các timestamp có lỗi"""
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
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """Main pipeline"""
        print(f"\n{'='*50}")
        print(f"🎬 Processing: {video_path}")
        print(f"   Model: {GEMINI_MODEL_NAME} | Thinking: HIGH")
        print(f"{'='*50}\n")
        
        self.reset_chat()
        
        # Setup paths
        video_name = Path(video_path).stem
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(output_path).parent if output_path else Path(f"data/ai_labeler/{video_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        final_output = output_dir / f"result_{timestamp_str}.json"
        training_output = output_dir / f"training_data_{timestamp_str}.json"
        
        # Load video
        video_bytes, mime_type = self._load_video(video_path)
        auto_fixer = ActionAutoFixer(video_path)
        
        # Initial call
        actions = self._call_ai(video_bytes, mime_type, "Xem video và tạo danh sách actions JSON.", is_first=True)
        
        if not actions:
            return {"video": video_path, "actions": [], "validation": {"passed": False, "score": 0}}
        
        # Save raw
        iteration = 0
        self._save_json(actions, str(output_dir / f"raw_iter_{iteration}.json"))
        
        validation = {"score": 0, "passed": False, "errors": [], "warnings": []}
        validated_actions = {}  # {index: action} - lưu actions đã pass validation
        
        # Validation loop
        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Auto-fix, skip những action đã validated
            print("🔧 Auto-fixing...")
            fix_result = auto_fixer.fix_actions(actions, skip_indices=set(validated_actions.keys()))
            if fix_result["fix_count"] > 0:
                print(f"   ✅ Fixed {fix_result['fix_count']} actions")
                actions = fix_result["fixed_actions"]
                self._save_json(actions, str(output_dir / f"fixed_iter_{iteration}.json"))
            
            # Lấy kết quả validation từ auto_fixer
            unfixable = fix_result.get("unfixable_errors", [])
            
            # Xác định actions nào lỗi
            error_indices = set()
            for err in unfixable:
                if err.startswith("["):
                    try:
                        idx = int(err.split("]")[0][1:])
                        error_indices.add(idx)
                    except:
                        pass
            
            # Lưu actions đã pass vào validated_actions
            for i, action in enumerate(actions):
                if i not in error_indices and i not in validated_actions:
                    validated_actions[i] = action.copy()
            
            total = len(actions)
            error_count = len(error_indices)
            score = ((total - error_count) / total * 100) if total > 0 else 0
            
            validation = {
                "passed": error_count == 0,
                "score": score,
                "total": total,
                "errors": unfixable,
                "warnings": []
            }
            
            print(f"📊 Score: {score:.1f}% ({total} actions, {len(validated_actions)} validated)")
            print(f"   Errors: {error_count}")
            
            if validation["passed"]:
                print("✅ PASSED!")
                break
            
            if not unfixable:
                print("✅ No more errors!")
                break
            
            # Show errors
            print("❌ Errors:")
            for err in unfixable[:5]:
                print(f"   {err}")
            
            if not self.key_manager.has_available_key():
                print("❌ Hết key")
                break
            
            # Build correction prompt - chỉ yêu cầu sửa actions lỗi
            prompt = CORRECTION_PROMPT_TEMPLATE.format(
                score=validation['score'],
                error_feedback="\n".join(unfixable[:20]),
                game_states_info="Xem video để kiểm tra lại"
            )
            
            self.key_manager.reset_blocked()
            new_actions = self._call_ai(video_bytes, mime_type, prompt, is_first=False)
            
            if not new_actions:
                break
            
            # Rebuild actions: giữ validated, thay/xóa error actions
            rebuilt_actions = []
            new_validated = {}  # Rebuild với index mới
            new_idx = 0
            
            for i, action in enumerate(actions):
                if i in validated_actions:
                    # Giữ nguyên action đã validated, cập nhật index mới
                    new_validated[len(rebuilt_actions)] = validated_actions[i]
                    rebuilt_actions.append(validated_actions[i])
                elif i in error_indices:
                    # Thay bằng action mới từ AI (nếu có)
                    if new_idx < len(new_actions):
                        rebuilt_actions.append(new_actions[new_idx])
                        new_idx += 1
                    # Nếu AI bỏ luôn (không có action mới) thì skip
                else:
                    rebuilt_actions.append(action)
            
            # Nếu AI trả về nhiều hơn số error, thêm vào cuối
            while new_idx < len(new_actions):
                rebuilt_actions.append(new_actions[new_idx])
                new_idx += 1
            
            actions = rebuilt_actions
            validated_actions = new_validated
            
            self._save_json(actions, str(output_dir / f"raw_iter_{iteration}.json"))
        
        auto_fixer.close()
        
        # Lấy actions đã validated (theo thứ tự index)
        clean_actions = [validated_actions[i] for i in sorted(validated_actions.keys())]
        if not clean_actions:
            clean_actions = actions
        print(f"\n📋 Clean actions: {len(clean_actions)}/{len(actions)}")
        
        # Build result
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
            },
            "actions": clean_actions,
            "all_actions": actions,
        }
        
        self._save_json(result, str(final_output))
        
        # Build training data if passed
        if validation["passed"] and validation["score"] >= 100:
            print("\n🎯 Building training data...")
            training_path = TrainingBuilder.build(video_path, clean_actions, output_dir, str(training_output))
            if training_path:
                result["training_data"] = training_path
        
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Labeler")
    parser.add_argument("video", help="Path to video")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("-k", "--api-key", help="Gemini API key")
    args = parser.parse_args()
    
    labeler = AIVideoLabeler(api_key=args.api_key)
    result = labeler.process_video(args.video, args.output)
    print(f"\nFinal: {result['validation']['score']:.1f}% | {len(result['actions'])} actions")


if __name__ == "__main__":
    main()

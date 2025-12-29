# -*- coding: utf-8 -*-
"""
Action Auto Fixer - Tự động fix timestamp bằng cách quét video
Quét ±2s quanh timestamp AI trả về để tìm frame seed packet ready
Chỉ gửi lại AI nếu không thể tự fix
"""

import sys
import os
from typing import Any, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ActionAutoFixer:
    """
    Tự động fix actions bằng cách quét video:
    1. Với mỗi action có lỗi seed cooldown
    2. Quét ±2s quanh timestamp
    3. Tìm frame đầu tiên seed packet ready
    4. Fix timestamp
    """
    
    SCAN_RANGE_SECONDS = 2.0  # Quét ±2s
    SCAN_STEP_MS = 42  # Bước quét ~1 frame (video 24fps = 42ms/frame)
    
    def __init__(self, video_path: str, model_path: Optional[str] = None):
        self.video_path = video_path
        self.model_path = model_path
        self.builder = None
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy load video và YOLO"""
        if self._loaded:
            return True
        
        from .video_dataset_builder import VideoDatasetBuilder
        self.builder = VideoDatasetBuilder(self.video_path, model_path=self.model_path)
        if not self.builder.load():
            print("❌ Cannot load video or YOLO")
            return False
        
        self._loaded = True
        return True
    
    def _parse_time(self, time_str: str) -> float:
        """Parse time string → seconds"""
        time_str = time_str.strip()
        if time_str.replace(".", "").replace("-", "").isdigit():
            return float(time_str)
        
        parts = time_str.replace(",", ".").split(":")
        if len(parts) == 2:
            mins = int(parts[0])
            secs = float(parts[1])
            return mins * 60 + secs
        elif len(parts) == 3:
            hours = int(parts[0])
            mins = int(parts[1])
            secs = float(parts[2])
            return hours * 3600 + mins * 60 + secs
        return 0
    
    def _format_time(self, seconds: float) -> str:
        """Seconds → M:SS.mmm"""
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{mins}:{secs:02d}.{millis:03d}"
    
    def _find_seed_ready_timestamp(
        self, 
        plant_type: str, 
        center_time: float
    ) -> Optional[float]:
        """
        Quét ±2s quanh center_time để tìm frame seed packet ready
        Returns: timestamp (seconds) hoặc None nếu không tìm thấy
        """
        if not self._ensure_loaded():
            return None
        
        start_time = max(0, center_time - self.SCAN_RANGE_SECONDS)
        end_time = min(self.builder.duration, center_time + self.SCAN_RANGE_SECONDS)
        
        step = self.SCAN_STEP_MS / 1000.0  # 100ms
        
        # Quét từ center ra 2 bên (ưu tiên gần center)
        times_to_check = []
        t = center_time
        while t >= start_time:
            times_to_check.append(t)
            t -= step
        t = center_time + step
        while t <= end_time:
            times_to_check.append(t)
            t += step
        
        # Sort theo khoảng cách từ center
        times_to_check.sort(key=lambda x: abs(x - center_time))
        
        for t in times_to_check:
            time_str = self._format_time(t)
            frame, frame_num, _ = self.builder.get_frame_at_time(time_str)
            if frame is None:
                continue
            
            game_state = self.builder.detect_game_state(frame)
            seeds = game_state.get("seeds", [])
            
            for seed in seeds:
                if seed.get("type") == plant_type and seed.get("status") == "ready":
                    return t
        
        return None
    
    def _check_cell_empty(self, row: int, col: int, time_seconds: float, grid: list) -> bool:
        """Check ô có trống không (từ tracking grid)"""
        if not (0 <= row < 5 and 0 <= col < 9):
            return False
        return grid[row][col] is None
    
    def fix_actions(self, actions: list[dict[str, Any]]) -> dict:
        """
        Fix actions tự động:
        1. Validate từng action
        2. Nếu lỗi seed cooldown → quét ±2s tìm timestamp đúng
        3. Nếu lỗi trồng chồng → không fix được
        4. Return: fixed actions + unfixable errors
        
        Returns:
            {
                "fixed_actions": list,  # Actions đã fix
                "fix_count": int,       # Số actions đã fix
                "unfixable_errors": list,  # Lỗi không fix được (cần gửi AI)
                "all_passed": bool      # True nếu tất cả OK
            }
        """
        if not self._ensure_loaded():
            return {
                "fixed_actions": actions,
                "fix_count": 0,
                "unfixable_errors": ["Cannot load video"],
                "all_passed": False
            }
        
        fixed_actions = []
        unfixable_errors = []
        fix_count = 0
        
        # Track grid (plants đã trồng)
        grid = [[None for _ in range(9)] for _ in range(5)]
        
        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                unfixable_errors.append(f"[{i}]: Action không phải dict")
                continue
            
            time_str = action.get("time", "0:00")
            action_type = action.get("action")
            args = action.get("args", {}) or {}
            
            # Copy action để fix
            fixed_action = dict(action)
            action_error = None
            was_fixed = False
            
            if action_type == "plant":
                plant_type = args.get("plant_type")
                row = args.get("row")
                col = args.get("col")
                
                # Check required fields
                if not plant_type:
                    action_error = f"[{i}] time={time_str}: plant thiếu plant_type"
                elif row is None or col is None:
                    action_error = f"[{i}] time={time_str}: plant thiếu row/col"
                else:
                    try:
                        row = int(row)
                        col = int(col)
                    except (ValueError, TypeError):
                        action_error = f"[{i}] time={time_str}: row/col phải là số"
                    
                    if action_error is None:
                        # Check range
                        if not (0 <= row < 5):
                            action_error = f"[{i}] time={time_str}: row={row} ngoài range 0-4"
                        elif not (0 <= col < 9):
                            action_error = f"[{i}] time={time_str}: col={col} ngoài range 0-8"
                        # Check trồng chồng
                        elif grid[row][col] is not None:
                            existing = grid[row][col]
                            if not (plant_type == "wall_nut" and existing == "wall_nut"):
                                action_error = f"[{i}] time={time_str}: Ô ({row},{col}) đã có {existing}"
                        else:
                            # Check seed packet ready
                            center_time = self._parse_time(time_str)
                            frame, _, _ = self.builder.get_frame_at_time(time_str)
                            
                            if frame is not None:
                                game_state = self.builder.detect_game_state(frame)
                                seeds = game_state.get("seeds", [])
                                seed_ready = False
                                
                                for seed in seeds:
                                    if seed.get("type") == plant_type and seed.get("status") == "ready":
                                        seed_ready = True
                                        break
                                
                                if not seed_ready:
                                    # TỰ FIX: Quét ±2s tìm timestamp seed ready
                                    print(f"   🔍 [{i}] Seed {plant_type} cooldown tại {time_str}, quét ±2s...")
                                    new_time = self._find_seed_ready_timestamp(plant_type, center_time)
                                    
                                    if new_time is not None:
                                        new_time_str = self._format_time(new_time)
                                        print(f"   ✅ Fixed: {time_str} → {new_time_str}")
                                        fixed_action["time"] = new_time_str
                                        fixed_action["_fixed_from"] = time_str
                                        was_fixed = True
                                        fix_count += 1
                                    else:
                                        action_error = f"[{i}] time={time_str}: Seed {plant_type} không ready trong ±2s"
                            
                            # Update grid nếu OK
                            if action_error is None:
                                grid[row][col] = plant_type
            
            elif action_type == "wait":
                pass  # wait luôn OK
            
            elif action_type is None:
                action_error = f"[{i}] time={time_str}: Thiếu action type"
            else:
                action_error = f"[{i}] time={time_str}: Invalid action '{action_type}'"
            
            if action_error:
                unfixable_errors.append(action_error)
            
            fixed_actions.append(fixed_action)
        
        return {
            "fixed_actions": fixed_actions,
            "fix_count": fix_count,
            "unfixable_errors": unfixable_errors,
            "all_passed": len(unfixable_errors) == 0
        }
    
    def close(self):
        if self.builder:
            self.builder.close()

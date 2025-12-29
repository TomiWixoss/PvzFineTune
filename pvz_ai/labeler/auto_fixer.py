# -*- coding: utf-8 -*-
"""
Action Auto Fixer - Tự động fix timestamp bằng cách quét video
GIỮ NGUYÊN LOGIC 100%
"""

from typing import Any, Dict, List, Optional

from ..core.constants import GRID_ROWS, GRID_COLS
from ..utils.time_utils import parse_time, format_time


class ActionAutoFixer:
    """Tự động fix actions bằng cách quét video"""
    
    SCAN_RANGE_SECONDS = 2.0
    SCAN_STEP_MS = 42
    
    def __init__(self, video_path: str, model_path: Optional[str] = None):
        self.video_path = video_path
        self.model_path = model_path
        self.builder = None
        self._loaded = False
    
    def _ensure_loaded(self) -> bool:
        """Lazy load video và YOLO"""
        if self._loaded:
            return True
        
        from ..data.video_dataset_builder import VideoDatasetBuilder
        self.builder = VideoDatasetBuilder(self.video_path, model_path=self.model_path)
        if not self.builder.load():
            print("❌ Cannot load video or YOLO")
            return False
        
        self._loaded = True
        return True
    
    def _find_seed_ready_timestamp(self, plant_type: str, center_time: float) -> Optional[float]:
        """Quét ±2s quanh center_time để tìm frame seed packet ready"""
        if not self._ensure_loaded():
            return None
        
        start_time = max(0, center_time - self.SCAN_RANGE_SECONDS)
        end_time = min(self.builder.duration, center_time + self.SCAN_RANGE_SECONDS)
        step = self.SCAN_STEP_MS / 1000.0
        
        times_to_check = []
        t = center_time
        while t >= start_time:
            times_to_check.append(t)
            t -= step
        t = center_time + step
        while t <= end_time:
            times_to_check.append(t)
            t += step
        
        times_to_check.sort(key=lambda x: abs(x - center_time))
        
        for t in times_to_check:
            time_str = format_time(t)
            frame, frame_num, _ = self.builder.get_frame_at_time(time_str)
            if frame is None:
                continue
            
            game_state = self.builder.detect_game_state(frame)
            seeds = game_state.get("seeds", [])
            
            for seed in seeds:
                if seed.get("type") == plant_type and seed.get("status") == "ready":
                    return t
        
        return None
    
    def fix_actions(self, actions: List[Dict[str, Any]], skip_indices: set = None) -> Dict:
        """
        Fix actions tự động
        
        Args:
            actions: List of actions
            skip_indices: Set of indices to skip (đã validated trước đó)
        """
        if not self._ensure_loaded():
            return {
                "fixed_actions": actions,
                "fix_count": 0,
                "unfixable_errors": ["Cannot load video"],
                "all_passed": False
            }
        
        skip_indices = skip_indices or set()
        fixed_actions = []
        unfixable_errors = []
        fix_count = 0
        
        grid = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        
        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                unfixable_errors.append(f"[{i}]: Action không phải dict")
                continue
            
            time_str = action.get("time", "0:00")
            action_type = action.get("action")
            args = action.get("args", {}) or {}
            
            fixed_action = dict(action)
            action_error = None
            
            # Skip nếu đã validated và chỉ update grid
            if i in skip_indices:
                if action_type == "plant":
                    row, col = int(args.get("row", 0)), int(args.get("col", 0))
                    if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
                        grid[row][col] = args.get("plant_type")
                fixed_actions.append(fixed_action)
                continue
            
            if action_type == "plant":
                plant_type = args.get("plant_type")
                row = args.get("row")
                col = args.get("col")
                
                if not plant_type:
                    action_error = f"[{i}] time={time_str}: plant thiếu plant_type"
                elif row is None or col is None:
                    action_error = f"[{i}] time={time_str}: plant thiếu row/col"
                else:
                    try:
                        row, col = int(row), int(col)
                    except (ValueError, TypeError):
                        action_error = f"[{i}] time={time_str}: row/col phải là số"
                    
                    if action_error is None:
                        if not (0 <= row < GRID_ROWS):
                            action_error = f"[{i}] time={time_str}: row={row} ngoài range 0-4"
                        elif not (0 <= col < GRID_COLS):
                            action_error = f"[{i}] time={time_str}: col={col} ngoài range 0-8"
                        elif grid[row][col] is not None:
                            existing = grid[row][col]
                            if not (plant_type == "wall_nut" and existing == "wall_nut"):
                                action_error = f"[{i}] time={time_str}: Ô ({row},{col}) đã có {existing}"
                        else:
                            center_time = parse_time(time_str)
                            frame, _, _ = self.builder.get_frame_at_time(time_str)
                            
                            if frame is not None:
                                game_state = self.builder.detect_game_state(frame)
                                seeds = game_state.get("seeds", [])
                                seed_ready = any(
                                    s.get("type") == plant_type and s.get("status") == "ready"
                                    for s in seeds
                                )
                                
                                if not seed_ready:
                                    print(f"   🔍 [{i}] Seed {plant_type} cooldown tại {time_str}, quét ±2s...")
                                    new_time = self._find_seed_ready_timestamp(plant_type, center_time)
                                    
                                    if new_time is not None:
                                        new_time_str = format_time(new_time)
                                        print(f"   ✅ Fixed: {time_str} → {new_time_str}")
                                        fixed_action["time"] = new_time_str
                                        fixed_action["_fixed_from"] = time_str
                                        fix_count += 1
                                    else:
                                        action_error = f"[{i}] time={time_str}: Seed {plant_type} không ready trong ±2s"
                            
                            if action_error is None:
                                grid[row][col] = plant_type
            
            elif action_type == "wait":
                pass
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

# -*- coding: utf-8 -*-
"""
Action Validator - Validate actions với game state từ video
"""

from typing import Any, Dict, List, Optional

from ..core.constants import VALID_ACTIONS, GRID_ROWS, GRID_COLS


class ActionValidator:
    """Validate actions"""
    
    @staticmethod
    def validate_with_video(
        actions: List[Dict[str, Any]], 
        video_path: str,
        model_path: Optional[str] = None
    ) -> Dict:
        """Validate actions bằng cách detect game state từ video"""
        from ..data.video_dataset_builder import VideoDatasetBuilder
        
        errors = []
        warnings = []
        validated_samples = []
        
        builder = VideoDatasetBuilder(video_path, model_path=model_path)
        if not builder.load():
            return {
                "passed": False,
                "score": 0,
                "total": len(actions),
                "errors": ["Cannot load video or YOLO model"],
                "warnings": [],
                "validated_samples": []
            }
        
        try:
            grid = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
            
            for i, action in enumerate(actions):
                if not isinstance(action, dict):
                    errors.append(f"[{i}]: Action không phải dict")
                    continue
                
                time_str = action.get("time", "0:00")
                action_type = action.get("action")
                args = action.get("args", {}) or {}
                
                frame, frame_num, seconds = builder.get_frame_at_time(time_str)
                if frame is None:
                    warnings.append(f"[{i}] time={time_str}: Cannot get frame")
                    continue
                
                game_state = builder.detect_game_state(frame)
                action_error = ActionValidator._validate_action(
                    i, time_str, action_type, args, game_state, grid
                )
                
                if action_error:
                    errors.append(action_error)
                elif action_type == "plant":
                    row, col = int(args.get("row", 0)), int(args.get("col", 0))
                    grid[row][col] = args.get("plant_type")
                
                validated_samples.append({
                    "id": i + 1,
                    "timestamp": time_str,
                    "frame_number": frame_num,
                    "game_state": game_state,
                    "action": {"type": action_type, "args": args},
                    "valid": action_error is None,
                    "error": action_error
                })
        finally:
            builder.close()
        
        total = len(actions)
        score = ((total - len(errors)) / total * 100) if total > 0 else 0
        
        return {
            "passed": score >= 100,
            "score": score,
            "total": total,
            "errors": errors,
            "warnings": warnings,
            "validated_samples": validated_samples
        }
    
    @staticmethod
    def validate_simple(actions: List[Dict[str, Any]]) -> Dict:
        """Validate đơn giản không cần video"""
        errors = []
        warnings = []
        grid = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        
        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                errors.append(f"[{i}]: Action không phải dict")
                continue
            
            action_type = action.get("action")
            time_str = action.get("time", "?")
            args = action.get("args", {}) or {}
            
            error = ActionValidator._validate_action(i, time_str, action_type, args, None, grid)
            if error:
                errors.append(error)
            elif action_type == "plant":
                row, col = int(args.get("row", 0)), int(args.get("col", 0))
                grid[row][col] = args.get("plant_type")
        
        total = len(actions)
        score = ((total - len(errors)) / total * 100) if total > 0 else 0
        
        return {
            "passed": score >= 100,
            "score": score,
            "total": total,
            "errors": errors,
            "warnings": warnings
        }
    
    @staticmethod
    def _validate_action(
        idx: int, 
        time_str: str, 
        action_type: str, 
        args: Dict,
        game_state: Optional[Dict],
        grid: List[List]
    ) -> Optional[str]:
        """Validate single action, return error message or None"""
        if action_type not in VALID_ACTIONS:
            return f"[{idx}] time={time_str}: Invalid action '{action_type}'"
        
        if action_type == "plant":
            plant_type = args.get("plant_type")
            row = args.get("row")
            col = args.get("col")
            
            if not plant_type:
                return f"[{idx}] time={time_str}: plant thiếu plant_type"
            if row is None or col is None:
                return f"[{idx}] time={time_str}: plant thiếu row/col"
            
            try:
                row, col = int(row), int(col)
            except (ValueError, TypeError):
                return f"[{idx}] time={time_str}: row/col phải là số"
            
            if not (0 <= row < GRID_ROWS):
                return f"[{idx}] time={time_str}: row={row} ngoài range 0-4"
            if not (0 <= col < GRID_COLS):
                return f"[{idx}] time={time_str}: col={col} ngoài range 0-8"
            
            if grid[row][col] is not None:
                existing = grid[row][col]
                if not (plant_type == "wall_nut" and existing == "wall_nut"):
                    return f"[{idx}] time={time_str}: Ô ({row},{col}) đã có {existing}"
            
            # Check seed ready nếu có game_state
            if game_state:
                seeds = game_state.get("seeds", [])
                seed_ready = any(
                    s.get("type") == plant_type and s.get("status") == "ready"
                    for s in seeds
                )
                if not seed_ready and seeds:
                    return f"[{idx}] time={time_str}: Seed {plant_type} không ready"
        
        return None
    
    @staticmethod
    def format_result(result: Dict) -> str:
        """Format validation result cho display"""
        lines = [
            f"📊 Score: {result['score']:.1f}% ({result['total']} actions)",
            f"   Errors: {len(result['errors'])}",
            f"   Warnings: {len(result['warnings'])}"
        ]
        
        if result['errors']:
            lines.append("\n❌ Errors:")
            for err in result['errors'][:10]:
                lines.append(f"   {err}")
            if len(result['errors']) > 10:
                lines.append(f"   ... và {len(result['errors']) - 10} lỗi khác")
        
        return "\n".join(lines)

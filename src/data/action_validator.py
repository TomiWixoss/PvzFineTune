# -*- coding: utf-8 -*-
"""
Action Validator - Validate actions bằng cách map với game state từ video
Sử dụng VideoDatasetBuilder để detect game state tại mỗi timestamp
"""

import sys
import os
from typing import Any, Optional
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_actions_with_video(
    actions: list[dict[str, Any]], 
    video_path: str,
    model_path: Optional[str] = None
) -> dict:
    """
    Validate actions bằng cách:
    1. Với mỗi action, lấy frame tại timestamp từ video
    2. YOLO detect → game_state
    3. Check action có hợp lệ với game_state không
    
    Returns:
        {
            "passed": bool,
            "score": float,
            "total": int,
            "errors": list,
            "warnings": list,
            "validated_samples": list  # Samples đã validate với game_state
        }
    """
    from .video_dataset_builder import VideoDatasetBuilder
    
    errors = []
    warnings = []
    validated_samples = []
    
    # Load video và YOLO
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
        # Track game state (grid của plants đã trồng)
        grid = [[None for _ in range(9)] for _ in range(5)]  # 5 rows x 9 cols
        
        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                errors.append(f"[{i}]: Action không phải dict")
                continue
            
            time_str = action.get("time", "0:00")
            action_type = action.get("action")
            args = action.get("args", {}) or {}
            
            # Get frame và detect game_state
            frame, frame_num, seconds = builder.get_frame_at_time(time_str)
            if frame is None:
                warnings.append(f"[{i}] time={time_str}: Cannot get frame")
                continue
            
            game_state = builder.detect_game_state(frame)
            
            # Validate action
            action_error = None
            
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
                        # Check trồng chồng (từ tracking)
                        elif grid[row][col] is not None:
                            existing = grid[row][col]
                            if not (plant_type == "wall_nut" and existing == "wall_nut"):
                                action_error = f"[{i}] time={time_str}: Ô ({row},{col}) đã có {existing}"
                        else:
                            # Check seed packet ready
                            seeds = game_state.get("seeds", [])
                            seed_ready = False
                            seed_status = "not_found"
                            
                            for seed in seeds:
                                if seed.get("type") == plant_type:
                                    seed_status = seed.get("status", "unknown")
                                    if seed_status == "ready":
                                        seed_ready = True
                                        break
                            
                            if not seed_ready and seeds:
                                # Có seeds nhưng không ready
                                action_error = f"[{i}] time={time_str}: Seed {plant_type} đang {seed_status}, không thể trồng"
                            
                            # Check cây có thực sự xuất hiện tại vị trí đó không
                            # Check frame sau 0.5s để chắc chắn cây đã được đặt
                            if action_error is None:
                                check_time = seconds + 0.5
                                check_frame, _, _ = builder.get_frame_at_time(str(check_time))
                                if check_frame is not None:
                                    future_state = builder.detect_game_state(check_frame)
                                    plants_in_frame = future_state.get("plants", [])
                                else:
                                    plants_in_frame = game_state.get("plants", [])
                                
                                plant_found = False
                                for p in plants_in_frame:
                                    if p.get("row") == row and p.get("col") == col:
                                        plant_found = True
                                        break
                                
                                if not plant_found:
                                    action_error = f"[{i}] time={time_str}: YOLO không thấy cây tại ({row},{col}) sau 0.5s - AI có thể ghi thừa action"
                        
                        # Nếu OK, update grid
                        if action_error is None:
                            grid[row][col] = plant_type
            
            elif action_type == "wait":
                # wait luôn OK
                pass
            
            elif action_type is None:
                action_error = f"[{i}] time={time_str}: Thiếu action type"
            
            else:
                action_error = f"[{i}] time={time_str}: Invalid action '{action_type}'"
            
            if action_error:
                errors.append(action_error)
            
            # Build validated sample
            validated_samples.append({
                "id": i + 1,
                "timestamp": time_str,
                "frame_number": frame_num,
                "game_state": game_state,
                "action": {
                    "type": action_type,
                    "args": args
                },
                "valid": action_error is None,
                "error": action_error
            })
        
    finally:
        builder.close()
    
    # Calculate score
    total = len(actions)
    error_count = len(errors)
    passed_count = total - error_count
    score = (passed_count / total * 100) if total > 0 else 0
    
    return {
        "passed": score >= 100,  # Phải 100% vì có auto-fix
        "score": score,
        "total": total,
        "errors": errors,
        "warnings": warnings,
        "validated_samples": validated_samples
    }


def validate_actions_simple(actions: list[dict[str, Any]]) -> dict:
    """
    Validate đơn giản không cần video (chỉ check logic)
    Dùng khi không có video hoặc YOLO model
    """
    VALID_ACTIONS = ["plant", "wait"]
    GRID_ROWS = 5
    GRID_COLS = 9
    
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
            
            try:
                row = int(row)
                col = int(col)
            except (ValueError, TypeError):
                errors.append(f"[{i}] time={time_str}: row/col phải là số")
                continue
            
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
            
            grid[row][col] = plant_type
    
    total = len(actions)
    error_count = len(errors)
    score = ((total - error_count) / total * 100) if total > 0 else 0
    
    return {
        "passed": score >= 100,  # Phải 100% vì có auto-fix
        "score": score,
        "total": total,
        "errors": errors,
        "warnings": warnings
    }


def format_validation_result(result: dict) -> str:
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
    
    if result['warnings']:
        lines.append("\n⚠️ Warnings:")
        for warn in result['warnings'][:5]:
            lines.append(f"   {warn}")
    
    return "\n".join(lines)

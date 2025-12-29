# -*- coding: utf-8 -*-
"""
Action Validator - Validate logic của PvZ actions
Kiểm tra: trồng chồng, row/col range, thiếu fields, cooldown
"""

from typing import Any

# ===========================================
# CONFIG
# ===========================================
VALID_ACTIONS = ["plant", "wait"]
VALID_PLANT_TYPES = [
    "pea_shooter", "sunflower", "wall_nut", "cherry_bomb",
    "snow_pea", "repeater", "potato_mine", "chomper",
    "squash", "threepeater", "jalapeno", "spikeweed",
    "torchwood", "tall_nut", "pumpkin", "magnet_shroom",
    "cabbage_pult", "kernel_pult", "melon_pult", "gatling_pea"
]
GRID_ROWS = 5  # 0-4
GRID_COLS = 9  # 0-8


def parse_time(time_str: str) -> float:
    """Parse 'M:SS' or 'M:SS.S' to seconds"""
    try:
        time_str = str(time_str).strip()
        parts = time_str.replace(".", ":").split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 60 + int(parts[1]) + float(parts[2]) / 10
        return 0
    except:
        return 0


def validate_actions(actions: list[dict[str, Any]]) -> dict:
    """
    Validate logic của actions
    
    Returns:
        {
            "passed": bool,      # True nếu score >= 90%
            "score": float,      # Phần trăm actions hợp lệ
            "total": int,        # Tổng số actions
            "errors": list,      # Danh sách lỗi
            "warnings": list     # Danh sách cảnh báo
        }
    """
    errors = []
    warnings = []
    
    # Track game state
    grid = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    last_plant_time: dict[str, float] = {}  # plant_type -> last_time
    
    for i, action in enumerate(actions):
        if not isinstance(action, dict):
            errors.append(f"[{i}]: Action không phải dict")
            continue
        
        action_type = action.get("action")
        time_str = action.get("time", "?")
        args = action.get("args", {}) or {}
        
        # 1. Check valid action type
        if action_type not in VALID_ACTIONS:
            errors.append(f"[{i}] time={time_str}: Invalid action '{action_type}'")
            continue
        
        if action_type == "plant":
            plant_type = args.get("plant_type")
            row = args.get("row")
            col = args.get("col")
            
            # 2. Check required fields
            if not plant_type:
                errors.append(f"[{i}] time={time_str}: plant thiếu plant_type")
                continue
            if row is None or col is None:
                errors.append(f"[{i}] time={time_str}: plant thiếu row/col")
                continue
            
            # 3. Check valid plant_type
            if plant_type not in VALID_PLANT_TYPES:
                warnings.append(f"[{i}] time={time_str}: plant_type '{plant_type}' không phổ biến")
            
            # 4. Check valid row/col range
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
            
            # 5. Check duplicate position (trồng chồng lên cây đã có)
            if grid[row][col] is not None:
                existing = grid[row][col]
                # Wall-nut có thể trồng đè lên wall-nut cũ (refresh)
                # Pumpkin có thể trồng đè lên cây khác
                if plant_type == "pumpkin" or (plant_type == "wall_nut" and existing == "wall_nut"):
                    pass  # OK
                else:
                    errors.append(
                        f"[{i}] time={time_str}: Ô ({row},{col}) đã có {existing}, "
                        f"không thể trồng {plant_type}"
                    )
                    continue
            
            # 6. Check cooldown (same plant within 3s is suspicious)
            time_seconds = parse_time(time_str)
            if plant_type in last_plant_time:
                diff = time_seconds - last_plant_time[plant_type]
                if 0 < diff < 3:
                    warnings.append(
                        f"[{i}] time={time_str}: {plant_type} trồng quá nhanh ({diff:.1f}s)"
                    )
            
            # Update state
            grid[row][col] = plant_type
            last_plant_time[plant_type] = time_seconds
    
    # Calculate score
    total = len(actions)
    error_count = len(errors)
    passed_count = total - error_count
    score = (passed_count / total * 100) if total > 0 else 0
    
    return {
        "passed": score >= 90,
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

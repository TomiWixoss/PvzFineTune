# -*- coding: utf-8 -*-
"""
Action Validator - Validate actions với game state từ video
"""

import re
from typing import Any, Dict, List, Optional

from ..core.constants import VALID_ACTIONS, GRID_ROWS, GRID_COLS


class ActionValidator:
    """Validate actions"""
    
    @staticmethod
    def parse_game_state_text(game_state_text: str) -> Dict[str, Any]:
        """
        Parse game_state string thành dict
        Format: "PLANTS:[(type,row,col),...]. ZOMBIES:[(type,row,col),...]. SEEDS:[(type,status),...]"
        """
        result = {"plants": [], "zombies": [], "seeds": []}
        
        if not game_state_text:
            return result
        
        # Parse PLANTS
        plants_match = re.search(r'PLANTS:\[(.*?)\]', game_state_text)
        if plants_match:
            plants_str = plants_match.group(1)
            if plants_str:
                plant_pattern = re.findall(r'\(([^,]+),(\d+),(\d+)\)', plants_str)
                for plant_type, row, col in plant_pattern:
                    result["plants"].append({
                        "type": plant_type,
                        "row": int(row),
                        "col": int(col)
                    })
        
        # Parse ZOMBIES
        zombies_match = re.search(r'ZOMBIES:\[(.*?)\]', game_state_text)
        if zombies_match:
            zombies_str = zombies_match.group(1)
            if zombies_str:
                zombie_pattern = re.findall(r'\(([^,]+),(\d+),(\d+)\)', zombies_str)
                for zombie_type, row, col in zombie_pattern:
                    result["zombies"].append({
                        "type": zombie_type,
                        "row": int(row),
                        "col": int(col)
                    })
        
        # Parse SEEDS
        seeds_match = re.search(r'SEEDS:\[(.*?)\]', game_state_text)
        if seeds_match:
            seeds_str = seeds_match.group(1)
            if seeds_str:
                seed_pattern = re.findall(r'\(([^,]+),([^)]+)\)', seeds_str)
                for seed_type, status in seed_pattern:
                    result["seeds"].append({
                        "type": seed_type,
                        "status": status
                    })
        
        return result
    
    @staticmethod
    def build_grid_from_game_state(game_state: Dict[str, Any]) -> List[List]:
        """Build grid từ game_state (parsed dict hoặc có plants list)"""
        grid = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        
        plants = game_state.get("plants", [])
        for plant in plants:
            row = plant.get("row", 0)
            col = plant.get("col", 0)
            plant_type = plant.get("type", "unknown")
            if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
                grid[row][col] = plant_type
        
        return grid
    
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
                # Build grid từ game_state detect được
                grid = ActionValidator.build_grid_from_game_state(game_state)
                
                action_error = ActionValidator._validate_action(
                    i, time_str, action_type, args, game_state, grid
                )
                
                if action_error:
                    errors.append(action_error)
                
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
        """Validate đơn giản không cần video - dùng cho sequential actions"""
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
    def validate_training_data(samples: List[Dict[str, Any]]) -> Dict:
        """
        Validate training data format (game_state + action)
        Mỗi sample độc lập - parse game_state để check action hợp lệ
        """
        errors = []
        warnings = []
        valid_samples = []
        invalid_samples = []
        
        for i, sample in enumerate(samples):
            if not isinstance(sample, dict):
                errors.append(f"[{i}]: Sample không phải dict")
                invalid_samples.append(sample)
                continue
            
            game_state_text = sample.get("game_state", "")
            action_type = sample.get("action")
            args = sample.get("arguments", {}) or {}
            
            # Parse game_state text thành dict
            parsed_state = ActionValidator.parse_game_state_text(game_state_text)
            
            # Build grid từ parsed state
            grid = ActionValidator.build_grid_from_game_state(parsed_state)
            
            # Validate action với grid từ game_state
            error = ActionValidator._validate_action(
                i, f"sample_{i}", action_type, args, parsed_state, grid
            )
            
            if error:
                errors.append(error)
                invalid_samples.append(sample)
            else:
                valid_samples.append(sample)
        
        total = len(samples)
        score = ((total - len(errors)) / total * 100) if total > 0 else 0
        
        return {
            "passed": score >= 100,
            "score": score,
            "total": total,
            "valid_count": len(valid_samples),
            "invalid_count": len(invalid_samples),
            "errors": errors,
            "warnings": warnings,
            "valid_samples": valid_samples,
            "invalid_samples": invalid_samples
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
            
            # Check plant chồng từ grid (đã build từ game_state)
            if grid[row][col] is not None:
                existing = grid[row][col]
                # Cho phép wall_nut chồng lên wall_nut (repair)
                if not (plant_type == "wall_nut" and existing == "wall_nut"):
                    return f"[{idx}] time={time_str}: Ô ({row},{col}) đã có {existing}"
            
            # Check seed ready nếu có game_state
            if game_state:
                seeds = game_state.get("seeds", [])
                if seeds:
                    seed_ready = any(
                        s.get("type") == plant_type and s.get("status") == "ready"
                        for s in seeds
                    )
                    if not seed_ready:
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

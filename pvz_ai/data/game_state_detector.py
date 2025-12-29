# -*- coding: utf-8 -*-
"""
Game State Detector - Detect game state từ frame bằng YOLO
"""

from typing import Dict, Any, List

from ..utils.grid_utils import get_row, get_col


class GameStateDetector:
    """Detect game state từ frame"""
    
    def __init__(self, detector, grid_rows_y: List[int], grid_cols_x: List[int]):
        self.detector = detector
        self.grid_rows_y = grid_rows_y
        self.grid_cols_x = grid_cols_x
    
    def detect(self, frame, conf: float = None) -> Dict[str, Any]:
        """
        YOLO detect frame → game_state
        
        Returns:
            {
                "text": "PLANTS:[...]. ZOMBIES:[...]. SEEDS:[...]",
                "plants": [...],
                "zombies": [...],
                "seeds": [...]
            }
        """
        grouped = self.detector.detect_grouped(frame, conf)
        
        zombies = grouped.get("zombie", [])
        plants = grouped.get("pea_shooter", [])
        seed_ready = grouped.get("pea_shooter_ready", [])
        seed_cooldown = grouped.get("pea_shooter_cooldown", [])
        
        # Add row/col info
        for z in zombies:
            z["row"] = get_row(z["y"], self.grid_rows_y)
            z["col"] = get_col(z["x"], self.grid_cols_x)
            z["type"] = "zombie"
        
        for p in plants:
            p["row"] = get_row(p["y"], self.grid_rows_y)
            p["col"] = get_col(p["x"], self.grid_cols_x)
            p["type"] = "pea_shooter"
        
        # Build seeds list
        seeds = []
        for s in seed_ready:
            seeds.append({"type": "pea_shooter", "status": "ready", "x": s["x"], "y": s["y"], "conf": s["conf"]})
        for s in seed_cooldown:
            seeds.append({"type": "pea_shooter", "status": "cooldown", "x": s["x"], "y": s["y"], "conf": s["conf"]})
        
        text = self._build_text(plants, zombies, seeds)
        
        return {
            "text": text,
            "plants": plants,
            "zombies": zombies,
            "seeds": seeds
        }
    
    def _build_text(self, plants: list, zombies: list, seeds: list) -> str:
        """Build text representation cho Gemma"""
        parts = []
        
        if plants:
            plant_str = ",".join([f"({p.get('type','plant')},{p.get('row',0)},{p.get('col',0)})" for p in plants])
            parts.append(f"PLANTS:[{plant_str}]")
        else:
            parts.append("PLANTS:[]")
        
        if zombies:
            zombie_str = ",".join([f"({z.get('type','zombie')},{z.get('row',0)},{z.get('col',8)})" for z in zombies])
            parts.append(f"ZOMBIES:[{zombie_str}]")
        else:
            parts.append("ZOMBIES:[]")
        
        if seeds:
            seed_str = ",".join([f"({s.get('type','unknown')},{s.get('status','unknown')})" for s in seeds])
            parts.append(f"SEEDS:[{seed_str}]")
        else:
            parts.append("SEEDS:[]")
        
        return ". ".join(parts)

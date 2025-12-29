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
        Tự động detect tất cả classes từ model
        """
        grouped = self.detector.detect_grouped(frame, conf)
        
        zombies = []
        plants = []
        seeds = []
        
        # Auto-categorize based on class name patterns
        for class_name, detections in grouped.items():
            if class_name == "zombie":
                for z in detections:
                    z["type"] = "zombie"
                    z["row"] = get_row(z["y"], self.grid_rows_y)
                    z["col"] = get_col(z["x"], self.grid_cols_x)
                    zombies.append(z)
            elif "_ready" in class_name:
                plant_type = class_name.replace("_ready", "")
                for s in detections:
                    seeds.append({"type": plant_type, "status": "ready", "x": s["x"], "y": s["y"], "conf": s["conf"]})
            elif "_cooldown" in class_name:
                plant_type = class_name.replace("_cooldown", "")
                for s in detections:
                    seeds.append({"type": plant_type, "status": "cooldown", "x": s["x"], "y": s["y"], "conf": s["conf"]})
            elif "_reward" in class_name:
                plant_type = class_name.replace("_reward", "")
                for s in detections:
                    seeds.append({"type": plant_type, "status": "ready", "x": s["x"], "y": s["y"], "conf": s["conf"]})
            elif class_name not in ["sun"]:
                # Assume it's a plant on field
                for p in detections:
                    p["type"] = class_name
                    p["row"] = get_row(p["y"], self.grid_rows_y)
                    p["col"] = get_col(p["x"], self.grid_cols_x)
                    plants.append(p)
        
        text = self._build_text(plants, zombies, seeds)
        
        return {
            "text": text,
            "plants": plants,
            "zombies": zombies,
            "seeds": seeds,
            "suns": grouped.get("sun", [])
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

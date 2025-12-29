# -*- coding: utf-8 -*-
"""
Training Data Builder - Build training data từ validated actions
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List


class TrainingBuilder:
    """Build training data từ actions đã validate"""
    
    @staticmethod
    def build(video_path: str, actions: List, output_dir: Path, training_path: str) -> Optional[str]:
        """
        Build training data từ actions
        
        Args:
            video_path: Path to video
            actions: List of validated actions
            output_dir: Output directory
            training_path: Path to save training data
        
        Returns:
            Path to training data or None if failed
        """
        try:
            from ..data.video_dataset_builder import VideoDatasetBuilder
            from ..data.dataset_converter import convert_dataset
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            actions_file = output_dir / f"actions_temp_{timestamp}.json"
            
            # Convert format
            builder_actions = []
            for action in actions:
                builder_actions.append({
                    "time": action.get("time", "0:00"),
                    "action": action.get("action", "wait"),
                    "args": action.get("args", {})
                })
            
            # Save temp actions
            with open(actions_file, "w", encoding="utf-8") as f:
                json.dump(builder_actions, f, indent=2, ensure_ascii=False)
            
            dataset_path = output_dir / f"dataset_temp_{timestamp}.json"
            
            # Build dataset
            builder = VideoDatasetBuilder(video_path)
            if builder.load():
                builder.process_actions_file(str(actions_file), str(dataset_path), save_frames=False)
                builder.close()
                
                # Convert to training format
                convert_dataset(str(dataset_path), training_path)
                
                # Cleanup temp files
                actions_file.unlink()
                dataset_path.unlink()
                
                print(f"✅ Training data: {training_path}")
                return training_path
            else:
                print("❌ Cannot load video for training data")
                return None
                
        except Exception as e:
            print(f"❌ Error building training data: {e}")
            return None

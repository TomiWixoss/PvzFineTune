# -*- coding: utf-8 -*-
"""
Video Dataset Builder - Tạo dataset training từ video gameplay
"""

import cv2
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from ..core.config import Config
from ..core.constants import (
    GRID_ROWS_Y_800, GRID_COLUMNS_X_800,
    GRID_ROWS_Y_1080, GRID_COLUMNS_X_1080,
    DEFAULT_CONFIDENCE
)
from ..utils.time_utils import parse_time, format_time
from .game_state_detector import GameStateDetector


class VideoDatasetBuilder:
    """Tạo dataset từ video gameplay"""
    
    def __init__(self, video_path: str, model_path: str = None, conf: float = None):
        self.video_path = Path(video_path)
        self.model_path = model_path or str(Config.YOLO_MODEL_PATH)
        self.conf = conf or DEFAULT_CONFIDENCE
        
        self.detector = None
        self.state_detector = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        self.fps = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.duration = 0.0
        
        self.grid_rows_y = GRID_ROWS_Y_800
        self.grid_cols_x = GRID_COLUMNS_X_800
        
        self.samples: List[Dict] = []
        self.output_dir: Optional[Path] = None
        self.frames_dir: Optional[Path] = None
    
    def load(self) -> bool:
        """Load video và YOLO model"""
        if not self.video_path.exists():
            print(f"✗ Video not found: {self.video_path}")
            return False
        
        from ..inference.yolo_detector import YOLODetector
        self.detector = YOLODetector(self.model_path)
        if not self.detector.load():
            return False
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"✗ Cannot open video: {self.video_path}")
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        # Auto-detect grid
        if self.height >= 1080:
            self.grid_rows_y = GRID_ROWS_Y_1080
            self.grid_cols_x = GRID_COLUMNS_X_1080
            print("  Grid: 1080p mode")
        else:
            print("  Grid: 800x600 mode")
        
        # Init state detector
        self.state_detector = GameStateDetector(self.detector, self.grid_rows_y, self.grid_cols_x)
        
        print(f"✓ Video: {self.video_path.name}")
        print(f"  {self.width}x{self.height} | {self.fps:.1f} FPS | {format_time(self.duration)}")
        return True
    
    def setup_output(self, output_path: str):
        """Setup output directories"""
        output_path = Path(output_path)
        self.output_dir = output_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
    
    def get_frame_at_time(self, time_str: str) -> Tuple[Optional[Any], int, float]:
        """Lấy frame tại timestamp"""
        seconds = parse_time(time_str)
        frame_num = int(seconds * self.fps)
        
        if frame_num < 0 or frame_num >= self.total_frames:
            return None, -1, seconds
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        return (frame, frame_num, seconds) if ret else (None, frame_num, seconds)
    
    def detect_game_state(self, frame) -> Dict[str, Any]:
        """YOLO detect frame → game_state"""
        return self.state_detector.detect(frame, self.conf)
    
    def process_actions_file(self, actions_path: str, output_path: str, save_frames: bool = True):
        """Đọc file actions → tạo dataset"""
        with open(actions_path, 'r', encoding='utf-8') as f:
            actions = json.load(f)
        
        print(f"\n✓ Loaded {len(actions)} actions")
        self.setup_output(output_path)
        self.samples = []
        
        for i, action_data in enumerate(actions):
            time_str = action_data.get("time", "0:00")
            action_type = action_data.get("action", "wait")
            action_args = action_data.get("args", {})
            
            print(f"[{i+1}/{len(actions)}] {time_str} → {action_type}")
            
            frame, frame_num, seconds = self.get_frame_at_time(time_str)
            if frame is None:
                continue
            
            game_state = self.detect_game_state(frame)
            
            frame_filename = None
            if save_frames:
                frame_filename = f"{format_time(seconds).replace(':', '_').replace('.', '_')}.png"
                cv2.imwrite(str(self.frames_dir / frame_filename), frame)
            
            self.samples.append({
                "id": i + 1,
                "timestamp": format_time(seconds),
                "frame_number": frame_num,
                "game_state": game_state,
                "action": {"type": action_type, "args": action_args},
                "frame_path": f"frames/{frame_filename}" if frame_filename else None
            })
        
        self._save_dataset(output_path)
    
    def _save_dataset(self, output_path: str):
        """Save dataset to JSON"""
        dataset = {
            "video": self.video_path.name,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_info": {
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "duration": format_time(self.duration),
                "total_frames": self.total_frames
            },
            "total_samples": len(self.samples),
            "samples": self.samples
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved {len(self.samples)} samples to {output_path}")
    
    def close(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build dataset from video')
    parser.add_argument('video', help='Input video path')
    parser.add_argument('-a', '--actions', help='Actions JSON file')
    parser.add_argument('-o', '--output', default='data/processed/dataset.json')
    parser.add_argument('--no-frames', action='store_true')
    args = parser.parse_args()
    
    builder = VideoDatasetBuilder(args.video)
    if not builder.load():
        return
    
    try:
        if args.actions:
            builder.process_actions_file(args.actions, args.output, save_frames=not args.no_frames)
        else:
            print("✗ Specify --actions")
    finally:
        builder.close()


if __name__ == "__main__":
    main()

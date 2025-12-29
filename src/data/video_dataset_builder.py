# -*- coding: utf-8 -*-
"""
Video Dataset Builder - Tạo dataset training cho Gemma từ video gameplay

WORKFLOW:
=========
1. Input: File actions (timestamp + action) - do người xem video ghi lại hoặc AI cao hơn quyết định
2. Script: Đọc frame tại timestamp → YOLO detect → tạo game_state (map data)
3. Output: Dataset {game_state, action} để train Gemma

INPUT FORMAT (actions.json):
============================
[
    {"time": "0:15", "action": "collect_sun", "args": {"x": 300, "y": 150}},
    {"time": "0:18", "action": "plant_pea_shooter", "args": {"row": 2, "col": 0}},
    {"time": "0:25", "action": "collect_sun"},
    {"time": "0:30", "action": "do_nothing"}
]

OUTPUT FORMAT (dataset.json):
=============================
{
    "video": "gameplay.mp4",
    "created": "2024-01-15 10:30:00",
    "samples": [
        {
            "id": 1,
            "timestamp": "00:15.000",
            "frame_number": 450,
            "game_state": {
                "text": "HAS_SUN x=305 y=148. NO_ZOMBIE. CAN_PLANT",
                "suns": [{"x": 305, "y": 148, "conf": 0.95}],
                "zombies": [],
                "plants": [{"x": 100, "y": 260, "conf": 0.92}],
                "can_plant": true
            },
            "action": {
                "type": "collect_sun",
                "args": {"x": 300, "y": 150}
            },
            "frame_path": "frames/00_15_000.png"
        }
    ]
}

USAGE:
======
# Từ file actions
python -m src.data.video_dataset_builder video.mp4 -a actions.json -o dataset.json

# Interactive mode - xem video và nhập action realtime
python -m src.data.video_dataset_builder video.mp4 -o dataset.json --interactive

# Chỉ extract game_state tại các timestamps (không cần action)
python -m src.data.video_dataset_builder video.mp4 --timestamps "0:15,0:30,1:00" -o states.json
"""

import cv2
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    YOLO_MODEL_PATH, DEFAULT_CONFIDENCE,
    GRID_ROWS_Y_800, GRID_COLUMNS_X_800,
    GRID_ROWS_Y_1080, GRID_COLUMNS_X_1080
)


class VideoDatasetBuilder:
    """
    Tạo dataset từ video gameplay
    
    Input: actions file (timestamp + action)
    Output: dataset với game_state (từ YOLO detection) + action
    """
    
    ACTION_TYPES = ["plant", "wait"]
    
    def __init__(self, video_path: str, model_path: str = None, conf: float = None):
        self.video_path = Path(video_path)
        self.model_path = model_path or YOLO_MODEL_PATH
        self.conf = conf or DEFAULT_CONFIDENCE
        
        self.detector = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Video info
        self.fps = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.duration = 0.0
        
        # Grid config (auto-detect based on resolution)
        self.grid_rows_y = GRID_ROWS_Y_800
        self.grid_cols_x = GRID_COLUMNS_X_800
        
        # Output
        self.samples: List[Dict] = []
        self.output_dir: Optional[Path] = None
        self.frames_dir: Optional[Path] = None
    
    def load(self) -> bool:
        """Load video và YOLO model"""
        if not self.video_path.exists():
            print(f"✗ Video not found: {self.video_path}")
            return False
        
        # Load YOLO
        from inference.yolo_detector import YOLODetector
        self.detector = YOLODetector(self.model_path)
        if not self.detector.load():
            return False
        
        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"✗ Cannot open video: {self.video_path}")
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        # Auto-detect grid based on resolution
        if self.height >= 1080:
            self.grid_rows_y = GRID_ROWS_Y_1080
            self.grid_cols_x = GRID_COLUMNS_X_1080
            print(f"  Grid: 1080p mode")
        else:
            self.grid_rows_y = GRID_ROWS_Y_800
            self.grid_cols_x = GRID_COLUMNS_X_800
            print(f"  Grid: 800x600 mode")
        
        print(f"✓ Video: {self.video_path.name}")
        print(f"  {self.width}x{self.height} | {self.fps:.1f} FPS | {self.total_frames} frames")
        print(f"  Duration: {self._format_time(self.duration)}")
        return True
    
    def setup_output(self, output_path: str):
        """Setup output directories"""
        output_path = Path(output_path)
        self.output_dir = output_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        print(f"✓ Output: {output_path}")
        print(f"  Frames: {self.frames_dir}")
    
    # =========================================
    # TIME UTILITIES
    # =========================================
    
    def _format_time(self, seconds: float) -> str:
        """Seconds → MM:SS.mmm"""
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"
    
    def _parse_time(self, time_str: str) -> float:
        """Parse time string → seconds. Supports: "1:30", "01:30.500", "90" """
        time_str = time_str.strip()
        
        # Pure seconds
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
    
    def _time_to_frame(self, seconds: float) -> int:
        return int(seconds * self.fps)
    
    def _frame_to_time(self, frame_num: int) -> float:
        return frame_num / self.fps if self.fps > 0 else 0
    
    # =========================================
    # FRAME & DETECTION
    # =========================================
    
    def get_frame_at_time(self, time_str: str) -> Tuple[Optional[Any], int, float]:
        """
        Lấy frame tại timestamp
        Returns: (frame, frame_number, seconds)
        """
        seconds = self._parse_time(time_str)
        frame_num = self._time_to_frame(seconds)
        
        if frame_num < 0 or frame_num >= self.total_frames:
            print(f"✗ Invalid time: {time_str} (max: {self._format_time(self.duration)})")
            return None, -1, seconds
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        if not ret:
            return None, frame_num, seconds
        return frame, frame_num, seconds
    
    def detect_game_state(self, frame) -> Dict[str, Any]:
        """
        YOLO detect frame → game_state (map data)
        
        Returns:
            {
                "text": "PLANTS: [(pea_shooter,2,0),(pea_shooter,2,1)]. ZOMBIES: 3 at rows [1,2,2]. CAN_PLANT",
                "plants": [{"type": "pea_shooter", "row": 2, "col": 0, "x": 100, "y": 260, "conf": 0.92}, ...],
                "zombies": [{"row": 2, "col": 7, "x": 600, "y": 260, "conf": 0.88}, ...],
                "zombie_count": 3,
                "seed_ready": [...],
                "can_plant": True
            }
        """
        grouped = self.detector.detect_grouped(frame, self.conf)
        
        zombies = grouped.get("zombie", [])
        plants = grouped.get("pea_shooter", [])
        seed_ready = grouped.get("pea_shooter_ready", [])
        
        # Add row/col/type info to zombies
        for z in zombies:
            z["row"] = self._get_row(z["y"])
            z["col"] = self._get_col(z["x"])
            z["type"] = "zombie"  # TODO: detect zombie type from YOLO (zombie, cone_zombie, bucket_zombie, ...)
        
        # Add row/col/type info to plants
        for p in plants:
            p["row"] = self._get_row(p["y"])
            p["col"] = self._get_col(p["x"])
            p["type"] = "pea_shooter"  # TODO: detect plant type from YOLO
        
        can_plant = len(seed_ready) > 0
        
        # Build text representation (for Gemma training)
        text = self._build_state_text(plants, zombies, can_plant)
        
        return {
            "text": text,
            "plants": plants,
            "zombies": zombies,
            "zombie_count": len(zombies),
            "seed_ready": seed_ready,
            "can_plant": can_plant
        }
    
    def _get_row(self, y: int) -> int:
        """Y coordinate → row index (0-4)"""
        min_dist = float('inf')
        row = 0
        for i, row_y in enumerate(self.grid_rows_y):
            dist = abs(y - row_y)
            if dist < min_dist:
                min_dist = dist
                row = i
        return row
    
    def _get_col(self, x: int) -> int:
        """X coordinate → col index (0-8)"""
        min_dist = float('inf')
        col = 0
        for i, col_x in enumerate(self.grid_cols_x):
            dist = abs(x - col_x)
            if dist < min_dist:
                min_dist = dist
                col = i
        return col
    
    def _build_state_text(self, plants: list, zombies: list, can_plant: bool) -> str:
        """Build text representation cho Gemma - bao gồm plants đã trồng và zombies"""
        parts = []
        
        # Plants info: (type,row,col) cho mỗi plant
        if plants:
            plant_positions = [(p.get("type", "plant"), p.get("row", 0), p.get("col", 0)) for p in plants]
            plant_str = ",".join([f"({t},{r},{c})" for t, r, c in plant_positions])
            parts.append(f"PLANTS:[{plant_str}]")
        else:
            parts.append("PLANTS:[]")
        
        # Zombies info: (type,row,col) cho mỗi zombie - giống format plant
        if zombies:
            zombie_positions = [(z.get("type", "zombie"), z.get("row", 0), z.get("col", 8)) for z in zombies]
            zombie_str = ",".join([f"({t},{r},{c})" for t, r, c in zombie_positions])
            parts.append(f"ZOMBIES:[{zombie_str}]")
        else:
            parts.append("ZOMBIES:[]")
        
        # Plant ability
        parts.append("CAN_PLANT" if can_plant else "CANNOT_PLANT")
        
        return ". ".join(parts)
    
    # =========================================
    # DATASET BUILDING
    # =========================================
    
    def process_actions_file(self, actions_path: str, output_path: str, save_frames: bool = True):
        """
        Main function: Đọc file actions → tạo dataset
        
        Args:
            actions_path: Path to actions.json
            output_path: Path to output dataset.json
            save_frames: Save frame images
        """
        # Load actions
        with open(actions_path, 'r', encoding='utf-8') as f:
            actions = json.load(f)
        
        print(f"\n✓ Loaded {len(actions)} actions from {actions_path}")
        print("=" * 60)
        
        self.setup_output(output_path)
        self.samples = []
        
        for i, action_data in enumerate(actions):
            time_str = action_data.get("time", "0:00")
            action_type = action_data.get("action", "do_nothing")
            action_args = action_data.get("args", {})
            
            print(f"\n[{i+1}/{len(actions)}] {time_str} → {action_type}")
            
            # Get frame
            frame, frame_num, seconds = self.get_frame_at_time(time_str)
            if frame is None:
                print(f"  ✗ Cannot get frame")
                continue
            
            # Detect → game_state
            game_state = self.detect_game_state(frame)
            print(f"  State: {game_state['text']}")
            print(f"  Plants: {len(game_state['plants'])} | Zombies: {game_state['zombie_count']}")
            
            # Save frame
            frame_filename = None
            if save_frames:
                frame_filename = f"{self._format_time(seconds).replace(':', '_').replace('.', '_')}.png"
                frame_path = self.frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
            
            # Build sample
            sample = {
                "id": i + 1,
                "timestamp": self._format_time(seconds),
                "frame_number": frame_num,
                "game_state": game_state,
                "action": {
                    "type": action_type,
                    "args": action_args
                },
                "frame_path": f"frames/{frame_filename}" if frame_filename else None
            }
            self.samples.append(sample)
        
        # Save dataset
        self._save_dataset(output_path)
    
    def process_timestamps(self, timestamps: List[str], output_path: str, save_frames: bool = True):
        """
        Extract game_state tại các timestamps (không có action)
        Useful để xem game state tại các thời điểm cụ thể
        """
        print(f"\n✓ Processing {len(timestamps)} timestamps")
        print("=" * 60)
        
        self.setup_output(output_path)
        self.samples = []
        
        for i, time_str in enumerate(timestamps):
            print(f"\n[{i+1}/{len(timestamps)}] {time_str}")
            
            frame, frame_num, seconds = self.get_frame_at_time(time_str)
            if frame is None:
                continue
            
            game_state = self.detect_game_state(frame)
            print(f"  State: {game_state['text']}")
            
            frame_filename = None
            if save_frames:
                frame_filename = f"{self._format_time(seconds).replace(':', '_').replace('.', '_')}.png"
                cv2.imwrite(str(self.frames_dir / frame_filename), frame)
            
            sample = {
                "id": i + 1,
                "timestamp": self._format_time(seconds),
                "frame_number": frame_num,
                "game_state": game_state,
                "action": None,
                "frame_path": f"frames/{frame_filename}" if frame_filename else None
            }
            self.samples.append(sample)
        
        self._save_dataset(output_path)
    
    def interactive_mode(self, output_path: str):
        """
        Interactive mode: Xem video, pause, nhập action, tạo sample
        
        Controls:
            SPACE: Pause/Resume
            A: Add action at current frame
            S: Skip 5 seconds
            D: Back 5 seconds
            Q: Quit and save
        """
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)
        print("Controls:")
        print("  SPACE : Pause/Resume")
        print("  A     : Add action at current frame")
        print("  S     : Skip 5 seconds")
        print("  D     : Back 5 seconds")
        print("  Q     : Quit and save")
        print("=" * 60)
        
        self.setup_output(output_path)
        self.samples = []
        
        paused = False
        frame_num = 0
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = self.cap.read()
                if not ret:
                    break
            
            # Draw info
            display = frame.copy()
            seconds = self._frame_to_time(frame_num)
            info = f"Time: {self._format_time(seconds)} | Frame: {frame_num}/{self.total_frames}"
            if paused:
                info += " | PAUSED"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Samples: {len(self.samples)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Video Dataset Builder", display)
            
            key = cv2.waitKey(30 if not paused else 0) & 0xFF
            
            if key == ord(' '):
                paused = not paused
            
            elif key == ord('a') or key == ord('A'):
                # Add action
                paused = True
                print(f"\n--- Add action at {self._format_time(seconds)} ---")
                print("Action types: 1=plant, 2=wait")
                
                try:
                    choice = input("Choose action (1/2): ").strip()
                    action_type = {
                        "1": "plant",
                        "2": "wait"
                    }.get(choice, "wait")
                    
                    args = {}
                    if action_type == "plant":
                        plant_type = input("  plant_type (pea_shooter/sunflower/wall_nut): ").strip() or "pea_shooter"
                        row = input("  row (0-4): ").strip()
                        col = input("  col (0-8): ").strip()
                        args["plant_type"] = plant_type
                        if row: args["row"] = int(row)
                        if col: args["col"] = int(col)
                    
                    # Detect game state
                    game_state = self.detect_game_state(frame)
                    
                    # Save frame
                    frame_filename = f"{self._format_time(seconds).replace(':', '_').replace('.', '_')}.png"
                    cv2.imwrite(str(self.frames_dir / frame_filename), frame)
                    
                    sample = {
                        "id": len(self.samples) + 1,
                        "timestamp": self._format_time(seconds),
                        "frame_number": frame_num,
                        "game_state": game_state,
                        "action": {"type": action_type, "args": args},
                        "frame_path": f"frames/{frame_filename}"
                    }
                    self.samples.append(sample)
                    print(f"✓ Added: {action_type} | State: {game_state['text']}")
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
            
            elif key == ord('s') or key == ord('S'):
                frame_num = min(frame_num + int(5 * self.fps), self.total_frames - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
            elif key == ord('d') or key == ord('D'):
                frame_num = max(frame_num - int(5 * self.fps), 0)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
            elif key == ord('q') or key == ord('Q'):
                break
        
        cv2.destroyAllWindows()
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
                "duration": self._format_time(self.duration),
                "total_frames": self.total_frames
            },
            "total_samples": len(self.samples),
            "samples": self.samples
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n" + "=" * 60)
        print(f"✓ Saved {len(self.samples)} samples to {output_path}")
        
        # Stats
        if self.samples:
            actions = {}
            for s in self.samples:
                if s.get("action"):
                    t = s["action"]["type"]
                    actions[t] = actions.get(t, 0) + 1
            print(f"  Actions: {actions}")
    
    def close(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Build game state + action dataset from video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Từ file actions
  python -m src.data.video_dataset_builder video.mp4 -a actions.json -o dataset.json
  
  # Interactive mode
  python -m src.data.video_dataset_builder video.mp4 -o dataset.json --interactive
  
  # Extract states tại timestamps
  python -m src.data.video_dataset_builder video.mp4 --timestamps "0:15,0:30,1:00" -o states.json
        """
    )
    
    parser.add_argument('video', help='Input video path')
    parser.add_argument('-a', '--actions', help='Actions JSON file (timestamp + action)')
    parser.add_argument('-o', '--output', default='data/processed/dataset.json', help='Output dataset path')
    parser.add_argument('-m', '--model', help='YOLO model path')
    parser.add_argument('-c', '--conf', type=float, help='Confidence threshold')
    parser.add_argument('--timestamps', help='Comma-separated timestamps to extract (e.g., "0:15,0:30,1:00")')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--no-frames', action='store_true', help='Do not save frame images')
    
    args = parser.parse_args()
    
    builder = VideoDatasetBuilder(
        video_path=args.video,
        model_path=args.model,
        conf=args.conf
    )
    
    if not builder.load():
        return
    
    try:
        if args.interactive:
            builder.interactive_mode(args.output)
        elif args.actions:
            builder.process_actions_file(args.actions, args.output, save_frames=not args.no_frames)
        elif args.timestamps:
            timestamps = [t.strip() for t in args.timestamps.split(',')]
            builder.process_timestamps(timestamps, args.output, save_frames=not args.no_frames)
        else:
            print("✗ Specify --actions, --timestamps, or --interactive")
            parser.print_help()
    finally:
        builder.close()


if __name__ == "__main__":
    main()

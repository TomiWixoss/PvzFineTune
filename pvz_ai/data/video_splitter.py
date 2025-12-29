# -*- coding: utf-8 -*-
"""
Video Splitter - Tách video dài thành các đoạn ngắn
"""

import os
import subprocess
from pathlib import Path
from typing import List

from ..utils.time_utils import parse_time, format_time


def get_video_duration(video_path: str) -> float:
    """Lấy duration của video (seconds)"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def split_video(
    video_path: str,
    max_duration: int = 60,
    output_dir: str = None
) -> List[str]:
    """
    Tách video thành các đoạn max_duration giây
    
    Args:
        video_path: Path to video
        max_duration: Max duration per segment (seconds)
        output_dir: Output directory
    
    Returns:
        List of output video paths
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"✗ Video not found: {video_path}")
        return []
    
    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get duration
    duration = get_video_duration(str(video_path))
    print(f"Video: {video_path.name}")
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
    
    if duration <= max_duration:
        print(f"  ✓ Video ngắn hơn {max_duration}s, không cần tách")
        return [str(video_path)]
    
    # Calculate segments
    num_segments = int(duration // max_duration) + (1 if duration % max_duration > 0 else 0)
    print(f"  Tách thành {num_segments} đoạn (max {max_duration}s mỗi đoạn)")
    
    output_files = []
    for i in range(num_segments):
        start = i * max_duration
        segment_duration = min(max_duration, duration - start)
        
        output_name = f"{video_path.stem}_part{i+1}{video_path.suffix}"
        output_path = output_dir / output_name
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(segment_duration),
            "-c", "copy",
            str(output_path)
        ]
        
        print(f"  [{i+1}/{num_segments}] {format_time(start)} - {format_time(start + segment_duration)}")
        subprocess.run(cmd, capture_output=True)
        output_files.append(str(output_path))
    
    print(f"✓ Tách xong {len(output_files)} files")
    return output_files


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Split video into segments')
    parser.add_argument('video', help='Input video path')
    parser.add_argument('-d', '--duration', type=int, default=60, help='Max duration per segment (seconds)')
    parser.add_argument('-o', '--output', help='Output directory')
    args = parser.parse_args()
    
    split_video(args.video, args.duration, args.output)


if __name__ == "__main__":
    main()

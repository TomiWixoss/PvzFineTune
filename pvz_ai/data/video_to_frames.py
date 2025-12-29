# -*- coding: utf-8 -*-
"""
Video to Frames - Tách frames từ video
"""

import os
import cv2
from pathlib import Path
from typing import List

from ..utils.time_utils import parse_time


def extract_frames(
    video_path: str, 
    output_dir: str = None,
    fps: float = 1.0, 
    prefix: str = None,
    start_time: str = None, 
    end_time: str = None
) -> int:
    """
    Extract frames from video at specified FPS
    
    Returns:
        Number of frames extracted
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"✗ Video not found: {video_path}")
        return 0
    
    if output_dir is None:
        output_dir = f"data/raw/frames/{video_path.stem}"
    
    os.makedirs(output_dir, exist_ok=True)
    prefix = prefix or video_path.stem
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    start_sec = parse_time(start_time) if start_time else 0
    end_sec = parse_time(end_time) if end_time else duration
    
    start_sec = max(0, min(start_sec, duration))
    end_sec = max(start_sec, min(end_sec, duration))
    
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)
    
    print(f"Video: {video_path.name}")
    print(f"  Duration: {duration:.1f}s | FPS: {video_fps:.1f}")
    print(f"  Extracting at {fps} FPS...")
    
    frame_interval = max(1, int(video_fps / fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    saved_count = 0
    
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        if (frame_count - start_frame) % frame_interval == 0:
            filename = f"{prefix}_{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1
            
            if saved_count % 50 == 0:
                print(f"  Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"✓ Extracted {saved_count} frames to {output_dir}")
    return saved_count


def extract_frames_batch(
    video_dir: str = "data/raw/videos",
    output_base: str = "data/raw/frames",
    fps: float = 1.0
) -> int:
    """Extract frames from all videos in directory"""
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"✗ Directory not found: {video_dir}")
        return 0
    
    extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in extensions]
    
    if not videos:
        print(f"No videos found in {video_dir}")
        return 0
    
    print(f"Found {len(videos)} videos")
    
    total_frames = 0
    for video in videos:
        output_dir = os.path.join(output_base, video.stem)
        total_frames += extract_frames(video, output_dir, fps)
    
    print(f"✓ Total: {total_frames} frames from {len(videos)} videos")
    return total_frames


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract frames from video')
    parser.add_argument('input', help='Video file or directory')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('-f', '--fps', type=float, default=1.0)
    parser.add_argument('-s', '--start', help='Start time (MM:SS)')
    parser.add_argument('-e', '--end', help='End time (MM:SS)')
    parser.add_argument('-b', '--batch', action='store_true')
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.input):
        extract_frames_batch(args.input, args.output or "data/raw/frames", args.fps)
    else:
        extract_frames(args.input, args.output, args.fps, start_time=args.start, end_time=args.end)


if __name__ == "__main__":
    main()

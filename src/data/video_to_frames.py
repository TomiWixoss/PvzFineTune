# -*- coding: utf-8 -*-
"""
Extract frames from video at specified FPS
"""

import os
import cv2
from pathlib import Path


def extract_frames(video_path: str, output_dir: str = None, 
                   fps: float = 1.0, prefix: str = None) -> int:
    """
    Extract frames from video at specified FPS
    
    Args:
        video_path: Path to video file
        output_dir: Output directory (default: data/raw/frames/<video_name>)
        fps: Frames per second to extract (default: 1.0)
        prefix: Prefix for frame filenames
    
    Returns:
        Number of frames extracted
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"✗ Video not found: {video_path}")
        return 0
    
    # Default output dir
    if output_dir is None:
        output_dir = f"data/raw/frames/{video_path.stem}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prefix for filenames
    if prefix is None:
        prefix = video_path.stem
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    print(f"Video: {video_path.name}")
    print(f"  Duration: {duration:.1f}s | FPS: {video_fps:.1f} | Total frames: {total_frames}")
    print(f"  Extracting at {fps} FPS...")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    if frame_interval < 1:
        frame_interval = 1
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at interval
        if frame_count % frame_interval == 0:
            filename = f"{prefix}_{saved_count:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            
            # Progress
            if saved_count % 50 == 0:
                print(f"  Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"✓ Extracted {saved_count} frames to {output_dir}")
    return saved_count


def extract_frames_batch(video_dir: str = "data/raw/videos", 
                         output_base: str = "data/raw/frames",
                         fps: float = 1.0) -> int:
    """
    Extract frames from all videos in directory
    
    Args:
        video_dir: Directory containing videos
        output_base: Base output directory
        fps: Frames per second to extract
    
    Returns:
        Total frames extracted
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"✗ Directory not found: {video_dir}")
        return 0
    
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]
    
    if not videos:
        print(f"No videos found in {video_dir}")
        return 0
    
    print(f"Found {len(videos)} videos")
    print("=" * 50)
    
    total_frames = 0
    for video in videos:
        output_dir = os.path.join(output_base, video.stem)
        frames = extract_frames(video, output_dir, fps)
        total_frames += frames
        print()
    
    print("=" * 50)
    print(f"✓ Total: {total_frames} frames from {len(videos)} videos")
    return total_frames


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract frames from video')
    parser.add_argument('input', help='Video file or directory')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('-f', '--fps', type=float, default=1.0, help='Frames per second (default: 1)')
    parser.add_argument('-p', '--prefix', help='Filename prefix')
    parser.add_argument('-b', '--batch', action='store_true', help='Process all videos in directory')
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.input):
        extract_frames_batch(
            video_dir=args.input,
            output_base=args.output or "data/raw/frames",
            fps=args.fps
        )
    else:
        extract_frames(
            video_path=args.input,
            output_dir=args.output,
            fps=args.fps,
            prefix=args.prefix
        )


if __name__ == "__main__":
    main()

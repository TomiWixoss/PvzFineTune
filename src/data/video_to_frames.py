# -*- coding: utf-8 -*-
"""
Extract frames from video at specified FPS with time range support
"""

import os
import cv2
from pathlib import Path


def parse_time(time_str: str) -> float:
    """Parse time string (MM:SS or HH:MM:SS) to seconds"""
    if time_str is None:
        return None
    
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(time_str)


def extract_frames(video_path: str, output_dir: str = None, 
                   fps: float = 1.0, prefix: str = None,
                   start_time: str = None, end_time: str = None) -> int:
    """
    Extract frames from video at specified FPS
    
    Args:
        video_path: Path to video file
        output_dir: Output directory (default: data/raw/frames/<video_name>)
        fps: Frames per second to extract (default: 1.0)
        prefix: Prefix for frame filenames
        start_time: Start time (MM:SS or HH:MM:SS)
        end_time: End time (MM:SS or HH:MM:SS)
    
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
    
    # Parse time range
    start_sec = parse_time(start_time) or 0
    end_sec = parse_time(end_time) or duration
    
    # Validate time range
    start_sec = max(0, min(start_sec, duration))
    end_sec = max(start_sec, min(end_sec, duration))
    
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)
    
    print(f"Video: {video_path.name}")
    print(f"  Duration: {duration:.1f}s | FPS: {video_fps:.1f} | Total frames: {total_frames}")
    
    if start_time or end_time:
        print(f"  Time range: {start_sec:.1f}s - {end_sec:.1f}s ({end_sec - start_sec:.1f}s)")
    
    print(f"  Extracting at {fps} FPS...")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    if frame_interval < 1:
        frame_interval = 1
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    saved_count = 0
    
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at interval
        if (frame_count - start_frame) % frame_interval == 0:
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
    parser.add_argument('-s', '--start', help='Start time (MM:SS or HH:MM:SS)')
    parser.add_argument('-e', '--end', help='End time (MM:SS or HH:MM:SS)')
    parser.add_argument('-r', '--ranges', nargs='+', help='Multiple time ranges: START-END (e.g., 0:10-0:15 1:30-2:00)')
    parser.add_argument('-b', '--batch', action='store_true', help='Process all videos in directory')
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.input):
        extract_frames_batch(
            video_dir=args.input,
            output_base=args.output or "data/raw/frames",
            fps=args.fps
        )
    elif args.ranges:
        # Multiple ranges mode
        extract_frames_multi_range(
            video_path=args.input,
            output_dir=args.output,
            fps=args.fps,
            prefix=args.prefix,
            ranges=args.ranges
        )
    else:
        extract_frames(
            video_path=args.input,
            output_dir=args.output,
            fps=args.fps,
            prefix=args.prefix,
            start_time=args.start,
            end_time=args.end
        )


def extract_frames_multi_range(video_path: str, output_dir: str = None,
                                fps: float = 1.0, prefix: str = None,
                                ranges: list = None) -> int:
    """
    Extract frames from multiple time ranges
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        fps: Frames per second to extract
        prefix: Prefix for frame filenames
        ranges: List of time ranges ["START-END", ...] e.g. ["0:10-0:15", "1:30-2:00"]
    
    Returns:
        Total frames extracted
    """
    video_path = Path(video_path)
    
    if output_dir is None:
        output_dir = f"data/raw/frames/{video_path.stem}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if prefix is None:
        prefix = video_path.stem
    
    print(f"Video: {video_path.name}")
    print(f"Extracting {len(ranges)} ranges at {fps} FPS...")
    print("=" * 50)
    
    total_saved = 0
    
    for i, time_range in enumerate(ranges):
        parts = time_range.split('-')
        if len(parts) != 2:
            print(f"✗ Invalid range format: {time_range} (use START-END)")
            continue
        
        start_time, end_time = parts[0], parts[1]
        range_prefix = f"{prefix}_r{i+1}"
        
        print(f"\nRange {i+1}: {start_time} - {end_time}")
        
        saved = extract_frames(
            video_path=str(video_path),
            output_dir=output_dir,
            fps=fps,
            prefix=range_prefix,
            start_time=start_time,
            end_time=end_time
        )
        total_saved += saved
    
    print("=" * 50)
    print(f"✓ Total: {total_saved} frames from {len(ranges)} ranges")
    return total_saved


if __name__ == "__main__":
    main()

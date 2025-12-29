# -*- coding: utf-8 -*-
"""
YouTube Downloader - Tải video với time range support
"""

import os
from yt_dlp import YoutubeDL

from ..utils.time_utils import parse_time


def download_video(
    url: str, 
    output_dir: str = "data/raw/videos",
    start_time: str = None, 
    end_time: str = None,
    filename: str = None
) -> str:
    """
    Download YouTube video with optional time range
    
    Args:
        url: YouTube video URL
        output_dir: Output directory
        start_time: Start time (format: "MM:SS" or "HH:MM:SS")
        end_time: End time (format: "MM:SS" or "HH:MM:SS")
        filename: Custom filename (without extension)
    
    Returns:
        Path to downloaded video
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename:
        output_template = os.path.join(output_dir, f"{filename}.%(ext)s")
    else:
        output_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
    }
    
    if start_time or end_time:
        ydl_opts['download_ranges'] = lambda info, ydl: [{
            'start_time': parse_time(start_time) if start_time else 0,
            'end_time': parse_time(end_time) if end_time else float('inf')
        }]
        ydl_opts['force_keyframes_at_cuts'] = True
    
    print(f"Downloading: {url}")
    if start_time or end_time:
        print(f"Time range: {start_time or '0:00'} - {end_time or 'end'}")
    
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = os.path.join(output_dir, f"{filename or info['title']}.mp4")
    
    print(f"✓ Downloaded: {video_path}")
    return video_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download YouTube video')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('-o', '--output', default='data/raw/videos')
    parser.add_argument('-s', '--start', help='Start time (MM:SS)')
    parser.add_argument('-e', '--end', help='End time (MM:SS)')
    parser.add_argument('-n', '--name', help='Custom filename')
    args = parser.parse_args()
    
    download_video(args.url, args.output, args.start, args.end, args.name)


if __name__ == "__main__":
    main()

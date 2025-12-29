# -*- coding: utf-8 -*-
"""
Time utilities - Parse và format timestamps
"""


def parse_time(time_str: str) -> float:
    """
    Parse time string → seconds
    Supports: "1:30", "01:30.500", "90", "1:30:00"
    """
    if time_str is None:
        return 0.0
    
    time_str = str(time_str).strip()
    
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
    
    return 0.0


def format_time(seconds: float) -> str:
    """
    Seconds → M:SS.mmm
    """
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{mins}:{secs:02d}.{millis:03d}"

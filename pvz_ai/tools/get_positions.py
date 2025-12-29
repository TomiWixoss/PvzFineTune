# -*- coding: utf-8 -*-
"""
Tool lấy tọa độ grid trong game
"""

import pyautogui
import pygetwindow as gw
import keyboard

from ..core.constants import PVZ_WINDOW_NAMES


def find_pvz_window():
    for title in gw.getAllTitles():
        if any(n.lower() in title.lower() for n in PVZ_WINDOW_NAMES):
            return gw.getWindowsWithTitle(title)[0]
    return None


def main():
    print("=" * 50)
    print("GET POSITIONS TOOL - FULL GRID")
    print("=" * 50)

    window = find_pvz_window()
    if not window:
        print("Open PvZ game first!")
        return

    print(f"✓ Found: {window.title}")
    print(f"  Window: ({window.left}, {window.top}) - {window.width}x{window.height}")
    print()
    print("Instructions:")
    print("  Press C for COLUMNS mode")
    print("  Press R for ROWS mode")
    print("  Press SPACE to record position")
    print("  Press ESC to finish")
    print()

    mode = "cols"
    cols_x = []
    rows_y = []

    def on_space():
        x, y = pyautogui.position()
        rel_x = x - window.left
        rel_y = y - window.top
        
        if mode == "cols":
            cols_x.append(rel_x)
            print(f"  Col {len(cols_x)-1}: x={rel_x}")
        else:
            rows_y.append(rel_y)
            print(f"  Row {len(rows_y)-1}: y={rel_y}")

    def on_c():
        nonlocal mode
        mode = "cols"
        print("\n>> Mode: COLUMNS")

    def on_r():
        nonlocal mode
        mode = "rows"
        print("\n>> Mode: ROWS")

    keyboard.on_press_key("space", lambda _: on_space())
    keyboard.on_press_key("c", lambda _: on_c())
    keyboard.on_press_key("r", lambda _: on_r())

    print("Ready! Press C/R to switch mode, SPACE to record...")
    print(f">> Mode: COLUMNS")

    keyboard.wait("esc")

    print()
    print("=" * 50)
    print("RESULTS - Copy to constants.py:")
    print("=" * 50)
    print()
    
    if cols_x:
        print(f"GRID_COLUMNS_X = {cols_x}")
    if rows_y:
        print(f"GRID_ROWS_Y = {rows_y}")


if __name__ == "__main__":
    main()

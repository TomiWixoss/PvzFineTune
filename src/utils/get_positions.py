# -*- coding: utf-8 -*-
"""
Tool to get mouse positions in PvZ window
Move mouse to position and press SPACE to record coordinates
"""

import pyautogui
import pygetwindow as gw
import keyboard


def find_pvz_window():
    for title in gw.getAllTitles():
        if any(n.lower() in title.lower() for n in ["Plants vs. Zombies", "Plants vs Zombies", "popcapgame1"]):
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
    print("  Mode 1: Get COLUMNS (press C)")
    print("    -> Click center of col 0, 1, 2... 8 (any row)")
    print()
    print("  Mode 2: Get ROWS (press R)")
    print("    -> Click center of row 0, 1, 2, 3, 4 (any col)")
    print()
    print("  Press SPACE to record position")
    print("  Press ESC to finish and see results")
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
        print("\n>> Mode: COLUMNS (click center of each column)")

    def on_r():
        nonlocal mode
        mode = "rows"
        print("\n>> Mode: ROWS (click center of each row)")

    keyboard.on_press_key("space", lambda _: on_space())
    keyboard.on_press_key("c", lambda _: on_c())
    keyboard.on_press_key("r", lambda _: on_r())

    print("Ready! Press C for columns, R for rows, SPACE to record...")
    print(f">> Mode: COLUMNS")
    print()

    keyboard.wait("esc")

    print()
    print("=" * 50)
    print("RESULTS - Copy to config.py:")
    print("=" * 50)
    print()
    
    if cols_x:
        print(f"GRID_COLUMNS_X = {cols_x}  # Col 0-{len(cols_x)-1}")
    else:
        print("GRID_COLUMNS_X = []  # No columns recorded")
    
    if rows_y:
        print(f"GRID_ROWS_Y = {rows_y}  # Row 0-{len(rows_y)-1}")
    else:
        print("GRID_ROWS_Y = []  # No rows recorded")
    
    print()
    print("# Grid visualization:")
    print(f"#      Col: {' '.join([f'{i:>4}' for i in range(len(cols_x))])}")
    for i, y in enumerate(rows_y):
        print(f"# Row {i}: {' '.join(['[ ]' + ' ' for _ in cols_x])}")


if __name__ == "__main__":
    main()

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
    print("GET POSITIONS TOOL")
    print("=" * 50)

    window = find_pvz_window()
    if not window:
        print("Open PvZ game first!")
        return

    print(f"✓ Found: {window.title}")
    print(f"  Window: ({window.left}, {window.top}) - {window.width}x{window.height}")
    print()
    print("Instructions:")
    print("  1. Move mouse to seed packet -> press SPACE")
    print("  2. Move mouse to slot 1 -> press SPACE")
    print("  3. Move mouse to slot 2,3,4,5... -> press SPACE")
    print("  4. Press ESC to stop and see results")
    print()

    positions = []

    def on_space():
        x, y = pyautogui.position()
        rel_x = x - window.left
        rel_y = y - window.top
        positions.append((rel_x, rel_y))
        print(f"  #{len(positions)}: ({rel_x}, {rel_y})")

    keyboard.on_press_key("space", lambda _: on_space())

    print("Ready! Move mouse and press SPACE...")
    print()

    keyboard.wait("esc")

    print()
    print("=" * 50)
    print("RESULTS:")
    print("=" * 50)
    if len(positions) >= 1:
        print(f"SEED_PACKET_POS = {positions[0]}")
    if len(positions) >= 2:
        slots = positions[1:]
        print(f"PLANT_SLOTS_X = {[p[0] for p in slots]}")
        print(f"ROW_Y = {slots[0][1]}")
    print()
    print("Copy these values to pvz_auto_play.py!")


if __name__ == "__main__":
    main()

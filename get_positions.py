# -*- coding: utf-8 -*-
"""
Di chuột vào vị trí và nhấn SPACE để ghi tọa độ
"""

import pyautogui
import pygetwindow as gw
import keyboard
import time

def find_pvz_window():
    for title in gw.getAllTitles():
        if any(n.lower() in title.lower() for n in ["Plants vs. Zombies", "Plants vs Zombies", "popcapgame1"]):
            return gw.getWindowsWithTitle(title)[0]
    return None

print("=" * 50)
print("GET POSITIONS TOOL")
print("=" * 50)

window = find_pvz_window()
if not window:
    print("Mở game PvZ trước!")
    exit()

print(f"✓ Found: {window.title}")
print(f"  Window: ({window.left}, {window.top}) - {window.width}x{window.height}")
print()
print("Hướng dẫn:")
print("  1. Di chuột vào seed packet -> nhấn SPACE")
print("  2. Di chuột vào slot 1 -> nhấn SPACE")
print("  3. Di chuột vào slot 2,3,4,5... -> nhấn SPACE")
print("  4. Nhấn ESC để dừng và xem kết quả")
print()

positions = []

def on_space():
    x, y = pyautogui.position()
    rel_x = x - window.left
    rel_y = y - window.top
    positions.append((rel_x, rel_y))
    print(f"  #{len(positions)}: ({rel_x}, {rel_y})")

keyboard.on_press_key("space", lambda _: on_space())

print("Sẵn sàng! Di chuột và nhấn SPACE...")
print()

keyboard.wait("esc")

print()
print("=" * 50)
print("KẾT QUẢ:")
print("=" * 50)
if len(positions) >= 1:
    print(f"SEED_PACKET_POS = {positions[0]}")
if len(positions) >= 2:
    slots = positions[1:]
    print(f"PLANT_SLOTS_X = {[p[0] for p in slots]}")
    print(f"ROW_Y = {slots[0][1]}")
print()
print("Copy vào pvz_auto_play.py!")

# -*- coding: utf-8 -*-
"""
Grid utilities - Convert pixel coordinates to grid positions
"""

from typing import List


def get_row(y: int, grid_rows_y: List[int]) -> int:
    """Y coordinate → row index (0-4)"""
    min_dist = float('inf')
    row = 0
    for i, row_y in enumerate(grid_rows_y):
        dist = abs(y - row_y)
        if dist < min_dist:
            min_dist = dist
            row = i
    return row


def get_col(x: int, grid_cols_x: List[int]) -> int:
    """X coordinate → col index (0-8)"""
    min_dist = float('inf')
    col = 0
    for i, col_x in enumerate(grid_cols_x):
        dist = abs(x - col_x)
        if dist < min_dist:
            min_dist = dist
            col = i
    return col

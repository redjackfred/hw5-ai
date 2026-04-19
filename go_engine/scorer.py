import numpy as np
from go_engine.board import Board, EMPTY, BLACK, WHITE

KOMI = 2.5
NEIGHBORS = [(-1,0),(1,0),(0,-1),(0,1)]


def _flood_territory(grid, visited, r, c):
    cells, bordering = [], set()
    stack = [(r, c)]
    while stack:
        row, col = stack.pop()
        if (row, col) in visited:
            continue
        if grid[row, col] != EMPTY:
            bordering.add(int(grid[row, col]))
            continue  # stone cells not added to visited — they can border multiple regions
        visited.add((row, col))
        cells.append((row, col))
        for dr, dc in NEIGHBORS:
            nr, nc = row+dr, col+dc
            if 0 <= nr < 9 and 0 <= nc < 9 and (nr, nc) not in visited:
                stack.append((nr, nc))
    return cells, bordering


def compute_score(board: Board, black_captured: int, white_captured: int) -> tuple:
    grid = board.grid
    black_stones = int((grid == BLACK).sum())
    white_stones = int((grid == WHITE).sum())
    visited = set()
    black_ter = white_ter = 0
    for r in range(9):
        for c in range(9):
            if grid[r, c] == EMPTY and (r, c) not in visited:
                cells, bordering = _flood_territory(grid, visited, r, c)
                if len(bordering) == 1:
                    owner = next(iter(bordering))
                    if owner == BLACK:
                        black_ter += len(cells)
                    else:
                        white_ter += len(cells)
    return (
        float(black_stones + black_ter + black_captured),
        float(white_stones + white_ter + white_captured + KOMI),
    )


def determine_winner(black_score: float, white_score: float) -> str:
    return "black" if black_score > white_score else "white"

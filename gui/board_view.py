import pygame
import numpy as np
from go_engine.board import BLACK, WHITE, EMPTY

BOARD_SIZE = 9
CELL = 54
MARGIN = 40
BOARD_PX = CELL * (BOARD_SIZE - 1) + 2 * MARGIN
STONE_R = 24
BOARD_COLOR = (188, 143, 83)
LINE_COLOR = (60, 40, 20)
LAST_COLOR = (220, 50, 50)


def grid_to_px(row, col):
    return MARGIN + col * CELL, MARGIN + row * CELL

def px_to_grid(px, py):
    return round((py - MARGIN) / CELL), round((px - MARGIN) / CELL)


class BoardView:
    def __init__(self, surface, offset_x=0):
        self.surface = surface
        self.ox = offset_x

    def draw(self, board_grid: np.ndarray, last_move=None):
        pygame.draw.rect(self.surface, BOARD_COLOR, (self.ox, 0, BOARD_PX, BOARD_PX))
        for i in range(BOARD_SIZE):
            x = self.ox + MARGIN + i * CELL
            pygame.draw.line(self.surface, LINE_COLOR, (x, MARGIN), (x, MARGIN+(BOARD_SIZE-1)*CELL))
            pygame.draw.line(self.surface, LINE_COLOR,
                (self.ox+MARGIN, MARGIN+i*CELL), (self.ox+MARGIN+(BOARD_SIZE-1)*CELL, MARGIN+i*CELL))
        for r in [2, 4, 6]:
            for c in [2, 4, 6]:
                x, y = grid_to_px(r, c)
                pygame.draw.circle(self.surface, LINE_COLOR, (self.ox+x, y), 4)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                color = board_grid[r, c]
                if color == EMPTY:
                    continue
                x, y = grid_to_px(r, c)
                cx, cy = self.ox + x, y
                sc = (20, 20, 20) if color == BLACK else (240, 240, 235)
                pygame.draw.circle(self.surface, sc, (cx, cy), STONE_R)
                hl = (80, 80, 80) if color == BLACK else (255, 255, 255)
                pygame.draw.circle(self.surface, hl, (cx-6, cy-6), 7)
        if last_move:
            x, y = grid_to_px(*last_move)
            pygame.draw.circle(self.surface, LAST_COLOR, (self.ox+x, y), 8)

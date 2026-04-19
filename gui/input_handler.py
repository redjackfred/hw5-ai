import pygame
from gui.board_view import px_to_grid, BOARD_SIZE

class InputHandler:
    def __init__(self, board_offset_x=0):
        self.ox = board_offset_x

    def get_board_click(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return None
        px, py = event.pos[0] - self.ox, event.pos[1]
        if px < 0:
            return None
        row, col = px_to_grid(px, py)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return row, col
        return None

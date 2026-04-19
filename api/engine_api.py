import numpy as np
from go_engine.game import Game
from go_engine.scorer import compute_score, determine_winner


class GoEngine:
    def __init__(self):
        self.game = Game()
        self._ai = None

    def set_ai(self, ai) -> None:
        self._ai = ai

    def new_game(self) -> None:
        self.game = Game()

    def place_stone(self, row: int, col: int) -> bool:
        return self.game.play(row, col)

    def is_legal(self, row: int, col: int) -> bool:
        return self.game.is_legal(row, col)

    def get_board(self) -> np.ndarray:
        return self.game.board.grid.copy()

    def get_score(self) -> tuple:
        return compute_score(
            self.game.board,
            self.game.captured.get(1, 0),
            self.game.captured.get(2, 0),
        )

    def is_game_over(self) -> bool:
        return self.game.is_over()

    def get_winner(self) -> str:
        return determine_winner(*self.get_score())

    def get_ai_move(self) -> tuple:
        if self._ai is None:
            raise RuntimeError("AI not set. Call set_ai() first.")
        return self._ai.select_move(self.game)

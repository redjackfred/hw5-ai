from go_engine.board import Board, EMPTY, BLACK, WHITE
from go_engine.rules import get_group, get_liberties, get_captured_stones


class Game:
    def __init__(self):
        self.board = Board()
        self._current_player = BLACK
        self._ko_board_hash = None   # hash of board state BEFORE the last move (forbidden to return to)
        self.captured = {BLACK: 0, WHITE: 0}
        self.move_history = []
        self._game_over = False

    @property
    def current_player(self):
        return self._current_player

    def is_legal(self, row: int, col: int) -> bool:
        if self._game_over:
            return False
        if not (0 <= row < 9 and 0 <= col < 9):
            return False
        if self.board.grid[row, col] != EMPTY:
            return False
        # Simulate the move on a copy
        test = self.board.copy()
        test.place_stone(row, col, self._current_player)
        cap = get_captured_stones(test, self._current_player, row, col)
        test.remove_stones(cap)
        # Ko rule: resulting board state must not equal the board state before the last move
        if test.zobrist_hash() == self._ko_board_hash:
            return False
        # Suicide rule
        if len(get_liberties(test, get_group(test, row, col))) == 0:
            return False
        return True

    def play(self, row: int, col: int) -> bool:
        if not self.is_legal(row, col):
            return False
        # Save current board hash BEFORE this move (becomes the Ko-forbidden state for next move)
        hash_before = self.board.zobrist_hash()
        self.board.place_stone(row, col, self._current_player)
        cap = get_captured_stones(self.board, self._current_player, row, col)
        self.board.remove_stones(cap)
        self.captured[self._current_player] += len(cap)
        self._ko_board_hash = hash_before
        self.move_history.append((row, col, self._current_player))
        self._current_player = WHITE if self._current_player == BLACK else BLACK
        return True

    def get_legal_moves(self) -> list:
        return [(r, c) for r in range(9) for c in range(9) if self.is_legal(r, c)]

    def pass_turn(self) -> None:
        self.move_history.append((None, None, self._current_player))
        self._current_player = WHITE if self._current_player == BLACK else BLACK
        self._game_over = True

    def is_over(self) -> bool:
        return self._game_over

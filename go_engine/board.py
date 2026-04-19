import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2

_ZOBRIST_TABLE = None

def _get_zobrist_table():
    global _ZOBRIST_TABLE
    if _ZOBRIST_TABLE is None:
        rng = np.random.default_rng(42)
        _ZOBRIST_TABLE = rng.integers(0, 2**63, size=(9, 9, 3), dtype=np.uint64)
    return _ZOBRIST_TABLE


class Board:
    def __init__(self):
        self.grid = np.zeros((9, 9), dtype=np.int8)

    def place_stone(self, row: int, col: int, color: int) -> None:
        self.grid[row, col] = color

    def remove_stones(self, stones: set) -> None:
        for row, col in stones:
            self.grid[row, col] = EMPTY

    def copy(self) -> "Board":
        b = Board()
        b.grid = self.grid.copy()
        return b

    def zobrist_hash(self) -> int:
        table = _get_zobrist_table()
        h = np.uint64(0)
        for r in range(9):
            for c in range(9):
                h ^= table[r, c, self.grid[r, c]]
        return int(h)

    def __eq__(self, other) -> bool:
        return np.array_equal(self.grid, other.grid)

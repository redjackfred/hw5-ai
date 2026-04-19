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


def _initial_hash() -> int:
    table = _get_zobrist_table()
    h = np.uint64(0)
    for r in range(9):
        for c in range(9):
            h ^= table[r, c, EMPTY]
    return int(h)


class Board:
    def __init__(self):
        self.grid = np.zeros((9, 9), dtype=np.int8)
        self._hash = _initial_hash()

    def place_stone(self, row: int, col: int, color: int) -> None:
        assert self.grid[row, col] == EMPTY, \
            f"Cell ({row},{col}) already occupied by {self.grid[row, col]}"
        table = _get_zobrist_table()
        self._hash ^= int(table[row, col, EMPTY])
        self._hash ^= int(table[row, col, color])
        self.grid[row, col] = color

    def remove_stones(self, stones: set) -> None:
        table = _get_zobrist_table()
        for row, col in stones:
            self._hash ^= int(table[row, col, self.grid[row, col]])
            self._hash ^= int(table[row, col, EMPTY])
            self.grid[row, col] = EMPTY

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.grid = self.grid.copy()
        b._hash = self._hash
        return b

    def zobrist_hash(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return np.array_equal(self.grid, other.grid)

    def __hash__(self) -> int:
        return self._hash

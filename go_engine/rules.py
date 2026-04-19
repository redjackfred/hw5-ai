from go_engine.board import Board, EMPTY, BLACK, WHITE

BOARD_SIZE = 9
NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def get_group(board: Board, row: int, col: int) -> set:
    color = board.grid[row, col]
    if color == EMPTY:
        return set()
    visited, stack = set(), [(row, col)]
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if _in_bounds(nr, nc) and board.grid[nr, nc] == color and (nr, nc) not in visited:
                stack.append((nr, nc))
    return visited


def get_liberties(board: Board, group: set) -> set:
    libs = set()
    for r, c in group:
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if _in_bounds(nr, nc) and board.grid[nr, nc] == EMPTY:
                libs.add((nr, nc))
    return libs


def get_captured_stones(board: Board, player_color: int, last_r: int, last_c: int) -> set:
    opponent = WHITE if player_color == BLACK else BLACK
    captured = set()
    for dr, dc in NEIGHBORS:
        nr, nc = last_r + dr, last_c + dc
        if not _in_bounds(nr, nc) or board.grid[nr, nc] != opponent:
            continue
        group = get_group(board, nr, nc)
        if len(get_liberties(board, group)) == 0:
            captured |= group
    return captured

import numpy as np
from go_engine.game import Game
from go_engine.board import BLACK, WHITE, EMPTY


def encode_board(game: Game) -> np.ndarray:
    """Returns float32 (17, 9, 9).
    Planes 0-7:  black positions at t, t-1, ..., t-7
    Planes 8-15: white positions at t, t-1, ..., t-7
    Plane 16:    1.0 if black to move, else 0.0
    """
    features = np.zeros((17, 9, 9), dtype=np.float32)
    boards = _reconstruct_history(game, 8)
    for i, grid in enumerate(boards):
        features[i] = (grid == BLACK).astype(np.float32)
        features[i + 8] = (grid == WHITE).astype(np.float32)
    features[16] = 1.0 if game.current_player == BLACK else 0.0
    return features


def _reconstruct_history(game: Game, n: int) -> list:
    from go_engine.board import Board
    from go_engine.rules import get_captured_stones
    replay = Board()
    snapshots = [replay.grid.copy()]
    for row, col, color in game.move_history:
        if row is None:  # pass move
            snapshots.append(snapshots[-1].copy())
            continue
        replay = replay.copy()
        replay.place_stone(row, col, color)
        cap = get_captured_stones(replay, color, row, col)
        replay.remove_stones(cap)
        snapshots.append(replay.grid.copy())
    recent = snapshots[-n:][::-1]
    while len(recent) < n:
        recent.append(np.zeros((9, 9), dtype=np.int8))
    return recent

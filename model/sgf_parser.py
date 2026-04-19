import numpy as np
from pathlib import Path
import sgfmill.sgf
from go_engine.game import Game
from go_engine.board import BLACK, WHITE
from model.features import encode_board


def _color_int(c: str) -> int:
    return BLACK if c == "b" else WHITE


def parse_sgf_file(path: str) -> list:
    samples = []
    try:
        with open(path, "rb") as f:
            game_tree = sgfmill.sgf.Sgf_game.from_bytes(f.read())
    except Exception:
        return samples
    if game_tree.get_size() != 9:
        return samples
    winner = game_tree.get_winner()
    if winner is None:
        return samples

    game = Game()
    for node in game_tree.get_main_sequence()[1:]:
        color, move = node.get_move()
        if color is None:
            continue
        if move is None:
            game.pass_turn()
            continue
        row, col = move
        if game.current_player != _color_int(color) or not game.is_legal(row, col):
            continue

        feat = encode_board(game)
        pol = np.zeros(82, dtype=np.float32)
        assert 0 <= row < 9 and 0 <= col < 9
        pol[row * 9 + col] = 1.0
        val = np.float32(1.0 if (winner == "b") == (game.current_player == BLACK) else -1.0)
        samples.append((feat, pol, val))
        game.play(row, col)
    return samples


def load_dataset(sgf_dir: str, max_games: int = 10000) -> tuple:
    all_f, all_p, all_v = [], [], []
    for path in list(Path(sgf_dir).rglob("*.sgf"))[:max_games]:
        for f, p, v in parse_sgf_file(str(path)):
            all_f.append(f); all_p.append(p); all_v.append(v)
    if not all_f:
        raise RuntimeError(f"No samples loaded from {sgf_dir}")
    return np.stack(all_f), np.stack(all_p), np.array(all_v, dtype=np.float32)

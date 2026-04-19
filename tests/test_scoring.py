from go_engine.board import Board, BLACK, WHITE
from go_engine.scorer import compute_score, determine_winner

def test_empty_board_white_wins_komi():
    b = Board()
    bs, ws = compute_score(b, 0, 0)
    assert bs == 0.0 and ws == 2.5

def test_black_territory_counts():
    b = Board()
    b.place_stone(0, 0, BLACK); b.place_stone(0, 2, BLACK)
    b.place_stone(1, 1, BLACK); b.place_stone(2, 0, BLACK)
    bs, ws = compute_score(b, 0, 0)
    assert bs > ws

def test_winner_black():
    b = Board()
    for r in range(9):
        for c in range(5):
            b.place_stone(r, c, BLACK)
    bs, ws = compute_score(b, 0, 0)
    assert determine_winner(bs, ws) == "black"

def test_winner_white_komi():
    b = Board()
    bs, ws = compute_score(b, 0, 0)
    assert determine_winner(bs, ws) == "white"

def test_captured_count_in_score():
    b = Board(); b.place_stone(4, 4, BLACK)
    bs, ws = compute_score(b, black_captured=3, white_captured=0)
    assert bs >= 4.0

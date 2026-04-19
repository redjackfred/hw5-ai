from go_engine.game import Game
from go_engine.board import BLACK, WHITE
from model.features import encode_board

def test_output_shape():
    assert encode_board(Game()).shape == (17, 9, 9)

def test_black_to_move_plane_all_ones():
    assert (encode_board(Game())[16] == 1).all()

def test_white_to_move_plane_all_zeros():
    g = Game(); g.play(4, 4)
    assert (encode_board(g)[16] == 0).all()

def test_black_stone_in_plane0():
    g = Game(); g.play(4, 4)
    feat = encode_board(g)
    assert feat[0, 4, 4] == 1.0
    assert feat[8, 4, 4] == 0.0

def test_history_plane_reflects_previous_state():
    g = Game()
    g.play(2, 2)   # Black at (2,2) — becomes t-1 after next two moves
    g.play(6, 6)   # White at (6,6)
    g.play(3, 3)   # Black at (3,3) — current state is t
    feat = encode_board(g)
    # plane 0 (black at t): both stones present
    assert feat[0, 2, 2] == 1.0 and feat[0, 3, 3] == 1.0
    # plane 1 (black at t-1): only (2,2) existed, (3,3) not yet played
    assert feat[1, 2, 2] == 1.0
    assert feat[1, 3, 3] == 0.0

def test_pass_does_not_change_board_plane():
    g = Game()
    g.play(4, 4)    # Black plays
    g.pass_turn()   # White passes — board unchanged
    feat = encode_board(g)
    # plane 0 (current board): black still at (4,4)
    assert feat[0, 4, 4] == 1.0
    # plane 1 (t-1): same board (pass doesn't change it)
    assert feat[1, 4, 4] == 1.0

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

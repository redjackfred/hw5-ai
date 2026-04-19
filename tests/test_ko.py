from go_engine.game import Game
from go_engine.board import BLACK, WHITE, EMPTY

def test_legal_on_empty():
    g = Game(); assert g.is_legal(4, 4) is True

def test_illegal_on_occupied():
    g = Game(); g.play(4, 4)
    assert g.is_legal(4, 4) is False

def test_suicide_is_illegal():
    g = Game()
    g.board.place_stone(0, 1, WHITE)
    g.board.place_stone(1, 0, WHITE)
    assert g.is_legal(0, 0) is False

def test_suicide_legal_if_captures():
    g = Game()
    g.board.place_stone(0, 0, WHITE)
    g.board.place_stone(0, 2, BLACK)
    g.board.place_stone(1, 1, BLACK)
    g.board.place_stone(1, 0, BLACK)
    assert g.is_legal(0, 1) is True

def test_ko_point_blocked():
    g = Game()
    g._ko_point = (1, 1)
    g._current_player = WHITE
    assert g.is_legal(1, 1) is False

def test_play_captures():
    g = Game()
    g.board.place_stone(4, 4, WHITE)
    g.board.place_stone(4, 3, BLACK)
    g.board.place_stone(4, 5, BLACK)
    g.board.place_stone(3, 4, BLACK)
    g.play(5, 4)  # Black completes capture
    assert g.board.grid[4, 4] == EMPTY
    assert g.captured[BLACK] == 1

def test_current_player_alternates():
    g = Game()
    assert g.current_player == BLACK
    g.play(0, 0)
    assert g.current_player == WHITE

def test_legal_moves_excludes_occupied():
    g = Game(); g.play(4, 4)
    assert (4, 4) not in g.get_legal_moves()
    assert len(g.get_legal_moves()) == 80

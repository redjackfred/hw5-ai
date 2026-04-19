import numpy as np
from api.engine_api import GoEngine

def test_new_game_empty():
    e = GoEngine(); e.new_game()
    assert (e.get_board() == 0).all()

def test_place_stone_legal():
    e = GoEngine(); e.new_game()
    assert e.place_stone(4, 4) is True

def test_place_stone_illegal_occupied():
    e = GoEngine(); e.new_game()
    e.place_stone(4, 4)
    assert e.place_stone(4, 4) is False

def test_is_legal():
    e = GoEngine(); e.new_game()
    assert e.is_legal(0, 0) is True
    e.place_stone(0, 0)
    assert e.is_legal(0, 0) is False

def test_get_board_reflects_moves():
    e = GoEngine(); e.new_game()
    e.place_stone(3, 3)
    assert e.get_board()[3, 3] == 1

def test_get_score_returns_floats():
    e = GoEngine(); e.new_game()
    b, w = e.get_score()
    assert isinstance(b, float) and isinstance(w, float)

def test_not_over_initially():
    e = GoEngine(); e.new_game()
    assert e.is_game_over() is False

def test_winner_returns_string():
    e = GoEngine(); e.new_game()
    assert e.get_winner() in ("black", "white")

def test_get_ai_move_no_ai_raises():
    e = GoEngine(); e.new_game()
    import pytest
    with pytest.raises(RuntimeError):
        e.get_ai_move()

def test_get_ai_move_with_stub_ai():
    e = GoEngine(); e.new_game()
    class StubAI:
        def select_move(self, game):
            return (4, 4)
    e.set_ai(StubAI())
    move = e.get_ai_move()
    assert move == (4, 4)

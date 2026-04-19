import numpy as np
from go_engine.board import Board, BLACK, WHITE, EMPTY

def test_initial_board_is_empty():
    b = Board()
    assert b.grid.shape == (9, 9)
    assert (b.grid == EMPTY).all()

def test_place_stone():
    b = Board()
    b.place_stone(4, 4, BLACK)
    assert b.grid[4, 4] == BLACK

def test_copy_does_not_mutate_original():
    b = Board()
    b2 = b.copy()
    b2.place_stone(0, 0, BLACK)
    assert b.grid[0, 0] == EMPTY

def test_remove_stones():
    b = Board()
    b.place_stone(0, 0, BLACK)
    b.remove_stones({(0, 0)})
    assert b.grid[0, 0] == EMPTY

def test_board_hash_differs_after_change():
    b = Board()
    h1 = b.zobrist_hash()
    b.place_stone(4, 4, BLACK)
    h2 = b.zobrist_hash()
    assert h1 != h2

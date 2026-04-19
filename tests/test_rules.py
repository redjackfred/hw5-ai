from go_engine.board import Board, BLACK, WHITE
from go_engine.rules import get_group, get_liberties, get_captured_stones

def test_get_group_single():
    b = Board(); b.place_stone(4, 4, BLACK)
    assert get_group(b, 4, 4) == {(4, 4)}

def test_get_group_connected():
    b = Board()
    b.place_stone(4, 4, BLACK); b.place_stone(4, 5, BLACK)
    assert get_group(b, 4, 4) == {(4, 4), (4, 5)}

def test_get_group_not_diagonal():
    b = Board()
    b.place_stone(4, 4, BLACK); b.place_stone(5, 5, BLACK)
    assert get_group(b, 4, 4) == {(4, 4)}

def test_liberties_center():
    b = Board(); b.place_stone(4, 4, BLACK)
    assert get_liberties(b, {(4, 4)}) == {(3,4),(5,4),(4,3),(4,5)}

def test_liberties_corner():
    b = Board(); b.place_stone(0, 0, BLACK)
    assert get_liberties(b, {(0, 0)}) == {(0,1),(1,0)}

def test_liberties_reduced_by_enemy():
    b = Board()
    b.place_stone(4, 4, BLACK); b.place_stone(4, 5, WHITE)
    libs = get_liberties(b, {(4, 4)})
    assert (4, 5) not in libs and len(libs) == 3

def test_capture_single_stone():
    b = Board()
    b.place_stone(4, 4, WHITE)
    b.place_stone(4, 3, BLACK); b.place_stone(4, 5, BLACK)
    b.place_stone(3, 4, BLACK); b.place_stone(5, 4, BLACK)
    assert (4, 4) in get_captured_stones(b, BLACK, 5, 4)

def test_no_capture_with_liberties():
    b = Board()
    b.place_stone(4, 4, WHITE); b.place_stone(4, 3, BLACK)
    assert len(get_captured_stones(b, BLACK, 4, 3)) == 0

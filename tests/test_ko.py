# tests/test_ko.py
from go_engine.game import Game
from go_engine.board import BLACK, WHITE, EMPTY


def test_legal_on_empty():
    g = Game()
    assert g.is_legal(4, 4) is True


def test_illegal_on_occupied():
    g = Game()
    g.play(4, 4)
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


def test_ko_blocks_recapture():
    """Real Ko: Black captures 1 white stone, White cannot immediately recapture."""
    g = Game()
    # Build a Ko position manually on the board (bypass play() to set up quickly)
    # Layout:
    #   . B . .
    #   B W B .
    #   . B . .
    # Black at (0,1),(1,0),(1,2),(2,1) surround White at (1,1)
    # but we need Ko: White has 1 stone surrounded except one liberty
    # Simpler Ko setup:
    #   Row 0: . B W .
    #   Row 1: B . B .   <- Black will capture W at (0,2), creating Ko at (0,2)
    #   White cannot recapture at (0,1) immediately
    #
    # Concrete:
    #   (0,1)=B, (0,3)=B, (1,2)=B  surround (0,2)=W from 3 sides
    #   Black plays (0,2) capturing... wait, (0,2) is White already.
    #
    # Cleanest single-ko setup:
    # Stones pre-placed so that Black plays one move and captures exactly 1 White stone.
    # Then White tries to recapture that exact cell — should be illegal.
    #
    # Board state pre-Black's move:
    #   (0,0)=W, (0,2)=B, (1,1)=B, (1,0)=B   <- White at (0,0) has 1 liberty: (0,1)
    # Black plays (0,1) -> captures W at (0,0). Ko point = board state before Black's move.
    # White cannot play (0,0) immediately because that would restore the pre-move board.

    g.board.place_stone(0, 0, WHITE)   # White stone to be captured
    g.board.place_stone(0, 2, BLACK)   # Black surrounding
    g.board.place_stone(1, 1, BLACK)   # Black surrounding
    g.board.place_stone(1, 0, BLACK)   # Black surrounding
    # Current player is BLACK
    assert g.is_legal(0, 1) is True
    g.play(0, 1)                       # Black captures White at (0,0)
    assert g.board.grid[0, 0] == EMPTY # White stone captured
    assert g.current_player == WHITE
    # White cannot immediately recapture at (0,0) — Ko rule
    assert g.is_legal(0, 0) is False


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
    g = Game()
    g.play(4, 4)
    assert (4, 4) not in g.get_legal_moves()
    assert len(g.get_legal_moves()) == 80


def test_pass_turn_records_history():
    g = Game()
    g.pass_turn()
    assert g.move_history[-1] == (None, None, BLACK)
    assert g.current_player == WHITE
    # Single pass does not end the game; two consecutive passes do
    assert g.is_over() is False
    g.pass_turn()
    assert g.is_over() is True

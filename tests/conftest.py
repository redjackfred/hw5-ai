import pytest

@pytest.fixture
def empty_board():
    from go_engine.board import Board
    return Board()

@pytest.fixture
def game():
    from go_engine.game import Game
    return Game()

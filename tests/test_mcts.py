import torch
from mcts.node import MCTSNode
from go_engine.game import Game

def test_node_initial():
    n = MCTSNode(Game(), None, 1.0)
    assert n.visit_count == 0 and n.is_leaf()

def test_node_q_zero_unvisited():
    assert MCTSNode(Game(), None, 1.0).q_value() == 0.0

def test_node_q_value():
    n = MCTSNode(Game(), None, 1.0)
    n.visit_count = 4; n.value_sum = 2.0
    assert abs(n.q_value() - 0.5) < 1e-6

def test_mcts_returns_legal_move():
    from model.network import GoNetwork
    from mcts.mcts import MCTS
    net = GoNetwork()
    mcts = MCTS(net, num_simulations=50, time_limit=5.0)
    game = Game()
    move = mcts.select_move(game)
    assert game.is_legal(*move)

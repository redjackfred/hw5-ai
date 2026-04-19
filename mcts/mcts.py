import time, copy
import numpy as np
import torch
from mcts.node import MCTSNode
from model.features import encode_board
from go_engine.board import BLACK, WHITE


def _copy_game(game):
    from go_engine.game import Game
    g = Game()
    g.board = game.board.copy()
    g._current_player = game._current_player
    g._ko_board_hash = game._ko_board_hash
    g.captured = dict(game.captured)
    g.move_history = list(game.move_history)
    g._game_over = game._game_over
    return g


class MCTS:
    def __init__(self, network, num_simulations=1600, time_limit=3.0,
                 c_puct=1.0, dir_alpha=0.3, dir_eps=0.25):
        self.network = network
        self.num_simulations = num_simulations
        self.time_limit = time_limit
        self.c_puct = c_puct
        self.dir_alpha = dir_alpha
        self.dir_eps = dir_eps
        self.device = next(network.parameters()).device

    def _infer(self, game):
        feat = torch.tensor(encode_board(game)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pol, val = self.network(feat)
        pol_softmax = torch.softmax(pol, dim=1)
        return pol_softmax.squeeze(0).cpu().numpy(), val.item()

    def _add_noise(self, root):
        moves = list(root.children.keys())
        if not moves:
            return
        noise = np.random.dirichlet([self.dir_alpha] * len(moves))
        for (m, child), n in zip(root.children.items(), noise):
            child.prior = (1 - self.dir_eps) * child.prior + self.dir_eps * n

    def _select(self, node):
        while not node.is_leaf():
            node = max(node.children.values(), key=lambda c: c.puct_score(self.c_puct))
        return node

    def _expand(self, node, policy):
        for r, c in node.game.get_legal_moves():
            g = _copy_game(node.game); g.play(r, c)
            node.children[(r, c)] = MCTSNode(g, node, float(policy[r * 9 + c]))

    def _backup(self, node, value, root_player):
        while node is not None:
            node.visit_count += 1
            sign = 1.0 if node.game.current_player == root_player else -1.0
            node.value_sum += sign * value
            node = node.parent

    def select_move(self, game, temperature=0.0):
        root = MCTSNode(_copy_game(game), None, 1.0)
        pol, _ = self._infer(root.game)
        self._expand(root, pol)
        self._add_noise(root)
        root_player = game.current_player

        t0 = time.time()
        for _ in range(self.num_simulations):
            if time.time() - t0 > self.time_limit:
                break
            leaf = self._select(root)
            if leaf.game.is_over() or not leaf.game.get_legal_moves():
                self._backup(leaf, 0.0, root_player); continue
            p, v = self._infer(leaf.game)
            self._expand(leaf, p)
            self._backup(leaf, v, root_player)

        moves = list(root.children.keys())
        counts = np.array([root.children[m].visit_count for m in moves], dtype=np.float32)
        if temperature == 0.0:
            return moves[int(np.argmax(counts))]
        probs = (counts ** (1 / temperature))
        probs /= probs.sum()
        return moves[np.random.choice(len(moves), p=probs)]

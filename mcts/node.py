import math


class MCTSNode:
    def __init__(self, game, parent, prior: float):
        self.game = game
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict = {}

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def q_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def puct_score(self, c_puct=1.0) -> float:
        pv = self.parent.visit_count if self.parent else 1
        return self.q_value() + c_puct * self.prior * math.sqrt(pv) / (1 + self.visit_count)

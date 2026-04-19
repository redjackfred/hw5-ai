import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = x
        x = F.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return F.relu(x + r)


class GoNetwork(nn.Module):
    def __init__(self, in_ch=17, ch=256, blocks=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])

        self.pol_conv = nn.Sequential(
            nn.Conv2d(ch, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())
        self.pol_fc = nn.Linear(2 * 81, 82)

        self.val_conv = nn.Sequential(
            nn.Conv2d(ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU())
        self.val_fc = nn.Sequential(
            nn.Linear(81, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())

    def forward(self, x):
        """Returns (policy_logits, value). Apply softmax to policy at inference time."""
        x = self.tower(self.stem(x))
        p = self.pol_fc(self.pol_conv(x).flatten(1))   # raw logits — use cross_entropy in training
        v = self.val_fc(self.val_conv(x).flatten(1))
        return p, v

    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

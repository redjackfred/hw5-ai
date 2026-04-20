"""Run: python -m model.train_rl --sl-checkpoint checkpoints/sl_best.pt"""
import argparse, copy, os, random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from model.network import GoNetwork
from mcts.mcts import MCTS, _copy_game
from go_engine.game import Game
from go_engine.board import BLACK, WHITE
from model.features import encode_board
from go_engine.scorer import compute_score, determine_winner

os.makedirs("checkpoints", exist_ok=True)

BUFFER = 100_000
SIM_TRAIN = 50
SIM_EVAL = 100
EVAL_GAMES = 10
WIN_THRESH = 0.45
RESIGN_THRESH = -0.6  # Q < -0.6 (~20% win prob) → resign in self-play
BATCH = 256
STEPS_PER_ITER = 50


def play_game(mcts: MCTS) -> list:
    game, traj, move_n = Game(), [], 0
    while not game.is_over() and game.get_legal_moves():
        tau = 1.0 if move_n < 30 else 0.0
        feat = encode_board(game)
        player = game.current_player
        move = mcts.select_move(game, temperature=tau)  # no resign during training
        pol = np.zeros(82, dtype=np.float32)
        pol[move[0] * 9 + move[1]] = 1.0
        traj.append((feat, pol, player))
        game.play(*move)
        move_n += 1
    bs, ws = compute_score(game.board, game.captured.get(BLACK, 0), game.captured.get(WHITE, 0))
    winner = determine_winner(bs, ws)
    return [(f, p, np.float32(1.0 if (winner == "black") == (pl == BLACK) else -1.0))
            for f, p, pl in traj]


def evaluate(cur_net, best_net, device) -> float:
    cur_net.eval(); best_net.eval()
    cm = MCTS(cur_net, SIM_EVAL, 10.0)
    bm = MCTS(best_net, SIM_EVAL, 10.0)
    wins = 0
    for i in range(EVAL_GAMES):
        game = Game()
        cur_is_black = (i % 2 == 0)
        cm_color = BLACK if cur_is_black else WHITE
        bm_color = WHITE if cur_is_black else BLACK
        move_n = 0
        resigned_winner = None
        while not game.is_over() and game.get_legal_moves() and move_n < 200:
            m = cm if game.current_player == cm_color else bm
            move = m.select_move(game, 0.0, resign_threshold=RESIGN_THRESH)
            if move is None:
                resigned_winner = "black" if game.current_player == WHITE else "white"
                break
            game.play(*move)
            move_n += 1
        if resigned_winner:
            w = resigned_winner
        else:
            bs, ws = compute_score(game.board, game.captured.get(BLACK, 0), game.captured.get(WHITE, 0))
            w = determine_winner(bs, ws)
        if (w == "black") == cur_is_black:
            wins += 1
    return wins / EVAL_GAMES


def train(sl_ckpt, output, iters=50):
    device = GoNetwork.get_device()
    net = GoNetwork().to(device)
    if os.path.exists(sl_ckpt):
        net.load_state_dict(torch.load(sl_ckpt, map_location=device, weights_only=True))
        print(f"Loaded SL weights: {sl_ckpt}")
    best = copy.deepcopy(net)
    torch.save(best.state_dict(), output)
    buf = deque(maxlen=BUFFER)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    train_mcts = MCTS(best, SIM_TRAIN, 60.0)

    for it in range(1, iters + 1):
        print(f"\n=== Iter {it}/{iters} ===")
        net.eval()
        for gi in range(20):
            buf.extend(play_game(train_mcts))
            if (gi + 1) % 5 == 0:
                print(f"  self-play {gi+1}/20  buf={len(buf)}")
        if len(buf) < BATCH:
            continue
        net.train()
        for _ in range(STEPS_PER_ITER):
            b = random.sample(buf, BATCH)
            f = torch.tensor(np.stack([x[0] for x in b])).to(device)
            p = torch.tensor(np.stack([x[1] for x in b])).to(device)
            v = torch.tensor(np.array([x[2] for x in b])).unsqueeze(1).to(device)
            pp, pv = net(f)
            loss = F.cross_entropy(pp, p) + F.mse_loss(pv, v)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  loss={loss.item():.4f}")
        wr = evaluate(net, best, device)
        print(f"  win_rate={wr:.2f}")
        if wr >= WIN_THRESH:
            best = copy.deepcopy(net)
            torch.save(best.state_dict(), output)
            train_mcts = MCTS(best, SIM_TRAIN, 60.0)
            print(f"  → new best (wr={wr:.2f})")

    print("RL done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sl-checkpoint", default="checkpoints/sl_best.pt")
    p.add_argument("--output", default="checkpoints/rl_best.pt")
    p.add_argument("--iterations", type=int, default=50)
    a = p.parse_args()
    train(a.sl_checkpoint, a.output, a.iterations)

"""Run: python -m model.train_rl --sl-checkpoint checkpoints/sl_best.pt"""
import argparse, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from model.network import GoNetwork
from mcts.mcts import MCTS
from go_engine.game import Game
from go_engine.board import BLACK, WHITE
from model.features import encode_board
from go_engine.scorer import compute_score, determine_winner

os.makedirs("checkpoints", exist_ok=True)

BUFFER = 200_000
SIM_TRAIN = 999
TIME_TRAIN = 0.05   # 0.05s per move in self-play
SIM_EVAL = 999
TIME_EVAL = 1.0     # 1s per move in evaluation vs SL baseline
GAMES_PER_ITER = 40
MAX_MOVES = 150     # cap game length to prevent runaway games
EVAL_GAMES = 10
RESIGN_THRESH = -0.6        # for evaluation vs SL
RESIGN_THRESH_TRAIN = -0.8  # ~10% win prob — resign in self-play to cut short hopeless games
BATCH = 256
STEPS_PER_ITER = 100


def play_game(mcts: MCTS) -> list:
    """Pure self-play: same network plays both Black and White."""
    game, traj, move_n = Game(), [], 0
    while not game.is_over() and game.get_legal_moves() and move_n < MAX_MOVES:
        tau = 1.0 if move_n < 30 else 0.0
        feat = encode_board(game)
        player = game.current_player
        move = mcts.select_move(game, temperature=tau, resign_threshold=RESIGN_THRESH_TRAIN)
        if move is None:
            resigned_color = game.current_player
            winner = "black" if resigned_color == WHITE else "white"
            return [(f, p, np.float32(1.0 if (winner == "black") == (pl == BLACK) else -1.0))
                    for f, p, pl in traj]
        pol = np.zeros(82, dtype=np.float32)
        pol[move[0] * 9 + move[1]] = 1.0
        traj.append((feat, pol, player))
        game.play(*move)
        move_n += 1
    bs, ws = compute_score(game.board, game.captured.get(BLACK, 0), game.captured.get(WHITE, 0))
    winner = determine_winner(bs, ws)
    return [(f, p, np.float32(1.0 if (winner == "black") == (pl == BLACK) else -1.0))
            for f, p, pl in traj]


def evaluate_vs_sl(net, sl_net, device) -> float:
    """Evaluate net against the SL baseline for logging purposes."""
    net.eval(); sl_net.eval()
    nm = MCTS(net, SIM_EVAL, TIME_EVAL)
    sm = MCTS(sl_net, SIM_EVAL, TIME_EVAL)
    wins = 0
    for i in range(EVAL_GAMES):
        game = Game()
        net_is_black = (i % 2 == 0)
        net_color = BLACK if net_is_black else WHITE
        move_n = 0
        resigned_winner = None
        while not game.is_over() and game.get_legal_moves() and move_n < 200:
            m = nm if game.current_player == net_color else sm
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
        if (w == "black") == net_is_black:
            wins += 1
    return wins / EVAL_GAMES


def train(sl_ckpt, output, iters=50):
    device = GoNetwork.get_device()

    # Load SL weights as starting point AND as fixed baseline for evaluation
    sl_net = GoNetwork().to(device)
    if os.path.exists(sl_ckpt):
        sl_net.load_state_dict(torch.load(sl_ckpt, map_location=device, weights_only=True))
        print(f"Loaded SL weights: {sl_ckpt}")
    sl_net.eval()

    net = GoNetwork().to(device)
    net.load_state_dict(sl_net.state_dict())   # start from SL

    buf = deque(maxlen=BUFFER)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    best_wr = 0.0

    for it in range(1, iters + 1):
        t_iter = time.time()
        print(f"\n=== Iter {it}/{iters} ===")

        # --- Self-play: net vs itself ---
        net.eval()
        mcts = MCTS(net, SIM_TRAIN, TIME_TRAIN)
        t_sp = time.time()
        total_moves = 0
        for gi in range(GAMES_PER_ITER):
            samples = play_game(mcts)
            total_moves += len(samples)
            buf.extend(samples)
            if (gi + 1) % 5 == 0:
                elapsed = time.time() - t_sp
                avg_moves = total_moves / (gi + 1)
                print(f"  self-play {gi+1}/{GAMES_PER_ITER}  buf={len(buf)}  {elapsed:.0f}s  avg={avg_moves:.0f}moves/game", flush=True)

        if len(buf) < BATCH:
            continue

        # --- Train with REINFORCE-style policy loss ---
        # Winner's moves: v=+1 → reinforce chosen move (increase prob)
        # Loser's moves:  v=-1 → penalize chosen move (decrease prob)
        net.train()
        total_loss = 0.0
        for _ in range(STEPS_PER_ITER):
            b = random.sample(buf, BATCH)
            f = torch.tensor(np.stack([x[0] for x in b])).to(device)
            p = torch.tensor(np.stack([x[1] for x in b])).to(device)
            v = torch.tensor(np.array([x[2] for x in b])).unsqueeze(1).to(device)
            pp, pv = net(f)
            # Policy: weight log-prob of chosen move by outcome sign
            log_probs = F.log_softmax(pp, dim=1)
            pol_loss = -(v.squeeze(1) * (log_probs * p).sum(dim=1)).mean()
            val_loss = F.mse_loss(pv, v)
            loss = pol_loss + val_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / STEPS_PER_ITER
        iter_time = time.time() - t_iter
        print(f"  loss={avg_loss:.4f}  iter_time={iter_time:.0f}s")

        # --- Always save latest checkpoint ---
        torch.save(net.state_dict(), output)

        # --- Evaluate vs SL baseline every 5 iters (slow, skip otherwise) ---
        if it % 5 == 0:
            wr = evaluate_vs_sl(net, sl_net, device)
            print(f"  vs_SL win_rate={wr:.2f}", end="")
            if wr > best_wr:
                best_wr = wr
                torch.save(net.state_dict(), output.replace(".pt", "_best.pt"))
                print(f"  → new best!")
            else:
                print()

    print(f"\nRL done. Best win rate vs SL: {best_wr:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sl-checkpoint", default="checkpoints/sl_best.pt")
    p.add_argument("--output", default="checkpoints/rl_best.pt")
    p.add_argument("--iterations", type=int, default=50)
    a = p.parse_args()
    train(a.sl_checkpoint, a.output, a.iterations)

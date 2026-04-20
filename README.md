# 9x9 Go Engine — AlphaGo Style

USF Physics 303 / CS 486 / CS 686 — Homework 5

---

## Setup

**Requirements:** Python 3.11+, macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Play

```bash
# Human (Black) vs AI (White)
python main.py --checkpoint checkpoints/sl_best.pt

# AI vs AI demo
python main.py --checkpoint checkpoints/sl_best.pt --ai-vs-ai

# If no checkpoint exists, AI plays with an untrained (random) network
python main.py
```

**Controls:**
- Click any intersection to place a stone
- Click **Resign** button in the sidebar to concede
- Close the window to quit

**GUI features:**
- 9×9 board with star points
- Black/White stones with highlight
- Red dot marks the most recent move
- Sidebar: captured stone counts, current score, turn indicator
- AI thinking indicator while MCTS runs
- Win/resign banner at game end

---

## Run Tests

```bash
pytest tests/ -v
```

56 tests covering: stone placement, capture, Ko rule, suicide rejection, territory scoring, engine API, feature encoding, network shapes, SGF parsing, MCTS node logic.

---

## Training Pipeline

### Step 1 — Download SGF games

Place 9×9 SGF files under `data/sgf/`. A ready-to-use dataset (37k games) can be downloaded from:

```bash
mkdir -p data/sgf
# CWI professional 9x9 collection (517 games)
curl -L -o /tmp/9x9.tgz https://homepages.cwi.nl/~aeb/go/games/9x9.tgz
tar -xzf /tmp/9x9.tgz -C data/sgf/

# OGS online games ~22k usable
curl -L -o /tmp/go9.zip "https://www.dropbox.com/s/7hzopmr000ndham/go9_20170307_200x200.zip?dl=1"
unzip -q /tmp/go9.zip -d data/sgf/
```

### Step 2 — Supervised Learning

```bash
python -m model.train_sl --sgf-dir data/sgf --epochs 25
# Saves checkpoints/sl_best.pt (~2-4 hours on M4 Pro, MPS backend)
```

### Step 3 — RL Self-Play (optional)

```bash
python -m model.train_rl --sl-checkpoint checkpoints/sl_best.pt --iterations 30
# Saves checkpoints/rl_best.pt
```

---

## AI Strategy

### Approach: AlphaGo-style (SL pretraining → MCTS at inference)

The AI is built in three layers:

**1. Dual-Head ResNet (`model/network.py`)**
A 10-block residual network with 256 channels (≈5M parameters). Given a board position encoded as a 17-plane tensor, it outputs:
- *Policy head*: 82 logits (81 board moves + 1 pass) — predicts where a strong player would move
- *Value head*: scalar ∈ (−1, 1) — predicts who is winning

**2. Supervised Learning (`model/train_sl.py`)**
The network is pretrained on 169k positions from real human/computer games (KGS, OGS). This gives the AI a strong prior over plausible moves before any self-play.

**3. MCTS with PUCT (`mcts/mcts.py`)**
At inference, Monte Carlo Tree Search (1600 simulations, 3-second budget) uses the network to guide search:
- *Selection*: PUCT formula balances exploitation (Q-value) and exploration (prior probability)
- *Expansion*: network policy initializes priors for new nodes
- *Backup*: negamax propagates value estimates up the tree
- *Dirichlet noise* at root encourages exploration

The final move is the child with the highest visit count (temperature=0 at inference).

**Why this approach?**
MCTS alone (without a network) plays random rollouts — weak on 9×9 because the game is tactically sharp. A pure neural network without search is fast but brittle. Combining them gives the best of both: the network prunes the search space, and MCTS corrects the network's errors by lookahead. This is the core insight of AlphaGo.

**AI resign:** After MCTS finishes, if the root Q-value < −0.6 (≈20% win probability), the AI resigns rather than play out a hopeless position.

---

## Design Decisions & Challenges

**Ko detection via Zobrist hashing**
The Ko rule forbids returning to the previous board state. A naïve coordinate-guard (storing the recaptured point) fails for snapback positions. We use incremental Zobrist hashing (XOR on place/remove) — O(1) per move — and compare the resulting hash against the pre-move state.

**Flood-fill territory scoring**
Chinese area scoring requires flood-filling empty regions and checking which color's stones border them. A subtle bug: if stone cells are added to the `visited` set, a stone shared between two empty regions is "consumed" by the first fill, making the second region appear neutral. The fix is to check stone neighbors without adding them to `visited`.

**Raw logits vs. softmax in training**
PyTorch's `F.cross_entropy` expects raw logits, not probabilities. Applying `softmax` inside `forward()` then passing to `cross_entropy` computes log(softmax(softmax(x))), compressing gradients near 0/1 and severely slowing training. The fix: return raw logits from the network; apply `softmax` only at MCTS inference time.

**Two consecutive passes end the game**
Standard Go ends when both players pass consecutively. The initial implementation ended the game after a single pass. This was caught when implementing the SGF parser (pass moves in real games need to advance the player without ending the game).

**MCTS backup sign convention (negamax)**
The MCTS backup must use negamax: negate the value at each step going up the tree, so every node's Q-value is from its own current player's perspective. A "fixed-sign" approach (always from root's perspective) causes opponent nodes to also maximize the root's Q, breaking minimax.

**Testing beyond the provided suite**
Beyond unit tests, we tested: (1) parsing 37k real SGF files and verifying sample shapes/values, (2) running 50-simulation MCTS on an empty board and confirming the returned move is legal, (3) a complete Human vs AI game via the GUI to check move legality, score display, and resign logic.

---

## File Structure

```
go_engine/        # Pure rules engine (no AI dependencies)
  board.py        # Board state, Zobrist hashing
  rules.py        # Liberties, capture, legality
  game.py         # Game flow, Ko tracking
  scorer.py       # Chinese area scoring
api/
  engine_api.py   # Clean interface for test harness
model/
  features.py     # 17-plane board encoding
  network.py      # Dual-head ResNet
  sgf_parser.py   # SGF → training samples
  train_sl.py     # Supervised learning script
  train_rl.py     # RL self-play script
mcts/
  node.py         # MCTS tree node (N, W, Q, P)
  mcts.py         # PUCT search loop
gui/
  board_view.py   # Pygame board renderer
  sidebar.py      # Sidebar (scores, resign button)
  input_handler.py# Mouse → board coordinate
main.py           # Entry point
tests/            # 56 unit tests
```

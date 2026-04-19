# 9x9 Go Engine — AlphaGo Style

## Setup
```bash
pip install -r requirements.txt
```

## Training

**Step 1** — Place 9x9 SGF files in `data/sgf/`
- KGS: https://www.gokgs.com/archives.jsp (filter 9x9)
- CGOS: http://www.yss-aya.com/cgos/9x9/

**Step 2** — Supervised learning (~2-4 hours on M4 Pro):
```bash
python -m model.train_sl --sgf-dir data/sgf --epochs 25
```

**Step 3** — RL self-play (~4-8 hours, run overnight):
```bash
python -m model.train_rl --sl-checkpoint checkpoints/sl_best.pt
```

## Play
```bash
python main.py                    # Human vs AI
python main.py --ai-vs-ai         # AI vs AI
python main.py --checkpoint path  # Custom checkpoint
```

## Tests
```bash
pytest tests/ -v
```

## AI Strategy
AlphaGo-style: supervised learning on human games → RL self-play fine-tuning → MCTS (1600 simulations / 3 seconds) guided by dual-head ResNet (10 residual blocks, 256 channels, ~5M parameters). Feature encoding uses 17 planes (8-step history for each player + current turn). Chinese scoring with komi 2.5.

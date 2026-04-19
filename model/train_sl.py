"""Run: python -m model.train_sl --sgf-dir data/sgf --epochs 25"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from model.network import GoNetwork
from model.sgf_parser import load_dataset

os.makedirs("checkpoints", exist_ok=True)


def train(sgf_dir, epochs, output, batch_size=256):
    device = GoNetwork.get_device()
    print(f"Device: {device}")
    features, policies, values = load_dataset(sgf_dir)
    print(f"Dataset: {len(features)} positions")

    ds = TensorDataset(torch.tensor(features), torch.tensor(policies),
                       torch.tensor(values).unsqueeze(1))
    val_n = max(1, int(0.05 * len(ds)))
    tr_ds, va_ds = random_split(ds, [len(ds) - val_n, val_n])
    tr_ldr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    va_ldr = DataLoader(va_ds, batch_size=batch_size)

    net = GoNetwork().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = float("inf")

    for ep in range(1, epochs + 1):
        net.train()
        tr_loss = 0.0
        for f, p, v in tr_ldr:
            f, p, v = f.to(device), p.to(device), v.to(device)
            pp, pv = net(f)
            loss = F.cross_entropy(pp, p) + F.mse_loss(pv, v)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(f)

        net.eval(); va_loss = 0.0
        with torch.no_grad():
            for f, p, v in va_ldr:
                f, p, v = f.to(device), p.to(device), v.to(device)
                pp, pv = net(f)
                va_loss += (F.cross_entropy(pp, p) + F.mse_loss(pv, v)).item() * len(f)
        va_loss /= len(va_ds)
        sched.step(va_loss)
        print(f"Epoch {ep}/{epochs}  tr={tr_loss/len(tr_ds):.4f}  va={va_loss:.4f}")
        if va_loss < best:
            best = va_loss
            torch.save(net.state_dict(), output)
            print(f"  → saved {output}")

    print(f"Done. Best val loss: {best:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sgf-dir", default="data/sgf")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--output", default="checkpoints/sl_best.pt")
    p.add_argument("--batch-size", type=int, default=256)
    a = p.parse_args()
    train(a.sgf_dir, a.epochs, a.output, a.batch_size)

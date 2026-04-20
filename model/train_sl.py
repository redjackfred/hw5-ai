"""Run: python -m model.train_sl --sgf-dir data/sgf --epochs 25"""
import argparse, os, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from model.network import GoNetwork
from model.sgf_parser import load_dataset

os.makedirs("checkpoints", exist_ok=True)


def _augment(f_np: np.ndarray, p_np: np.ndarray):
    """Apply a random one of 8 board symmetries to a batch.

    Go board has 4-fold rotational + 2-fold reflective symmetry.
    Augmenting with all 8 transforms effectively multiplies training data 8x.

    f_np: (B, 17, 9, 9)   p_np: (B, 82)
    Returns transformed copies as numpy arrays.
    """
    t = random.randint(0, 7)
    if t == 0:
        return f_np, p_np

    flip = t >= 4   # transforms 4-7 include a left-right flip first
    k = t % 4       # number of 90° CCW rotations after optional flip

    f = f_np.copy()
    p = p_np.copy()

    if flip:
        f = np.flip(f, axis=3).copy()   # flip columns (left-right)

    if k > 0:
        f = np.rot90(f, k, axes=(2, 3)).copy()   # k × 90° CCW

    # Remap policy move indices to match the spatial transformation.
    # Original index = r*9 + c. Apply the same flip/rotation to each (r,c).
    rs = np.arange(81) // 9   # original row for each of the 81 moves
    cs = np.arange(81) % 9    # original col

    if flip:
        cs = 8 - cs

    if k == 1:      # 90° CCW: (r,c) → (8-c, r)
        rs, cs = 8 - cs, rs.copy()
    elif k == 2:    # 180°:    (r,c) → (8-r, 8-c)
        rs, cs = 8 - rs, 8 - cs
    elif k == 3:    # 270° CCW:(r,c) → (c, 8-r)
        rs, cs = cs.copy(), 8 - rs

    new_idx = rs * 9 + cs       # (81,) destination indices
    new_p = np.zeros_like(p)
    new_p[:, new_idx] = p[:, :81]   # remap board moves
    new_p[:, 81] = p[:, 81]         # pass move index 81 is position-independent

    return f, new_p


def train(sgf_dir, epochs, output, batch_size=256):
    device = GoNetwork.get_device()
    print(f"Device: {device}")
    features, policies, values = load_dataset(sgf_dir)
    print(f"Dataset: {len(features)} positions (×8 via symmetry augmentation = "
          f"{len(features)*8:,} effective)")

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
            # Apply random symmetry augmentation to each training batch
            f_aug, p_aug = _augment(f.numpy(), p.numpy())
            f = torch.tensor(f_aug).to(device)
            p = torch.tensor(p_aug).to(device)
            v = v.to(device)
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
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--output", default="checkpoints/sl_best.pt")
    p.add_argument("--batch-size", type=int, default=256)
    a = p.parse_args()
    train(a.sgf_dir, a.epochs, a.output, a.batch_size)

"""Χάρτης του φύλλου 8×8 από ένα checkpoint: ποιος κερδίζει πού, στο val.

    python3 -m lightstorm.sheetmap runs_e20/C2/epoch20.pt [runs_e20/B/epoch20.pt ...]

Ανά κελί: % των tokens όπου ο expert είναι νικητής (· = κάτω από 0,1%, δηλαδή όχι νικητής).
Έλεγχος σκακιέρας: πόσοι νικητές κάθονται σε (γραμμή+στήλη) ζυγό / μονό, και πόσα ζευγάρια
νικητών είναι γείτονες στον σταυρό (σε τέλεια σκακιέρα: 0).
"""
import os
import sys

import torch

from . import metrics
from .model import Config, Lightstorm
from .train import ROOT, load_bin


@torch.no_grad()
def winner_counts(model, val, ctx, device, batch=32):
    model.eval()
    n = (len(val) - 1) // ctx
    xs = val[:n * ctx].view(n, ctx)
    counts = torch.zeros(metrics.N_EXPERTS, dtype=torch.long)
    for i in range(0, n, batch):
        _, _, aux = model(xs[i:i + batch].to(device))
        counts += torch.bincount(aux["winner"].cpu(), minlength=metrics.N_EXPERTS)
    return counts


def draw(counts, side=metrics.SIDE):
    share = counts.float() / counts.sum()
    win = share >= metrics.ACTIVE_SHARE
    lines = []
    for r in range(side):
        cells = []
        for c in range(side):
            s = share[r * side + c].item()
            cells.append(f"{100 * s:5.1f}" if s >= metrics.ACTIVE_SHARE else "    ·")
        lines.append(" ".join(cells))
    idx = win.nonzero().flatten().tolist()
    even = sum(1 for i in idx if (i // side + i % side) % 2 == 0)
    cr = metrics.cross(side)
    adjacent = sum(1 for i in idx for j in cr[i, 1:].tolist() if j > i and win[j])
    lines.append(f"νικητές {len(idx)} · ζυγά κελιά {even} / μονά {len(idx) - even} · "
                 f"ζευγάρια γειτονικών νικητών {adjacent} (από {2 * side * side} ακμές)")
    return "\n".join(lines)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val = load_bin(os.path.join(ROOT, "corpus", "val.bin"))
    for path in sys.argv[1:]:
        ck = torch.load(path, map_location=device)
        model = Lightstorm(Config(**ck["cfg"])).to(device)
        model.load_state_dict(ck["model"])
        if model.cfg.arm != "A":
            model.moe.sigma = ck["bench"].get("sigma", 0.7)
        print(f"== {path} (σκέλος {model.cfg.arm})")
        print(draw(winner_counts(model, val, model.cfg.ctx, device)))
        print()


if __name__ == "__main__":
    main()

"""Πείραμα 13β: «μία τεράστια πόλη» ή «προάστιο»; Ανάλυση έτοιμων checkpoints, χωρίς training.

Για κάθε run, πάνω σε ένα κομμάτι του val:
- base: val loss όπως είναι.
- zero: Δloss αν το MoE layer βγάζει 0 (κανένας ειδικός). Μικρό → προάστιο: το μοντέλο δεν χρειάζεται τις πόλεις.
- one_city: Δloss αν κάθε token παίρνει τον μέσο όρο ΟΛΩΝ των ειδικών (μία πόλη για όλους).
  Μικρό ενώ το zero είναι μεγάλο → τεράστια πόλη: οι ειδικοί χρειάζονται αλλά είναι ίδιοι μεταξύ τους.
- cos_all / cos_near: μέση ομοιότητα (cosine) των εξόδων δύο ειδικών στο ίδιο token, για όλα τα ζεύγη
  και για γείτονες στον τόρο (απόσταση 1).

    python -m lightstorm.cities --runs DIR [DIR ...] --corpus corpus_r3 --out cities.json
DIR: φάκελος με s*/{A,B}/epoch1.pt (ή απευθείας με {A,B}/epoch1.pt).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics
from .model import Config, Lightstorm
from .train import load_bin


def all_experts(ex, x):
    """x (N, d) → (n, N, d): η έξοδος κάθε ειδικού για κάθε token."""
    h = F.gelu(torch.einsum("nd,edh->enh", x, ex.w1) + ex.b1[:, None])
    return torch.einsum("enh,ehd->end", h, ex.w2) + ex.b2[:, None]


@torch.no_grad()
def analyse(ck_path, val, n_win, device):
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    m = Lightstorm(Config(**ck["cfg"])).to(device).eval()
    m.load_state_dict(ck["model"])
    moe, side, ctx = m.moe, m.cfg.side, m.cfg.ctx
    starts = np.linspace(0, len(val) - ctx - 2, n_win).astype(int)
    xs = torch.stack([val[s:s + ctx] for s in starts]).to(device)
    ys = torch.stack([val[s + 1:s + ctx + 1] for s in starts]).to(device)
    mode, grab = ["base"], {}

    def hook(mod, inp, out):
        if mode[0] == "base":
            grab["x"] = inp[0].reshape(-1, inp[0].shape[-1])
            return None
        y, aux = out
        if mode[0] == "zero":
            return torch.zeros_like(y), aux
        x = inp[0].reshape(-1, inp[0].shape[-1])
        mean = torch.cat([all_experts(moe.experts, x[i:i + 512]).mean(0) for i in range(0, len(x), 512)])
        return mean.reshape(y.shape), aux

    h = m.blocks[m.cfg.moe_layer].mlp.register_forward_hook(hook)
    loss = {}
    for mode[0] in ("base", "zero", "one_city"):
        tot = 0.0
        for i in range(0, n_win, 8):
            _, _, aux = m(xs[i:i + 8], ys[i:i + 8])
            tot += aux["ce"].item() * len(xs[i:i + 8])
        loss[mode[0]] = tot / n_win
    h.remove()
    # ομοιότητα ειδικών σε 512 tokens
    x = grab["x"][torch.linspace(0, len(grab["x"]) - 1, 512).long()]
    out = F.normalize(all_experts(moe.experts, x), dim=-1)          # (n, 512, d)
    cos = torch.einsum("ind,jnd->ij", out, out) / out.shape[1]      # (n, n)
    n = side ** 2
    off = ~torch.eye(n, dtype=torch.bool, device=device)
    D = metrics.torus_distances(side).to(device)
    near = (D > 0) & (D <= 1.0001)
    return {"side": side, "d_model": m.cfg.d_model, "arm": m.cfg.arm, "loss_base": loss["base"],
            "d_zero": loss["zero"] - loss["base"], "d_one_city": loss["one_city"] - loss["base"],
            "cos_all": cos[off].mean().item(), "cos_near": cos[near].mean().item(),
            "shuffle": ck.get("bench", {}).get("shuffle", {}).get("delta_mean")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--corpus", default="corpus_r3")
    ap.add_argument("--windows", type=int, default=64, help="παράθυρα των 256 tokens από το val")
    ap.add_argument("--out", default="cities.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val = load_bin(os.path.join(a.corpus, "val.bin"))
    res = {}
    for d in a.runs:
        for ck in sorted(glob.glob(os.path.join(d, "s*", "*", "epoch1.pt")) + glob.glob(os.path.join(d, "*", "epoch1.pt"))):
            name = os.path.relpath(ck, os.path.dirname(d.rstrip("/")))[:-len("/epoch1.pt")]
            try:
                r = analyse(ck, val, a.windows, device)
            except Exception as e:                       # ένα χαλασμένο checkpoint δεν σταματά τα υπόλοιπα
                print(f"{name}: ΣΦΑΛΜΑ {e}", flush=True)
                continue
            res[name] = r
            print(f"{name:28s} {r['side']:2d}×{r['side']:<2d} d{r['d_model']} · loss {r['loss_base']:.3f} · "
                  f"χωρίς MoE {r['d_zero']:+.3f} · μία πόλη {r['d_one_city']:+.3f} · "
                  f"cos όλοι {r['cos_all']:.2f} γείτονες {r['cos_near']:.2f}", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

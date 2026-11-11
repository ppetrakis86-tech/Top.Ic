"""G1α: ζουμ. Έτοιμα checkpoints, val με k=5 (όπως εκπαιδεύτηκαν) και k=1 (ένας ειδικός). Χωρίς training.

B k=1: μόνο ο νικητής (cross0[:, :1] = ο ίδιος), βάρος 1. A k=1: top-1, βάρος κανονικοποιημένο στο 1.
Το σ του B: model.load_checkpoint (τελικό σ από το checkpoint).

    python -m lightstorm.zoom --runs DIR [DIR ...] --corpus corpus_r3 --out zoom.json
DIR: φάκελος με s*/{A,B}/epoch1.pt.
"""
import argparse
import glob
import json
import os

import torch

from .model import load_checkpoint
from .train import evaluate, load_bin


def set_k(m, k):
    moe = m.moe
    if m.cfg.arm == "A":
        moe.cfg.k = k
    else:
        moe.cross = moe.cross0[:, :k].clone() if k == 1 else moe.cross0.clone()


@torch.no_grad()
def run(ck_path, val, device):
    m, ck = load_checkpoint(ck_path, device)
    bench = ck.get("bench", {})
    out = {"arm": m.cfg.arm, "side": m.cfg.side, "sigma": getattr(m.moe, "sigma", None), "bench_val": bench.get("val_loss")}
    for k in (5, 1):
        set_k(m, k)
        out[f"val_k{k}"] = evaluate(m, val, m.cfg.ctx, device, {})["val_loss"]
    set_k(m, 5)
    out["P"] = out["val_k1"] - out["val_k5"]
    out["repro_ok"] = out["bench_val"] is not None and abs(out["val_k5"] - out["bench_val"]) <= 0.002
    return out


def verdict(res):
    seeds = sorted({k.split("/")[0] for k in res if k.endswith("/A")} & {k.split("/")[0] for k in res if k.endswith("/B")})
    if len(seeds) < 3:
        return {"verdict": "ΧΩΡΙΣ ΕΤΥΜΗΓΟΡΙΑ (λιγότερα από 3 seeds A+B)", "seeds": seeds}
    a = all(res[f"{s}/B"]["P"] <= res[f"{s}/A"]["P"] - 0.02 for s in seeds)
    b = all(res[f"{s}/B"]["val_k1"] <= res[f"{s}/A"]["val_k1"] for s in seeds)
    v = "ΣΤΕΚΕΙ" if a and b else "ΜΙΣΟ" if a else "ΔΕΝ ΣΤΕΚΕΙ"
    return {"verdict": v, "B_drops_less_3of3": a, "B_better_abs_3of3": b, "seeds": seeds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--corpus", default="corpus_r3")
    ap.add_argument("--out", default="zoom.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val = load_bin(os.path.join(a.corpus, "val.bin"))
    allres = {}
    for d in a.runs:
        res = {}
        for ck in sorted(glob.glob(os.path.join(d, "s*", "*", "epoch1.pt"))):
            name = os.path.relpath(os.path.dirname(ck), d)
            try:
                r = run(ck, val, device)
            except Exception as e:
                print(f"{d} {name}: ΣΦΑΛΜΑ {e}", flush=True)
                continue
            res[name] = r
            print(f"{os.path.basename(d.rstrip('/'))} {name:6s} {r['side']}×{r['side']} · bench {r['bench_val']:.4f} · "
                  f"k5 {r['val_k5']:.4f} {'✅' if r['repro_ok'] else '❌ ΔΕΝ ΒΓΑΙΝΕΙ'} · k1 {r['val_k1']:.4f} · Π {r['P']:+.4f}", flush=True)
        allres[d] = {"runs": res, **verdict(res)}
        print(f"→ {d}: {allres[d]['verdict']}", flush=True)
        json.dump(allres, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

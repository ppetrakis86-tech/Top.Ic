"""Πείραμα 11β: βάση ↔ τσίμπημα μετά από ταίριασμα με τις 512 συμμετρίες του τόρου, και πόσοι γείτονες έμειναν γείτονες.

    python -m lightstorm.symmetry --runs runs_r7 --corpus corpus_r3
"""
import argparse
import json
import os

import numpy as np

from . import metrics

MIN_COUNT, TOP, NULLS = 5, 2000, 200


def symmetries(side):
    """Οι 8·side² συμμετρίες του τόρου ως πίνακες κελί → κελί."""
    r, c = np.divmod(np.arange(side * side), side)
    out = []
    for flip in (False, True):
        for rot in range(4):
            rr, cc = (c, r) if flip else (r, c)
            for _ in range(rot):
                rr, cc = cc, (-rr) % side
            for dr in range(side):
                for dc in range(side):
                    out.append(((rr + dr) % side) * side + (cc + dc) % side)
    return np.stack(out)


def best_agreement(hb, hn, S):
    return float((S[:, hb] == hn[None, :]).mean(1).max())


def neighbour_kept(hb, hn, idx, D):
    a, b = np.triu_indices(len(idx), 1)
    near = D[hb[idx][a], hb[idx][b]] <= 1
    return float((D[hn[idx][a], hn[idx][b]][near] <= 1).mean()) if near.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--side", type=int, default=8)
    args = ap.parse_args()
    val = np.fromfile(os.path.join(args.corpus, "val.bin"), dtype=np.uint16)
    count = np.bincount(val, minlength=2 ** 16)
    S = symmetries(args.side)
    D = metrics.torus_distances(args.side).numpy()
    rng = np.random.default_rng(0)
    out = {}
    for k in ("k9", "k5"):
        for s in ("s3", "s4", "s5"):
            hb, hn = (np.load(os.path.join(args.runs, f"{k}_{v}", s, "B", "census_homes.npy"))[-1].astype(np.int64)
                      for v in ("base", "nudge"))
            ok = (count[:len(hb)] >= MIN_COUNT) & (hb >= 0) & (hn >= 0)
            b, n = hb[ok], hn[ok]
            idx = np.argsort(-count[:len(hb)][ok])[:TOP]
            matched = 1 - best_agreement(b, n, S)
            kept = neighbour_kept(b, n, idx, D)
            null_m, null_k = [], []
            for _ in range(NULLS):
                perm = rng.permutation(args.side ** 2)
                null_m.append(1 - best_agreement(perm[b], n, S))
                null_k.append(neighbour_kept(perm[b], n, idx, D))
            out[f"{k}/{s}"] = {"raw_moved": round(float((b != n).mean()), 4), "matched_moved": round(matched, 4),
                               "null_matched_p01": round(float(np.percentile(null_m, 1)), 4),
                               "null_matched_mean": round(float(np.mean(null_m)), 4),
                               "neighbour_kept": round(kept, 4), "null_kept_mean": round(float(np.nanmean(null_k)), 4),
                               "whole_village": bool(matched <= 0.30 and matched < np.percentile(null_m, 1))}
            print(k, s, out[f"{k}/{s}"], flush=True)
    for k in ("k9", "k5"):
        v = [out[f"{k}/{s}"] for s in ("s3", "s4", "s5")]
        print(f"{k}: μέσο matched_moved {np.mean([x['matched_moved'] for x in v]):.4f} · "
              f"μέσο neighbour_kept {np.mean([x['neighbour_kept'] for x in v]):.4f}")
    json.dump(out, open(os.path.join(args.runs, "symmetry.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

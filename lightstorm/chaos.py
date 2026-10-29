"""Πείραμα 11: βάση ↔ τσίμπημα. Πόσα tokens άλλαξαν σπίτι, Δ el↔en, πού χωρίζουν τα traces.

    python -m lightstorm.chaos --runs runs_r7 --corpus corpus_r3
Περιμένει runs/k{9,5}_{base,nudge}/s{3,4,5}/B (census_homes.npy, epoch1.json), trace.tsv στο s*/ και traps.json στο k*_*/.
"""
import argparse
import json
import os

import numpy as np

MIN_COUNT = 5


def first_diff(a, b):
    A, B = open(a).read().splitlines(), open(b).read().splitlines()
    for x, y in zip(A, B):
        sx, sy = x.split("\t"), y.split("\t")
        if sx[1] != sy[1] or sx[2] != sy[2]:
            return int(sx[0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--corpus", required=True)
    args = ap.parse_args()
    val = np.fromfile(os.path.join(args.corpus, "val.bin"), dtype=np.uint16)
    out = {}
    count = np.bincount(val, minlength=2 ** 16)
    for k in ("k9", "k5"):
        traps = {v: json.load(open(os.path.join(args.runs, f"{k}_{v}", "traps.json")))["runs"] for v in ("base", "nudge")}
        for s in ("s3", "s4", "s5"):
            d = {v: os.path.join(args.runs, f"{k}_{v}", s) for v in ("base", "nudge")}
            hb, hn = (np.load(os.path.join(d[v], "B", "census_homes.npy"))[-1].astype(np.int64) for v in ("base", "nudge"))
            ok = (count[:len(hb)] >= MIN_COUNT) & (hb >= 0) & (hn >= 0)
            moved = float((hb[ok] != hn[ok]).mean())
            el = [traps[v][f"{s}/B"]["elen"]["ratio_no_same"] for v in ("base", "nudge")]
            vl = [json.load(open(os.path.join(d[v], "B", "epoch1.json")))["val_loss"] for v in ("base", "nudge")]
            fd = first_diff(*(os.path.join(d[v], "trace.tsv") for v in ("base", "nudge")))
            out[f"{k}/{s}"] = {"moved_share": round(moved, 4), "elen": el, "d_elen": round(abs(el[0] - el[1]), 4),
                               "val": [round(x, 4) for x in vl], "first_diff_step": fd, "identical": fd is None}
            print(k, s, out[f"{k}/{s}"], flush=True)
    for k in ("k9", "k5"):
        m = [v["moved_share"] for kk, v in out.items() if kk.startswith(k) and not v["identical"]]
        e = [v["d_elen"] for kk, v in out.items() if kk.startswith(k) and not v["identical"]]
        print(f"{k}: μέσο moved_share {np.mean(m):.4f} · μέσο |Δ el↔en| {np.mean(e):.4f} · runs {len(m)}")
    json.dump(out, open(os.path.join(args.runs, "chaos.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

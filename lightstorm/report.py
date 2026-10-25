"""Πίνακας του Γύρου 1 από τα <runs>/<σκέλος>/epochN.json. Ο στόχος loss = το τελικό val του A.

    python3 -m lightstorm.report [runs_dir]
"""
import glob
import json
import os
import re
import sys

from . import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(runs, arm):
    out = {}
    for path in glob.glob(os.path.join(runs, arm, "epoch*.json")):
        out[int(re.search(r"epoch(\d+)\.json", path).group(1))] = json.load(open(path))
    return out


def main(runs=os.path.join(ROOT, "runs")):
    bench = {arm: b for arm in ("A", "B", "C1", "C2") if (b := load(runs, arm))}
    last = max(bench["A"])
    show = sorted({1, 2, last} | ({10} if last > 10 else set()))
    target = bench["A"][last]["val_loss"]
    print(f"στόχος loss (τελικό val του A, εποχή {last}): {target:.4f} · εποχές στον πίνακα: {show}\n")
    print("σκέλος | val | tokens μέχρι στόχο | νικ/γειτ/άνεργ | gold ratio | shuffle Δ")
    for arm, b in bench.items():
        ep = [e for e in show if e in b]
        t = metrics.tokens_to_target(b[max(b)]["history"], target)
        val = " → ".join(f"{b[e]['val_loss']:.3f}" for e in ep)
        use = " → ".join(f"{b[e]['winners']}/{b[e]['neighbours']}/{b[e]['idle']}{'⚠️' if b[e]['idle_alarm'] else ''}" for e in ep)
        gold = " → ".join(f"{b[e]['gold_open']['ratio']:.3f}" for e in ep)
        shuf = " → ".join(f"{b[e]['shuffle'].get('delta_mean', 0):+.3f}±{b[e]['shuffle'].get('delta_std', 0):.3f}" for e in ep)
        print(f"{arm} | {val} | {f'{t:,}' if t else '—'} | {use} | {gold} | {shuf}")


if __name__ == "__main__":
    main(*sys.argv[1:])

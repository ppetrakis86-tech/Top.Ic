"""Πίνακας του Γύρου 1 από τα runs/<σκέλος>/epochN.json. Ο στόχος loss = το τελικό val του A."""
import json
import os
import sys

from . import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(runs=os.path.join(ROOT, "runs")):
    bench = {}
    for arm in ("A", "B", "C1", "C2"):
        path = os.path.join(runs, arm, "epoch2.json")
        if os.path.exists(path):
            bench[arm] = {e: json.load(open(os.path.join(runs, arm, f"epoch{e}.json"))) for e in (1, 2)}
    target = bench["A"][2]["val_loss"]
    print(f"στόχος loss (τελικό val του A): {target:.4f}\n")
    print("σκέλος | val εποχή 1 | val εποχή 2 | tokens μέχρι στόχο | νικ/γειτ/άνεργ (εποχή 1 → 2) | gold ratio (1 → 2) | shuffle Δ (1 → 2)")
    for arm, b in bench.items():
        t = metrics.tokens_to_target(b[2]["history"], target)
        use = " → ".join(f"{b[e]['winners']}/{b[e]['neighbours']}/{b[e]['idle']}{' ⚠️' if b[e]['idle_alarm'] else ''}" for e in (1, 2))
        gold = " → ".join(f"{b[e]['gold_open']['ratio']:.3f}" for e in (1, 2))
        shuf = " → ".join(f"{b[e]['shuffle'].get('delta_mean', 0):+.4f}±{b[e]['shuffle'].get('delta_std', 0):.4f}" for e in (1, 2))
        print(f"{arm} | {b[1]['val_loss']:.4f} | {b[2]['val_loss']:.4f} | {t if t else '—'} | {use} | {gold} | {shuf}")


if __name__ == "__main__":
    main(*sys.argv[1:])

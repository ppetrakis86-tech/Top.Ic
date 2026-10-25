"""Πίνακας του Γύρου 1 από τα <runs>/<σκέλος>/epochN.json. Ο στόχος loss = το τελικό val του A.

    python3 -m lightstorm.report [runs_dir]
    python3 -m lightstorm.report runs_e20 runs_e20_s1 runs_e20_s2      # πολλά seeds
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


def seeds(*dirs):
    """Μέσος όρος ± διασπορά στην τελευταία εποχή, πάνω σε πολλά seeds (ένας φάκελος ανά seed)."""
    import numpy as np
    rows = {}
    for d in dirs:
        bench = {arm: b for arm in ("A", "B", "C1", "C2") if (b := load(d, arm))}
        target = bench["A"][max(bench["A"])]["val_loss"]
        for arm, b in bench.items():
            last = b[max(b)]
            rows.setdefault(arm, []).append([last["val_loss"], last["val_loss"] - target,
                                             last["shuffle"].get("delta_mean", 0), last["gold_open"]["ratio"],
                                             last["winners"], last["idle"]])
    print(f"{len(dirs)} seeds · μέσος ± διασπορά στην τελευταία εποχή\n")
    print("σκέλος | val | val − A (ίδιο seed) | shuffle Δ | gold ratio | νικητές | άνεργοι")
    for arm, r in rows.items():
        r = np.array(r)
        cells = [f"{m:.3f} ± {s:.3f}" for m, s in zip(r.mean(0), r.std(0))]
        print(f"{arm} | " + " | ".join(cells))



def frozen(baselines=("runs_e20", "runs_e20_s1", "runs_e20_s2"),
           frozen_dirs=("runs_e20_frozen_s0", "runs_e20_frozen_s1", "runs_e20_frozen_s2")):
    """Πείραμα 1α: κενό από το A του ίδιου seed, για το B και για το B με παγωμένα κλειδιά."""
    import numpy as np
    rows = {"B": [], "B παγωμένο": []}
    for base, fz in zip(baselines, frozen_dirs):
        a = load(base, "A"); b = load(base, "B"); f = load(fz, "B")
        target = a[max(a)]["val_loss"]
        for name, run in (("B", b), ("B παγωμένο", f)):
            last = run[max(run)]
            rows[name].append([last["val_loss"] - target, last["shuffle"]["delta_mean"],
                               last["shuffle"]["delta_std"], last["gold_open"]["ratio"], last["winners"], last["idle"]])
    print("σκέλος | val − A | shuffle Δ | διασπορά shuffle | gold ratio | νικητές | άνεργοι")
    for name, r in rows.items():
        r = np.array(r)
        print(f"{name} | " + " | ".join(f"{m:.3f} ± {s:.3f}" for m, s in zip(r.mean(0), r.std(0))))
    gap = np.mean([x[0] for x in rows["B παγωμένο"]])
    verdict = ("η αστάθεια εξηγεί το μεγαλύτερο μέρος" if gap <= 0.018 else
               "η αστάθεια δεν είναι ο λόγος" if gap >= 0.030 else "μερική εξήγηση, χωρίς συμπέρασμα")
    print(f"\nκενό παγωμένου: {gap:.3f} → {verdict} (κατώφλια από το METRICS.md)")


if __name__ == "__main__":
    if sys.argv[1:] == ["frozen"]:
        frozen()
    elif len(sys.argv) > 2:
        seeds(*sys.argv[1:])
    else:
        main(*sys.argv[1:])

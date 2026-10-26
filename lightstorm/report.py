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


def layer1(l3=("runs_e20", "runs_e20_s1", "runs_e20_s2"),
           l1=("runs_e20_L1_s0", "runs_e20_L1_s1", "runs_e20_L1_s2")):
    """Πείραμα 2: A και B με το MoE στο layer 1, απέναντι στο layer 3. Κενό από το A του ίδιου layer και seed."""
    import numpy as np
    rows = {}
    for layer, dirs in (("L3", l3), ("L1", l1)):
        for d in dirs:
            a = load(d, "A"); target = a[max(a)]["val_loss"]
            for arm in ("A", "B"):
                last = load(d, arm)[max(load(d, arm))]
                sh = last["shuffle"]
                rows.setdefault(f"{arm} {layer}", []).append([last["val_loss"], last["val_loss"] - target,
                    sh.get("delta_mean", 0), sh.get("delta_std", 0), last["gold_open"]["ratio"], last["winners"], last["idle"]])
    print("σκέλος | val | val − A | shuffle Δ | διασπορά shuffle | gold ratio | νικητές | άνεργοι")
    for name, r in rows.items():
        r = np.array(r)
        print(f"{name} | " + " | ".join(f"{m:.3f} ± {s:.3f}" for m, s in zip(r.mean(0), r.std(0))))
    b = np.array(rows["B L1"]); gold, gap, d, ds = b[:, 4].mean(), b[:, 1].mean(), b[:, 2].mean(), b[:, 3].mean()
    real = d >= 10 * ds
    print(f"\nshuffle B L1: {d:.3f} vs 10×{ds:.3f} → {'πραγματική' if real else 'διακοσμητική'} τοπογραφία")
    print(f"gold B L1: {gold:.3f} → " + ("σήμα" if gold <= 0.925 and real else "κανένα σήμα"))
    print(f"κενό B L1: {gap:.3f} → " + ("η θέση εξηγεί την απώλεια" if gap <= 0.018 else
                                       "καμία βελτίωση" if gap >= 0.037 else "χωρίς συμπέρασμα"))
    print("(κατώφλια από το METRICS.md)")


def luck():
    """Πείραμα 2β: διασπορά του B (layer 1) με ίδιο χάρτη (i) και με μόνο τον χάρτη να αλλάζει (ii)."""
    import numpy as np
    series = {"όλα αλλάζουν": [f"runs_e20_L1_s{s}" for s in range(5)],
              "(i) ίδιος χάρτης": [f"runs_e20_L1_samemap_s{s}" for s in range(5)],
              "(ii) μόνο ο χάρτης": [f"runs_e20_L1_onlymap_k{s}" for s in range(5)]}
    stds = {}
    print("σειρά | val | std val | shuffle Δ | διασπορά shuffle | gold ratio")
    for name, dirs in series.items():
        r = []
        for d in dirs:
            b = load(d, "B"); last = b[max(b)]
            r.append([last["val_loss"], last["shuffle"]["delta_mean"], last["shuffle"]["delta_std"], last["gold_open"]["ratio"]])
        r = np.array(r); stds[name] = r[:, 0].std()
        ok = r[:, 1].mean() >= 10 * r[:, 2].mean()
        print(f"{name} | {r[:, 0].mean():.3f} | {stds[name]:.3f} | {r[:, 1].mean():.3f} | {r[:, 2].mean():.3f}"
              f"{'' if ok else ' (δεν μετράει)'} | {r[:, 3].mean():.3f}")
    i, ii = stds["(i) ίδιος χάρτης"], stds["(ii) μόνο ο χάρτης"]
    verdict = ("η τύχη είναι κυρίως στον αρχικό χάρτη" if i <= 0.009 and ii >= 0.012 else
               "η τύχη είναι κυρίως στο δίκτυο και στα batches" if i >= 0.012 and ii <= 0.009 else
               "μικτή πηγή, χωρίς συμπέρασμα")
    print(f"\n(i) {i:.3f} · (ii) {ii:.3f} → {verdict} (κατώφλια από το METRICS.md)")


def scale():
    """Πείραμα 3: A και B σε 4 μεγέθη, seed 0. Κάθε σκέλος στην εποχή με τον καλύτερο val (όχι την τελευταία)."""
    dirs = {"7M": "runs_e20", "25M": "runs_e20_size25M", "58M": "runs_e20_size58M", "110M": "runs_e20_size110M"}
    print("μέγεθος | A καλύτερος val (εποχή) | B καλύτερος val (εποχή) | B − A | shuffle B | gold B | A τελευταίος | B τελευταίος")
    gaps = []
    for size, d in dirs.items():
        a, b = load(d, "A"), load(d, "B")
        ea, eb = min(a, key=lambda e: a[e]["val_loss"]), min(b, key=lambda e: b[e]["val_loss"])
        gap = b[eb]["val_loss"] - a[ea]["val_loss"]
        gaps.append(gap)
        print(f"{size} | {a[ea]['val_loss']:.3f} ({ea}) | {b[eb]['val_loss']:.3f} ({eb}) | {gap:+.3f} | "
              f"{b[eb]['shuffle']['delta_mean']:+.3f}±{b[eb]['shuffle']['delta_std']:.3f} | {b[eb]['gold_open']['ratio']:.3f} | "
              f"{a[max(a)]['val_loss']:.3f} | {b[max(b)]['val_loss']:.3f}")
    shrinks = all(x > y for x, y in zip(gaps, gaps[1:]))
    verdict = ("το μέγεθος κλείνει το κενό (ένδειξη, θέλει seeds)" if shrinks and gaps[-1] <= 0.018 else
               "το μέγεθος δεν βοηθάει το B" if gaps[-1] >= gaps[0] else "χωρίς καθαρή τάση")
    print(f"\nκενό ανά μέγεθος: {' → '.join(f'{g:+.3f}' for g in gaps)} → {verdict} (κατώφλια από το METRICS.md)")


if __name__ == "__main__":
    if sys.argv[1:] == ["frozen"]:
        frozen()
    elif sys.argv[1:] == ["scale"]:
        scale()
    elif sys.argv[1:] == ["luck"]:
        luck()
    elif sys.argv[1:] == ["layer1"]:
        layer1()
    elif sys.argv[1:] == ["layer1", "5"]:
        layer1(l1=tuple(f"runs_e20_L1_s{s}" for s in range(5)))
    elif len(sys.argv) > 2:
        seeds(*sys.argv[1:])
    else:
        main(*sys.argv[1:])

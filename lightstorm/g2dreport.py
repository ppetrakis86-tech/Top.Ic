"""G2δ: ξέχασμα σε ίση μάθηση (Δ) και άπλωμα της νέας πηγής (N_eff). Κανόνες: METRICS.md «G2δ».

Δ ανά seed = Ξ_A@Μ_B − Ξ_B: Ξ του A-όλοι με γραμμική παρεμβολή στα σημεία (Μ, Ξ) του, στο Μ_B του B-όλοι στα 5M.
Αν Μ_B > Μ_A(τέλος): γραμμική προέκταση από τα δύο τελευταία σημεία του A (σημειώνεται «προέκταση»).
N_eff = 1 / Σ win_share² (πέρασμα 200K tokens πριν την ενημέρωση).

    python -m lightstorm.g2dreport --root DRIVE/TopIc/runs_g2d [--old-root DRIVE/TopIc/runs_g2] --out summary_g2d.txt
"""
import argparse
import glob
import json
import os

import numpy as np


def curve(r):
    h = r["history"]
    m = [h[0]["val_new"] - p["val_new"] for p in h]
    x = [p["val_old"] - h[0]["val_old"] for p in h]
    return np.array(m), np.array(x)


def delta(a, b):
    ma, xa = curve(a)
    mb, xb = curve(b)
    m_b, x_b = mb[-1], xb[-1]
    order = np.argsort(ma)
    ma, xa = ma[order], xa[order]
    if m_b <= ma[-1]:
        xa_at, extra = float(np.interp(m_b, ma, xa)), False
    else:
        slope = (xa[-1] - xa[-2]) / (ma[-1] - ma[-2])
        xa_at, extra = float(xa[-1] + slope * (m_b - ma[-1])), True
    return xa_at - x_b, m_b, x_b, xa_at, extra


def neff(r):
    w = np.array(r["win_share"])
    return float(1 / (w ** 2).sum())


def load(root):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "s*"))):
        s = os.path.basename(d)
        try:
            out[s] = {arm: json.load(open(os.path.join(d, "ft", arm, "finetune.json"))) for arm in ("A-all", "B-all")}
        except OSError:
            out[s] = None
    return out


def spread(roots, corpus, manifest, n_tokens=200_000):
    """Αναφορά (#04, 12/11): N_eff ανά πηγή (train, ίδιο πέρασμα 200K tokens) σε κάθε checkpoint βάσης."""
    import torch
    from .finetune import routing, segments
    from .model import load_checkpoint
    from .train import load_bin
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seg = segments(manifest)
    train = load_bin(os.path.join(corpus, "train.bin"))
    names = ("wiki_en", "wiki_el", "greeklish", "uesp_en")
    out = [f"N_eff ανά πηγή (βάση, πριν την ενημέρωση): {' / '.join(names)}"]
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "s*"))):
            for arm in ("A", "B"):
                ck = os.path.join(d, arm, "epoch1.pt")
                if not os.path.exists(ck):
                    continue
                m, _ = load_checkpoint(ck, dev)
                vals = []
                for k in names:
                    w, _ = routing(m, train[seg[k][0]:seg[k][1]], n_tokens, dev)
                    w = (w / w.sum()).numpy()
                    vals.append(1 / (w ** 2).sum())
                out.append(f"{os.path.basename(d):3s} {arm} · " + " / ".join(f"{v:5.2f}" for v in vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--old-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--corpus", default="corpus_r3", help="αν υπάρχει: και N_eff ανά πηγή στα checkpoints βάσης")
    ap.add_argument("--manifest", default="r3_data_manifest.json")
    a = ap.parse_args()
    lines = []
    say = lines.append
    rows = {}
    for tag, root in (("νέο", a.root), ("παλιό", a.old_root)):
        if not root:
            continue
        for s, r in load(root).items():
            if r is None:
                say(f"{tag} {s}: ΛΕΙΠΕΙ")
                continue
            d, m_b, x_b, xa, ex = delta(r["A-all"], r["B-all"])
            nb, na = neff(r["B-all"]), neff(r["A-all"])
            ha, hb = r["A-all"]["history"], r["B-all"]["history"]
            rows[(tag, s)] = (d, nb, na)
            say(f"{tag:5s} {s:3s} · Μ_B {m_b:.4f} · Ξ_B {x_b:.4f} · Ξ_A@Μ_B {xa:.4f}{' (προέκταση)' if ex else ''} · Δ {d:+.4f} · "
                f"N_eff B {nb:5.2f} ({'απλωμένο' if nb >= 10 else 'μαζεμένο'}) · N_eff A {na:5.2f} · "
                f"val παλιό τέλος A {ha[-1]['val_old']:.4f} B {hb[-1]['val_old']:.4f} · νέο A {ha[-1]['val_new']:.4f} B {hb[-1]['val_new']:.4f}")
    new = [(s, v) for (t, s), v in rows.items() if t == "νέο"]
    if len(new) == 6:
        q1 = sum(d >= 0.02 for _, (d, _, _) in new)
        say(f"ΕΡΩΤΗΣΗ 1 (κέρδος): Δ ≥ 0,02 σε {q1}/6 → {'ΣΤΕΚΕΙ' if q1 >= 5 else 'ΔΕΝ ΣΤΕΚΕΙ'}")
        cats = [nb >= 10 for _, (_, nb, _) in new]
        if all(cats) or not any(cats):
            say("ΕΡΩΤΗΣΗ 2 (εξήγηση): ΧΩΡΙΣ ΣΥΜΠΕΡΑΣΜΑ (όλα τα seeds στην ίδια κατηγορία)")
        else:
            ok = sum((nb >= 10) == (d >= 0.02) for _, (d, nb, _) in new)
            say(f"ΕΡΩΤΗΣΗ 2 (εξήγηση): σωστή πρόβλεψη σε {ok}/6 (απλωμένα {sum(cats)}, μαζεμένα {6 - sum(cats)}) → "
                f"{'ΣΤΕΚΕΙ' if ok >= 5 else 'ΔΕΝ ΣΤΕΚΕΙ'}")
    else:
        say(f"ΧΩΡΙΣ ΕΤΥΜΗΓΟΡΙΑ: {len(new)}/6 νέα seeds")
    allv = list(rows.values())
    if len(allv) >= 3:
        d, nb = np.array([v[0] for v in allv]), np.array([v[1] for v in allv])
        say(f"Αναφορά: συσχέτιση Pearson N_eff(B)–Δ σε {len(allv)} seeds = {np.corrcoef(nb, d)[0, 1]:+.3f}")
    if os.path.exists(os.path.join(a.corpus, "train.bin")) and os.path.exists(a.manifest):
        try:
            lines += spread([r for r in (a.old_root, a.root) if r], a.corpus, a.manifest)
        except Exception as e:                    # η αναφορά δεν ρίχνει την ετυμηγορία
            lines.append(f"N_eff ανά πηγή: ΣΦΑΛΜΑ {e}")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()

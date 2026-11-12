"""G2: πίνακας και ετυμηγορία από τα finetune.json (κανόνες: METRICS.md «G2»).

    python -m lightstorm.g2report --root DRIVE/TopIc/runs_g2 --out summary_g2.txt
ROOT: s{0,1,2}/ft/{A-local,B-local,B-shuffle,A-all,B-all}/finetune.json
"""
import argparse
import glob
import json
import os

ARMS = ("A-local", "B-local", "B-shuffle", "A-all", "B-all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.root, "s*")))
    R, lines = {}, []
    say = lines.append
    for s in seeds:
        for arm in ARMS:
            f = os.path.join(a.root, s, "ft", arm, "finetune.json")
            if os.path.exists(f):
                R[s, arm] = json.load(open(f))
    say("seed  σκέλος     Μ        Ξ        νέο τέλος  παλιό τέλος  ειδικοί            νίκες/μάζα τους  σταυρός B")
    for s in seeds:
        for arm in ARMS:
            r = R.get((s, arm))
            if r is None:
                say(f"{s:5s} {arm:9s} ΛΕΙΠΕΙ")
                continue
            h = r["history"][-1]
            cw = r.get("cross_win_share")
            say(f"{s:5s} {arm:9s} {r['M']:+.4f}  {r['X']:+.4f}  {h['val_new']:.4f}     {h['val_old']:.4f}       "
                f"{str(r['chosen'][:5]):18s} {r['chosen_win_share']:.0%}/{r['chosen_mass_share']:.0%}"
                + (f"          {cw:.0%}" if cw is not None else ""))
    ok_full, shuf, clean = [], [], []
    for s in seeds:
        A, B, Sh = R.get((s, "A-local")), R.get((s, "B-local")), R.get((s, "B-shuffle"))
        if not (A and B):
            continue
        dM, dX = B["M"] - A["M"], A["X"] - B["X"]            # θετικό = το B καλύτερο
        ok = dM >= -0.01 and dX >= -0.01 and (dM >= 0.02 or dX >= 0.02)
        ok_full.append(ok)
        say(f"{s}: B−A Μ {dM:+.4f} · Ξ (A−B) {dX:+.4f} → {'✅' if ok else '❌'}")
        if Sh:
            shuf.append(abs(Sh["M"] - B["M"]) <= 0.01 and abs(Sh["X"] - B["X"]) <= 0.01)
        hb, ha = B["history"][-1], A["history"][-1]
        clean.append(hb["val_new"] <= ha["val_new"] and hb["val_old"] <= ha["val_old"] + 0.01)
    n = sum(ok_full)
    v = "ΣΤΕΚΕΙ" if len(ok_full) == 3 and n == 3 else "ΜΙΣΟ" if len(ok_full) == 3 and n == 2 else \
        "ΔΕΝ ΣΤΕΚΕΙ" if len(ok_full) == 3 else f"ΧΩΡΙΣ ΕΤΥΜΗΓΟΡΙΑ ({len(ok_full)} seeds)"
    say(f"ΕΤΥΜΗΓΟΡΙΑ G2: {v} ({n}/{len(ok_full)} seeds)")
    if len(shuf) == 3:
        say("Τεστ ανακατέματος: " + ("Η ΓΕΙΤΟΝΙΑ ΕΙΝΑΙ ΔΙΑΚΟΣΜΗΤΙΚΗ (±0,01 σε 3/3)" if all(shuf) else f"η γειτονιά μετράει ({3 - sum(shuf)}/3 seeds έξω από ±0,01)"))
    say(f"Αναφορά, καθαρό κέρδος B-τοπικό: {sum(clean)}/{len(clean)} seeds")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()

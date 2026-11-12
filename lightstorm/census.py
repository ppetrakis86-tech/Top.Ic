"""Πείραμα 8: απογραφή του router κατά τη διάρκεια του training. Διαγνωστικό, χωρίς ετυμηγορία.

Σε κάθε αξιολόγηση (και στο βήμα 0), πάνω σε όλο το val:
- πληθυσμός: πόσα tokens κερδίζει κάθε ειδικός, συνολικά και ανά πηγή του val
- τι κρατάει: τα 8 πιο συχνά tokens κάθε ειδικού
- μετακόμιση: πόσα tokens άλλαξαν σπίτι από την προηγούμενη απογραφή, και πόσο μακριά στον τόρο
- gold: ίδιος ειδικός, λόγος χωρίς τον ίδιο ειδικό (όπως στις παγίδες, χωρίς τις 1000 τοποθετήσεις)
Γράφει μία γραμμή JSON ανά απογραφή στο census.jsonl και τα σπίτια όλων των tokens στο census_homes.npy.
"""
import json
import os

import numpy as np

from . import metrics

TOP = 8
MIN_COUNT = 5           # για τη μετακόμιση μετράνε μόνο tokens με ≥ 5 εμφανίσεις στο val


class Census:
    def __init__(self, run_dir, side, vocab, gold_sets, segments=None, state=None):
        self.path = os.path.join(run_dir, "census.jsonl")
        self.homes_path = os.path.join(run_dir, "census_homes.npy")
        self.side, self.vocab, self.gold = side, vocab, gold_sets
        self.segments = segments or []          # [(όνομα, πλήθος tokens στο val)], με τη σειρά του val.bin
        self.D = metrics.torus_distances(side).numpy()
        self.prev, self.homes = None, []
        if state is None:
            open(self.path, "w").close()
        else:                                   # συνέχεια από checkpoint: κρατάμε ό,τι γράφτηκε ως το βήμα του
            self.prev, self.homes, self.step = state["prev"], list(state["homes"]), state["step"]
            lines = [l for l in open(self.path, encoding="utf-8") if json.loads(l)["step"] <= state["step"]]
            open(self.path, "w", encoding="utf-8").writelines(lines)

    def state(self):
        return {"prev": self.prev, "homes": self.homes, "step": self.step}

    def __call__(self, step, seen, win, toks):
        from .traps import ratio                # εδώ, για να μη γίνει κυκλικό import με το train
        n = self.side ** 2
        win, toks = np.asarray(win), np.asarray(toks)
        hist = np.zeros((self.vocab, n), dtype=np.int64)
        np.add.at(hist, (toks, win), 1)
        count = hist.sum(1)
        home = hist.argmax(1)
        home[count == 0] = -1
        self.step = step
        out = {"step": step, "tokens": seen, "population": np.bincount(win, minlength=n).tolist()}

        p = np.asarray(out["population"]) / max(1, len(win))
        out["entropy_bits"] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        if self.segments:
            out["by_source"], start = {}, 0
            for name, length in self.segments:
                out["by_source"][name] = np.bincount(win[start:start + length], minlength=n).tolist()
                start += length
        out["top"] = [[[int(t), int(hist[t, e])] for t in np.argsort(-hist[:, e])[:TOP] if hist[t, e] > 0]
                      for e in range(n)]

        if self.prev is not None:
            ok = (count >= MIN_COUNT) & (self.prev >= 0) & (home >= 0)
            moved = ok & (home != self.prev)
            out["moved_tokens"] = int(moved.sum())
            out["moved_share"] = round(float(moved.sum() / max(1, ok.sum())), 4)
            out["moved_share_weighted"] = round(float(count[moved].sum() / max(1, count[ok].sum())), 4)
            out["moved_distance"] = round(float(self.D[self.prev[moved], home[moved]].mean()), 3) if moved.any() else 0.0

        out["gold"] = {}
        for name, pairs in self.gold.items():
            P = [(g, e) for g, e in pairs if home[g] >= 0 and home[e] >= 0]
            if not P:
                continue
            hg, he = home[[g for g, _ in P]], home[[e for _, e in P]]
            diff = hg != he
            rnd = he[np.random.default_rng(metrics.GOLD_RANDOM_SEED).permutation(len(he))]
            out["gold"][name] = {"pairs": len(P), "same": int((~diff).sum()), "ratio": round(float(ratio(self.D, hg, he)), 4),
                                 "ratio_no_same": round(float(self.D[hg[diff], he[diff]].mean()
                                                              / self.D[hg[hg != rnd], rnd[hg != rnd]].mean()), 4)
                                 if diff.any() else None}

        self.prev = home
        self.homes.append(home.astype(np.int16))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        np.save(self.homes_path, np.stack(self.homes))
        return out

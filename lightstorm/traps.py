"""Παγίδες: όλοι οι έλεγχοι μαζί, πάνω σε runs που υπάρχουν ήδη (κανένα νέο training).

Κάθε παγίδα απαντά «μήπως το αποτέλεσμα βγαίνει από κάτι άλλο;». Κριτήρια στο METRICS.md («Παγίδες»).
Μόνο ανοιχτά gold. Τα σφραγισμένα αρχεία ΔΕΝ ανοίγουν· ελέγχεται μόνο το sha256 τους.

    python3 -m lightstorm.traps --runs runs_r2 --corpus corpus_r2 --out runs_r2/traps.json
"""
import argparse
import glob
import hashlib
import json
import math
import os

import numpy as np
import torch

from . import metrics
from .data import sha256
from .model import Config, Lightstorm
from .train import load_bin, read_gold
from .wiki import ROOT

N_LAYOUTS = 1000        # τυχαίες τοποθετήσεις των ειδικών στον τόρο (παγίδα 1)
N_SHUFFLES = 20         # ανακατέματα συντρόφων ανά πεμπτημόριο συχνότητας (παγίδα 4)
WINDOW = 64             # tokens ανά παράθυρο για τον έλεγχο επικάλυψης train/val (παγίδα 6)
STRIDE = 32


@torch.no_grad()
def homes(ck_path, val, device):
    """Διεύθυνση κάθε token (mode του νικητή) πάνω στο val, όπως στο train.evaluate."""
    ck = torch.load(ck_path, map_location=device)
    m = Lightstorm(Config(**ck["cfg"])).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    ctx, n = m.cfg.ctx, (len(val) - 1) // m.cfg.ctx
    xs, ys = val[:n * ctx].view(n, ctx), val[1:n * ctx + 1].view(n, ctx)
    win, tok = [], []
    for i in range(0, n, 32):
        _, _, aux = m(xs[i:i + 32].to(device), ys[i:i + 32].to(device))
        win.append(aux["winner"].cpu())
        tok.append(xs[i:i + 32].reshape(-1))
    win, tok = torch.cat(win).numpy(), torch.cat(tok).numpy()
    side = m.cfg.side
    return metrics.home_positions(tok, win, m.cfg.vocab, side ** 2), side, np.bincount(win, minlength=side ** 2)


def ratio(D, hg, he, rng_seed=metrics.GOLD_RANDOM_SEED):
    rand = he[np.random.default_rng(rng_seed).permutation(len(he))]
    return D[hg, he].mean() / D[hg, rand].mean()


def geometry_traps(home, side, pairs, counts):
    D = metrics.torus_distances(side).numpy()
    P = [(g, e) for g, e in pairs if home[g] >= 0 and home[e] >= 0]
    g, e = np.array([p[0] for p in P]), np.array([p[1] for p in P])
    hg, he = home[g], home[e]
    out = {"pairs": len(P), "ratio": float(ratio(D, hg, he))}

    # 1. Τοποθέτηση: ίδια δρομολόγηση, ειδικοί σε τυχαίες θέσεις του τόρου. Το «ίδιος ειδικός» δεν αλλάζει,
    #    οπότε ό,τι κερδίζει η αληθινή τοποθέτηση πάνω από τις τυχαίες είναι μόνο γεωμετρία.
    rng = np.random.default_rng(0)
    null = []
    for _ in range(N_LAYOUTS):
        pi = rng.permutation(side ** 2)
        null.append(ratio(D, pi[hg], pi[he]))
    null = np.array(null)
    out["layout_null_mean"] = float(null.mean())
    out["layout_p"] = float((1 + (null <= out["ratio"]).sum()) / (1 + N_LAYOUTS))

    # 2. Χωρίς τα ζευγάρια στον ίδιο ειδικό (και στα αληθινά και στα τυχαία).
    diff = hg != he
    rand = he[np.random.default_rng(metrics.GOLD_RANDOM_SEED).permutation(len(he))]
    out["ratio_no_same"] = float(D[hg[diff], he[diff]].mean() / D[hg[hg != rand], rand[hg != rand]].mean())

    # 3. Ίδιος ειδικός: πόσα ζευγάρια, και πόσα θα περίμενε κανείς από τύχη (ίδιες λέξεις, ανακατεμένοι σύντροφοι).
    out["same"] = int((~diff).sum())
    chance = [(hg == he[np.random.default_rng(s).permutation(len(he))]).sum() for s in range(N_SHUFFLES)]
    out["same_chance"] = float(np.mean(chance))

    # 4. Συχνότητα: σύντροφος από το ίδιο πεμπτημόριο συχνότητας.
    fe = np.log1p(counts[e])
    bins = np.digitize(fe, np.quantile(fe, [.2, .4, .6, .8]))
    rs = []
    for s in range(N_SHUFFLES):
        r = np.random.default_rng(s)
        er = he.copy()
        for b in range(5):
            idx = np.where(bins == b)[0]
            er[idx] = he[idx][r.permutation(len(idx))]
        rs.append(D[hg, he].mean() / D[hg, er].mean())
    out["ratio_freq"] = float(np.mean(rs))
    return out


def run_traps(run_dir, arm, val, golds, counts, device):
    ck = os.path.join(run_dir, arm, "epoch1.pt")
    ev = json.load(open(os.path.join(run_dir, arm, "epoch1.json")))
    home, side, wins = homes(ck, val, device)
    out = {name: geometry_traps(home, side, pairs, counts) for name, pairs in golds.items()}
    # 7. Training: καθαρό τέλος, χωρίς NaN, το loss δεν ανεβαίνει στο τέλος, ειδικοί σε χρήση.
    hist = [l for _, l in ev["history"]]
    p = wins / wins.sum()
    out["training"] = {"nan": any(math.isnan(l) for l in hist), "last_evals": [round(l, 4) for l in hist[-4:]],
                       "rising_at_end": hist[-1] > min(hist[-4:]) + 0.01, "winners": ev["winners"],
                       "idle": ev["idle"], "winner_entropy_bits": float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
                       "sigma": ev.get("sigma"), "eta": ev.get("eta")}
    return out


def data_traps(corpus, manifest_path, model_vocab):
    out = {}
    man = json.load(open(manifest_path))
    # 5. Σφραγισμένα: μόνο το hash, το περιεχόμενο δεν διαβάζεται.
    out["sealed_ok"] = {
        "gold_sealed.tsv": sha256(os.path.join(corpus, "gold_sealed.tsv")) == man["gold"]["greeklish"]["sealed_sha256"],
        "gold_elen_sealed.tsv": sha256(os.path.join(corpus, "gold_elen_sealed.tsv")) == man["gold"]["elen"]["sealed_sha256"]}
    # 6. Επικάλυψη train/val: κομμάτια του val που υπάρχουν αυτούσια στο train. Το train κόβεται ανά STRIDE και το val
    #    ελέγχεται σε κάθε θέση, οπότε πιάνεται κάθε κοινό κομμάτι μήκους >= WINDOW + STRIDE - 1 tokens.
    train = np.fromfile(os.path.join(corpus, "train.bin"), dtype=np.uint16)
    val = np.fromfile(os.path.join(corpus, "val.bin"), dtype=np.uint16)
    key = lambda a: hashlib.blake2b(a.tobytes(), digest_size=8).digest()
    tr = {key(train[i:i + WINDOW]) for i in range(0, len(train) - WINDOW, STRIDE)}
    hit = np.zeros(len(val), dtype=bool)
    for i in range(len(val) - WINDOW):
        if key(val[i:i + WINDOW]) in tr:
            hit[i:i + WINDOW] = True
    out["overlap_min_tokens"] = WINDOW + STRIDE - 1
    out["val_tokens_in_train_pct"] = round(100 * float(hit.mean()), 3)
    # 8. Tokens: ids μέσα στο vocab του μοντέλου.
    out["max_id_ok"] = int(max(train.max(), val.max())) < model_vocab
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs_r2"))
    ap.add_argument("--corpus", default=os.path.join(ROOT, "corpus_r2"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "r2_data_manifest.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs_r2", "traps.json"))
    ap.add_argument("--data-only", type=int, default=0, metavar="VOCAB",
                    help="πείραμα 8: μόνο οι παγίδες δεδομένων (πύλη πριν το training), με αυτό το vocab")
    args = ap.parse_args()
    if args.data_only:
        res = data_traps(args.corpus, args.manifest, args.data_only)
        print(json.dumps(res, ensure_ascii=False), flush=True)
        with open(args.out, "w", encoding="utf-8") as fo:
            json.dump(res, fo, indent=1, ensure_ascii=False)
        ok = all(res["sealed_ok"].values()) and res["max_id_ok"] and res["val_tokens_in_train_pct"] <= 1.0
        raise SystemExit(0 if ok else 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val = load_bin(os.path.join(args.corpus, "val.bin"))
    counts = np.bincount(val.numpy(), minlength=2 ** 16)
    golds = {"greeklish": read_gold(os.path.join(args.corpus, "gold_open.tsv")),
             "elen": read_gold(os.path.join(args.corpus, "gold_elen.tsv"))}
    res = {"runs": {}}
    vocab = None
    for run in sorted(glob.glob(os.path.join(args.runs, "s*"))):
        for arm in ("A", "B", "E"):
            if not os.path.exists(os.path.join(run, arm, "epoch1.pt")):
                continue
            r = run_traps(run, arm, val, golds, counts, device)
            res["runs"][f"{os.path.basename(run)}/{arm}"] = r
            vocab = vocab or torch.load(os.path.join(run, arm, "epoch1.pt"), map_location="cpu")["cfg"]["vocab"]
            print(os.path.basename(run), arm, json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                                              for kk, vv in v.items()} for k, v in r.items()},
                                                         ensure_ascii=False), flush=True)
    res["data"] = data_traps(args.corpus, args.manifest, vocab)
    print(json.dumps(res["data"], ensure_ascii=False), flush=True)
    with open(args.out, "w", encoding="utf-8") as fo:
        json.dump(res, fo, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

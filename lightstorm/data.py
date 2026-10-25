"""Γ1·2 Dataset: encode με τα παγωμένα merges, split train/val, σφράγισμα gold pairs.

Μόνο encode. Ο tokenizer δεν εκπαιδεύεται ποτέ εδώ, και το merges.json ελέγχεται με md5
πριν και μετά. Όλα τα αρχεία εξόδου πάνε στο corpus/, που δεν ανεβαίνει ποτέ στο GitHub.
Στο repo μένει μόνο το data_manifest.json: μετρήσεις και hashes, κανένα κείμενο.

    python3 -m lightstorm.data            # από τη ρίζα του repo
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenizer import Tokenizer  # noqa: E402

BLOCK = 2048        # tokens ανά μπλοκ του split
VAL_EVERY = 10      # κάθε 10ο μπλοκ πάει στο val -> 10%, απλωμένο σε όλες τις 90 ημέρες
SEAL_SALT = "Top.Ic-round1-seal"


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def split_blocks(ids, block=BLOCK, val_every=VAL_EVERY):
    """Κόβει το ρεύμα σε μπλοκ και στέλνει κάθε val_every-οστό στο val."""
    train, val = [], []
    for n, start in enumerate(range(0, len(ids), block)):
        (val if n % val_every == val_every - 1 else train).append(ids[start:start + block])
    cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0, np.uint16)
    return cat(train), cat(val)


def is_sealed(greek, greeklish):
    """Σταθερή, ντετερμινιστική απόφαση: περίπου τα μισά ζευγάρια κλειδώνονται."""
    h = hashlib.sha256(f"{SEAL_SALT}\t{greek}\t{greeklish}".encode("utf-8")).digest()
    return h[0] % 2 == 1


def read_gold(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    header, rows = lines[0], [l.split("\t") for l in lines[1:] if l.strip()]
    return header, rows


def check_gold(tok, rows):
    """Κάθε id πρέπει να είναι η λέξη, με ή χωρίς το ▁ μπροστά."""
    bad = []
    for g, l, gi, li in rows:
        for w, i in ((g, int(gi)), (l, int(li))):
            if tok.vocab[i] not in (w, "▁" + w):
                bad.append((w, i, tok.vocab[i]))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    ap.add_argument("--model", default=os.path.join(ROOT, "merges.json"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data_manifest.json"))
    args = ap.parse_args()

    clean = os.path.join(args.corpus, "CLEAN.txt")
    gold_path = os.path.join(args.corpus, "gold_pairs.tsv")
    merges_md5 = md5(args.model)

    tok = Tokenizer.load(args.model)
    text = open(clean, encoding="utf-8").read()
    ids = tok.encode(text)
    assert tok.decode(ids) == text, "round-trip απέτυχε"
    assert len(tok.vocab) < 2 ** 16
    ids = np.array(ids, dtype=np.uint16)

    train, val = split_blocks(ids)
    train.tofile(os.path.join(args.corpus, "train.bin"))
    val.tofile(os.path.join(args.corpus, "val.bin"))

    header, rows = read_gold(gold_path)
    bad = check_gold(tok, rows)
    assert not bad, f"gold ids δεν ταιριάζουν: {bad[:5]}"
    open_rows = [r for r in rows if not is_sealed(r[0], r[1])]
    sealed_rows = [r for r in rows if is_sealed(r[0], r[1])]
    for name, part in (("gold_open.tsv", open_rows), ("gold_sealed.tsv", sealed_rows)):
        with open(os.path.join(args.corpus, name), "w", encoding="utf-8") as f:
            f.write("\n".join([header] + ["\t".join(r) for r in part]) + "\n")

    assert md5(args.model) == merges_md5, "το merges.json άλλαξε!"
    manifest = {
        "merges_md5": merges_md5, "merges": len(tok.merges), "vocab": len(tok.vocab),
        "clean_md5": md5(clean), "chars": len(text),
        "tokens": int(len(ids)), "train_tokens": int(len(train)), "val_tokens": int(len(val)),
        "split": {"block": BLOCK, "val_every": VAL_EVERY},
        "gold_md5": md5(gold_path), "gold_pairs": len(rows),
        "gold_open": len(open_rows), "gold_sealed": len(sealed_rows),
        # δέσμευση: αν το κλειστό αρχείο αλλάξει, το hash δεν θα ταιριάζει
        "gold_sealed_sha256": sha256(os.path.join(args.corpus, "gold_sealed.tsv")),
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

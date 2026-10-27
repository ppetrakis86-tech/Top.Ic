"""Greeklish Wikipedia: άρθρα της ελληνικής Wikipedia σε greeklish, με τη γραφή του Πέτρου.

Ποια άρθρα: ΟΧΙ αυτά που μπήκαν στο πείραμα 6 (τα πρώτα N, από το wiki_manifest.json), και από τα υπόλοιπα
μόνο το μισό (hash του id). Το άλλο μισό μένει ελληνικό. Έτσι κανένα άρθρο δεν υπάρχει και στις δύο γραφές,
οπότε η «γάτα» και η «gata» δεν μοιράζονται ποτέ τα ίδια συμφραζόμενα (αλλιώς το τεστ θα ήταν τετριμμένο).

Γράφει corpus_wiki/parquet_greeklish/ (gitignored, μένει στο PC) και greeklish_manifest.json (μόνο μετρήσεις).

    python3 -m lightstorm.greeklish_wiki --clean corpus/CLEAN.txt
"""
import argparse
import glob
import hashlib
import json
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

from .greeklish import Greeklish, his_vocab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_SALT = "Top.Ic-greeklish-split"


def to_greeklish(art_id):
    return hashlib.sha256(f"{SPLIT_SALT}\t{art_id}".encode()).digest()[0] % 2 == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "corpus_wiki", "parquet_el"))
    ap.add_argument("--out", default=os.path.join(ROOT, "corpus_wiki", "parquet_greeklish"))
    ap.add_argument("--clean", default=os.path.join(ROOT, "corpus", "CLEAN.txt"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "wiki_manifest.json"))
    ap.add_argument("--rows", type=int, default=20000, help="άρθρα ανά αρχείο parquet")
    args = ap.parse_args()

    skip = json.load(open(args.manifest))["per_lang"]["el"]["articles"]   # τα άρθρα του πειράματος 6
    g = Greeklish(his_vocab(args.clean))
    os.makedirs(args.out, exist_ok=True)
    t0, n_seen, n_out, n_chars, part, buf = time.time(), 0, 0, 0, 0, {"id": [], "title": [], "text": []}

    def flush():
        nonlocal part, buf
        if buf["id"]:
            pq.write_table(pa.table(buf), os.path.join(args.out, f"greeklish-{part:03d}.parquet"))
            part += 1
            buf = {"id": [], "title": [], "text": []}

    for path in sorted(glob.glob(os.path.join(args.src, "*.parquet"))):
        table = pq.read_table(path, columns=["id", "title", "text"])
        for art_id, title, text in zip(*(table.column(c).to_pylist() for c in ("id", "title", "text"))):
            n_seen += 1
            if n_seen <= skip or not to_greeklish(art_id):
                continue
            buf["id"].append(art_id)
            buf["title"].append(g.text(title))
            buf["text"].append(g.text(text))
            n_out += 1
            n_chars += len(text)
            if len(buf["id"]) == args.rows:
                flush()
                print(f"{n_out:,} άρθρα · {n_chars / 1e6:.0f}M χαρακτήρες · {time.time() - t0:.0f}s", flush=True)
    flush()
    stats = {"source": "el wikipedia 20231101, άρθρα μετά τα πρώτα %d, μισά με hash" % skip,
             "articles": n_out, "chars": n_chars, "files": part,
             "words_his_spelling": g.stats["his"], "words_rules": g.stats["rule"],
             "clean_vocab_words": len(g.vocab)}
    json.dump(stats, open(os.path.join(ROOT, "greeklish_manifest.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

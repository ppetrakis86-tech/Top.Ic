"""Γύρος 2: νέος tokenizer από το μηδέν πάνω στο μείγμα (κατοχυρωμένο στο METRICS.md, «Γύρος 2: tokenizer»).

Έξι ίσα μέρη: wiki_el, wiki_en, greeklish, uesp_en, uesp_el (25M χαρακτήρες το καθένα) και το CLEAN.txt με βάρος.
Τα κείμενα με is_val() μένουν έξω και τα χρησιμοποιεί το round2_eval.py. Τα merges γράφονται στο corpus_wiki/round2/
(μένει στο PC) και στο repo μπαίνει μόνο το round2_tok_manifest.json (μετρήσεις, κανένα κείμενο).

    python3 -m lightstorm.round2_tok --vocab 16384
"""
import argparse
import glob
import json
import os
import time
from collections import Counter
from multiprocessing import Pool

import pyarrow.parquet as pq

from .data import md5
from .greeklish_wiki import to_greeklish
from .wiki import ROOT, is_val
from tokenizer import Tokenizer, pre_tokenize, train_bpe_freqs

W = os.path.join(ROOT, "corpus_wiki")


def parquet_docs(pattern, key_col, keep=None):
    """(κλειδί, κείμενο) από parquet. keep(i, key) αποφασίζει ποια μένουν (π.χ. μόνο τα ελληνικά άρθρα)."""
    i = 0
    for path in sorted(glob.glob(pattern)):
        t = pq.read_table(path, columns=[key_col, "text"])
        for key, text in zip(t.column(key_col).to_pylist(), t.column("text").to_pylist()):
            i += 1
            if keep is None or keep(i, key):
                yield str(key), text


def sources():
    skip = json.load(open(os.path.join(ROOT, "wiki_manifest.json")))["per_lang"]["el"]["articles"]
    greek = lambda i, key: i <= skip or not to_greeklish(key)   # ό,τι δεν έγινε greeklish
    return {
        "wiki_el": (os.path.join(W, "parquet_el", "*.parquet"), "id", greek),
        "wiki_en": (os.path.join(W, "parquet_simple", "*.parquet"), "id", None),
        "greeklish": (os.path.join(W, "parquet_greeklish", "*.parquet"), "id", None),
        "uesp_en": (os.path.join(W, "uesp", "parquet_en", "*.parquet"), "title", None),
        "uesp_el": (os.path.join(W, "uesp", "parquet_el", "*.parquet"), "title", None),
    }


def count(texts):
    c = Counter()
    for t in texts:
        c.update(pre_tokenize(t + "\n\n"))
    return c


def batches(docs, budget, st, size=200):
    buf = []
    for key, text in docs:
        if is_val(key):
            st["held_out"] += 1
            continue
        buf.append(text)
        st["docs"] += 1
        st["chars"] += len(text)
        if len(buf) == size:
            yield buf
            buf = []
        if st["chars"] >= budget:
            break
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16384, help="συνολικό μέγεθος vocab (μαζί με τα 256 bytes)")
    ap.add_argument("--budget", type=float, default=25e6, help="χαρακτήρες ανά πηγή")
    ap.add_argument("--clean", default=os.path.join(ROOT, "corpus", "CLEAN.txt"))
    ap.add_argument("--min-char", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(W, "round2", "merges.json"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "round2_tok_manifest.json"))
    args = ap.parse_args()

    t0, total, stats = time.time(), Counter(), {}
    with Pool(args.workers) as pool:
        for name, (pattern, key_col, keep) in sources().items():
            st = {"docs": 0, "held_out": 0, "chars": 0}
            c = Counter()
            for part in pool.imap_unordered(count, batches(parquet_docs(pattern, key_col, keep), args.budget, st)):
                c.update(part)
            st["words"], st["types"] = sum(c.values()), len(c)
            c = Counter({w: f for w, f in c.items() if f >= 2})
            st["types_kept"] = len(c)
            total.update(c)
            stats[name] = st
            print(f"{name}: {st} · {time.time() - t0:.0f}s", flush=True)
            del c

    clean = open(args.clean, encoding="utf-8").read()
    weight = max(1, round(args.budget / len(clean)))
    c = count([clean])
    clean_chars = set(clean)
    stats["clean"] = {"chars": len(clean), "weight": weight, "words": sum(c.values()), "types": len(c)}
    c = Counter({w: f * weight for w, f in c.items() if f >= 2})
    stats["clean"]["types_kept"] = len(c)
    total.update(c)
    del c
    print(f"clean: {stats['clean']} · μείγμα {len(total):,} λέξεις-τύποι · {time.time() - t0:.0f}s", flush=True)

    chars = Counter()
    for w, f in total.items():
        for ch in w:
            chars[ch] += f
    alphabet = sorted(clean_chars | {ch for ch, f in chars.items() if f >= args.min_char})
    # Λέξεις με χαρακτήρα έξω από το alphabet φεύγουν από το training: αλλιώς θα μάθαινε merges πάνω σε bytes,
    # και ένα token σαν «<e2><80>» θα έβγαινε στο decode ως κείμενο.
    keep = set(alphabet)
    n_types = len(total)
    total = Counter({w: f for w, f in total.items() if keep.issuperset(w)})
    print(f"{n_types - len(total):,} λέξεις-τύποι με σπάνιους χαρακτήρες έξω από το training", flush=True)
    tok = Tokenizer()
    tok.add_alphabet(alphabet)
    n_merges = args.vocab - len(tok.vocab)
    print(f"alphabet {len(alphabet)} · {n_merges} merges για vocab {args.vocab}", flush=True)
    train_bpe_freqs(tok, total, n_merges)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tok.save(args.out)

    manifest = {"vocab": len(tok.vocab), "merges": len(tok.merges), "alphabet": len(alphabet),
                "budget_chars": args.budget, "min_char": args.min_char, "min_word_count": 2,
                "clean_md5": md5(args.clean), "merges_md5": md5(args.out), "sources": stats,
                "mix_types": n_types, "train_types": len(total), "seconds": round(time.time() - t0)}
    with open(args.manifest, "w", encoding="utf-8") as fo:
        json.dump(manifest, fo, indent=1, ensure_ascii=False)
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

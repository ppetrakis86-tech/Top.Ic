"""Γύρος 2: μέτρηση του νέου tokenizer έναντι του Γύρου 1 (με τον κανόνα των ψηφίων), στα κρατημένα κείμενα.

Κριτήρια στο METRICS.md («Γύρος 2: tokenizer»). Gold: μόνο τα ανοιχτά μισά (gold_open.tsv, gold_elen.tsv).

    python3 -m lightstorm.round2_eval --new corpus_wiki/round2/merges.json
"""
import argparse
import json
import os
import re

from .round2_tok import parquet_docs, sources, W
from .wiki import ROOT, is_val
from tokenizer import BYTE_TOKENS, SPACE, Tokenizer

WORD = re.compile(r"[^\W\d_]+")


def held_out(name, cap):
    pattern, key_col, keep = sources()[name]
    out, n = [], 0
    for key, text in parquet_docs(pattern, key_col, keep):
        if is_val(key):
            out.append(text)
            n += len(text)
            if n >= cap:
                break
    return out


def measure(tok, texts):
    tokens = words = whole = byte = 0
    exact = True
    for text in texts:
        ids = tok.encode(text)
        exact &= tok.decode(ids) == text
        tokens += len(ids)
        byte += sum(i < len(BYTE_TOKENS) for i in ids)
        for w in WORD.findall(text):
            words += 1
        for t in tok.tokenize(text):
            whole += bool(WORD.fullmatch(t.lstrip(SPACE)))
    return {"tokens_per_word": round(tokens / max(words, 1), 3), "whole_word_pct": round(100 * whole / max(words, 1), 1),
            "byte_pct": round(100 * byte / max(tokens, 1), 3), "round_trip": exact, "tokens": tokens, "words": words}


def single(tok, w):
    return len(tok.tokenize(" " + w)) == 1


def gold_cover(tok, path, cols=(0, 1)):
    rows = [l.split("\t") for l in open(path, encoding="utf-8").read().splitlines()[1:] if l.strip()]
    a = sum(single(tok, r[cols[0]]) for r in rows)
    b = sum(single(tok, r[cols[1]]) for r in rows)
    both = sum(single(tok, r[cols[0]]) and single(tok, r[cols[1]]) for r in rows)
    return {"pairs": len(rows), "left_single": a, "right_single": b, "both_single": both}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=os.path.join(ROOT, "merges.json"))
    ap.add_argument("--new", default=os.path.join(W, "round2", "merges.json"))
    ap.add_argument("--cap", type=int, default=1_000_000, help="χαρακτήρες κρατημένου κειμένου ανά πηγή")
    ap.add_argument("--clean", default=os.path.join(ROOT, "corpus", "CLEAN.txt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "round2_eval.json"))
    args = ap.parse_args()

    toks = {"round1": Tokenizer.load(args.old), "round2": Tokenizer.load(args.new)}
    texts = {name: held_out(name, args.cap) for name in sources()}
    texts["clean"] = [open(args.clean, encoding="utf-8").read()]
    res = {"vocab": {k: len(t.vocab) - len(t.dead) for k, t in toks.items()}, "sources": {}, "gold": {}}
    for name, tx in texts.items():
        res["sources"][name] = {k: measure(t, tx) for k, t in toks.items()}
        r1, r2 = res["sources"][name]["round1"], res["sources"][name]["round2"]
        print(f"{name:10s} r1 {r1['tokens_per_word']:.3f} ({r1['whole_word_pct']}%) → "
              f"r2 {r2['tokens_per_word']:.3f} ({r2['whole_word_pct']}%) · "
              f"{100 * (r2['tokens_per_word'] / r1['tokens_per_word'] - 1):+.1f}% · bytes {r2['byte_pct']}% · rt {r2['round_trip']}",
              flush=True)
    for name, path in (("greek_greeklish", os.path.join(ROOT, "corpus", "gold_open.tsv")),
                       ("el_en", os.path.join(W, "gold_elen.tsv"))):
        if os.path.exists(path):
            res["gold"][name] = {k: gold_cover(t, path) for k, t in toks.items()}
            print(name, res["gold"][name], flush=True)
    with open(args.out, "w", encoding="utf-8") as fo:
        json.dump(res, fo, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

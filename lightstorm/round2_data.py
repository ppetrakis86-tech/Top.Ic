"""Πείραμα 7: encode του μείγματος του Γύρου 2 με τον νέο tokenizer (2γ), και νέα gold pairs.

Πηγές όπως στο round2_tok.py. Val: τα κείμενα με is_val(κλειδί), τα ίδια που κράτησε έξω ο tokenizer.
CLEAN.txt: όλο το train του (κάθε 10ο μπλοκ στο val, όπως στον Γύρο 1).
Gold: ελληνικά↔greeklish από το gold_pairs.tsv και el↔en από το MUSE, μόνο ζευγάρια όπου κάθε λέξη είναι
ένα token στο νέο vocab. Σφράγισμα με το ίδιο hash με τον Γύρο 1, άρα το σφραγισμένο μισό μένει κλειστό.
Όλα πάνε στο corpus_r2/ (gitignored). Στο repo μπαίνει μόνο το r2_data_manifest.json.

    python3 -m lightstorm.round2_data --model corpus_wiki/round2c/merges.json
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from .data import is_sealed, md5, sha256, split_blocks
from .round2_tok import W, parquet_docs, sources
from .wiki import ROOT, elen_gold, is_val
from tokenizer import BYTE_TOKENS, Tokenizer

_tok = None


def _init(model):
    global _tok
    _tok = Tokenizer.load(model)


def _encode(text):
    return np.array(_tok.encode(text + "\n\n"), dtype=np.uint16)


def encode_source(name, args, ftrain, pool, t0):
    pattern, key_col, keep = sources()[name]
    st = {"docs": 0, "chars": 0, "train_tokens": 0, "val_tokens": 0, "val_docs": 0, "byte_tokens": 0}
    val_parts, buf, done = [], [], False

    def flush(buf):
        nonlocal done
        for (key, text), ids in zip(buf, pool.map(_encode, [t for _, t in buf], chunksize=32)):
            st["docs"] += 1
            st["chars"] += len(text)
            st["byte_tokens"] += int((ids < len(BYTE_TOKENS)).sum())
            if is_val(key):
                if st["val_tokens"] < args.val_cap:
                    val_parts.append(ids)
                    st["val_tokens"] += len(ids)
                    st["val_docs"] += 1
                continue
            if st["train_tokens"] < args.cap:
                ids.tofile(ftrain)
                st["train_tokens"] += len(ids)
        done = st["train_tokens"] >= args.cap and st["val_tokens"] >= args.val_cap

    for item in parquet_docs(pattern, key_col, keep):
        if st["train_tokens"] >= args.cap and not is_val(item[0]):
            continue                       # το train γέμισε: διαβάζουμε μόνο για το val
        buf.append(item)
        if len(buf) == 2048:
            flush(buf)
            buf = []
            print(f"{name}: {st['docs']:,} κείμενα · {st['train_tokens']:,} train · {st['val_tokens']:,} val · "
                  f"{time.time() - t0:.0f}s", flush=True)
            if done:
                break
    if buf and not done:
        flush(buf)
    val = np.concatenate(val_parts)[:args.val_cap] if val_parts else np.zeros(0, np.uint16)
    st["val_tokens"] = int(len(val))
    st["tokens_per_char"] = round((st["train_tokens"] + st["val_tokens"]) / max(1, st["chars"]), 4)
    return val, st


def write_gold(rows, header, out, open_name, sealed_name):
    open_rows = [r for r in rows if not is_sealed(r[0], r[1])]
    sealed_rows = [r for r in rows if is_sealed(r[0], r[1])]
    for name, part in ((open_name, open_rows), (sealed_name, sealed_rows)):
        with open(os.path.join(out, name), "w", encoding="utf-8") as fo:
            fo.write("\n".join([header] + ["\t".join(r) for r in part]) + "\n")
    return {"open": len(open_rows), "sealed": len(sealed_rows),
            "sealed_sha256": sha256(os.path.join(out, sealed_name))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(W, "round2c", "merges.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "corpus_r2"))
    ap.add_argument("--cap", type=int, default=20_000_000, help="train tokens ανά πηγή")
    ap.add_argument("--val-cap", type=int, default=300_000, help="val tokens ανά πηγή")
    ap.add_argument("--clean", default=os.path.join(ROOT, "corpus", "CLEAN.txt"))
    ap.add_argument("--gold", default=os.path.join(ROOT, "corpus", "gold_pairs.tsv"))
    ap.add_argument("--muse", default=os.path.join(W, "muse_el-en.txt"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "r2_data_manifest.json"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    merges_md5 = md5(args.model)
    tok = Tokenizer.load(args.model)
    assert len(tok.vocab) < 2 ** 16
    os.makedirs(args.out, exist_ok=True)
    t0, vals, stats = time.time(), [], {}
    with open(os.path.join(args.out, "train.bin"), "wb") as ftrain:
        with Pool(args.workers, initializer=_init, initargs=(args.model,)) as pool:
            for name in sources():
                val, stats[name] = encode_source(name, args, ftrain, pool, t0)
                vals.append(val)
        text = open(args.clean, encoding="utf-8").read()
        ids = tok.encode(text)
        assert tok.decode(ids) == text, "round-trip απέτυχε στο CLEAN"
        train, val = split_blocks(np.array(ids, dtype=np.uint16))
        train.tofile(ftrain)
        vals.append(val)
        stats["clean"] = {"chars": len(text), "train_tokens": int(len(train)), "val_tokens": int(len(val))}
    np.concatenate(vals).tofile(os.path.join(args.out, "val.bin"))

    def single(w):
        t = tok.tokenize(" " + w)
        return tok.token_to_id[t[0]] if len(t) == 1 else None

    lines = open(args.gold, encoding="utf-8").read().splitlines()
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        g, l = line.split("\t")[:2]
        gi, li = single(g), single(l)
        if gi is not None and li is not None:
            rows.append([g, l, str(gi), str(li)])
    gold = {"greeklish": write_gold(rows, lines[0], args.out, "gold_open.tsv", "gold_sealed.tsv")}
    gold["greeklish"]["from"] = len(lines) - 1
    gold["elen"] = write_gold(elen_gold(tok, args.muse), "greek\tenglish\tgreek_id\tenglish_id",
                              args.out, "gold_elen.tsv", "gold_elen_sealed.tsv")

    assert md5(args.model) == merges_md5, "τα merges άλλαξαν!"
    manifest = {"merges_md5": merges_md5, "vocab": len(tok.vocab), "cap_per_source": args.cap,
                "val_cap_per_source": args.val_cap, "clean_md5": md5(args.clean), "gold_md5": md5(args.gold),
                "muse_md5": md5(args.muse), "train_md5": md5(os.path.join(args.out, "train.bin")),
                "val_md5": md5(os.path.join(args.out, "val.bin")), "sources": stats, "gold": gold,
                "seconds": round(time.time() - t0)}
    with open(args.manifest, "w", encoding="utf-8") as fo:
        json.dump(manifest, fo, indent=1, ensure_ascii=False)
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

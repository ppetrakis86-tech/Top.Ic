"""Πείραμα 6: η ελληνική Wikipedia ως corpus, με τα παγωμένα merges του Γύρου 1.

Μόνο encode. Τα parquet και τα .bin μένουν στο corpus_wiki/, που δεν ανεβαίνει ποτέ στο GitHub.
Στο repo μένει μόνο το wiki_manifest.json: μετρήσεις και hashes, κανένα κείμενο.
Το split γίνεται ανά άρθρο (hash του id), ώστε κανένα άρθρο να μη μοιράζεται ανάμεσα σε train και val.

    python3 -m lightstorm.wiki --sample 2000     # μόνο μέτρηση ταχύτητας, δεν γράφει τίποτα
    python3 -m lightstorm.wiki                   # ολόκληρο -> corpus_wiki/{train,val}.bin
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenizer import BYTE_TOKENS, Tokenizer  # noqa: E402
from lightstorm.data import md5  # noqa: E402

VAL_EVERY = 100          # ~1% των άρθρων στο val
VAL_CAP = 1_000_000      # το val κόβεται εδώ, ώστε κάθε αξιολόγηση να μένει γρήγορη
SPLIT_SALT = "Top.Ic-wiki-split"

_tok = None


def _init(model):
    global _tok
    _tok = Tokenizer.load(model)


def _encode(item):
    art_id, text = item
    ids = np.array(_tok.encode(text + "\n\n"), dtype=np.uint16)
    return art_id, ids


def is_val(art_id):
    return hashlib.sha256(f"{SPLIT_SALT}\t{art_id}".encode()).digest()[0] % VAL_EVERY == 0


def articles(files, limit=0):
    import pyarrow.parquet as pq
    n = 0
    for path in files:
        table = pq.read_table(path, columns=["id", "text"])
        for art_id, text in zip(table.column("id").to_pylist(), table.column("text").to_pylist()):
            yield art_id, text
            n += 1
            if limit and n >= limit:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "corpus_wiki", "parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "corpus_wiki"))
    ap.add_argument("--model", default=os.path.join(ROOT, "merges.json"))
    ap.add_argument("--gold", default=os.path.join(ROOT, "corpus", "gold_open.tsv"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "wiki_manifest.json"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--sample", type=int, default=0, help="μόνο τόσα άρθρα, για μέτρηση ταχύτητας")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.parquet")))
    assert files, f"κανένα parquet στο {args.src}"
    merges_md5 = md5(args.model)
    tok = Tokenizer.load(args.model)
    byte_ids = len(BYTE_TOKENS)          # τα ids < 256 είναι byte fallback

    os.makedirs(args.out, exist_ok=True)
    write = not args.sample
    ftrain = open(os.path.join(args.out, "train.bin"), "wb") if write else None
    val_parts, n_val = [], 0
    stats = {"articles": 0, "chars": 0, "train_tokens": 0, "val_tokens": 0, "val_articles": 0, "byte_tokens": 0}
    t0 = time.time()
    texts = {}
    def feed():
        for art_id, text in articles(files, args.sample):
            texts[art_id] = len(text)
            yield art_id, text
    with Pool(args.workers, initializer=_init, initargs=(args.model,)) as pool:
        for art_id, ids in pool.imap(_encode, feed(), chunksize=64):
            stats["articles"] += 1
            stats["chars"] += texts.pop(art_id)
            stats["byte_tokens"] += int((ids < byte_ids).sum())
            if is_val(art_id):
                if n_val < VAL_CAP:
                    val_parts.append(ids)
                    n_val += len(ids)
                    stats["val_articles"] += 1
                continue                       # τα άρθρα του val πέρα από το όριο δεν πάνε ούτε στο train
            stats["train_tokens"] += len(ids)
            if write:
                ids.tofile(ftrain)
            if stats["articles"] % 20000 == 0:
                print(f"{stats['articles']:,} άρθρα · {stats['train_tokens']:,} train tokens · {time.time() - t0:.0f}s",
                      flush=True)
    val = np.concatenate(val_parts)[:VAL_CAP] if val_parts else np.zeros(0, np.uint16)
    stats["val_tokens"] = int(len(val))
    secs = time.time() - t0
    stats["tokens_per_char"] = round((stats["train_tokens"] + stats["val_tokens"]) / max(1, stats["chars"]), 4)
    print(json.dumps(stats, indent=1), f"\n{secs:.0f}s · {stats['chars'] / secs:,.0f} chars/s", flush=True)
    if not write:
        return

    ftrain.close()
    val.tofile(os.path.join(args.out, "val.bin"))
    shutil.copy(args.gold, os.path.join(args.out, "gold_open.tsv"))   # μόνο το ανοιχτό μισό, ΠΟΤΕ το sealed
    assert md5(args.model) == merges_md5, "το merges.json άλλαξε!"
    manifest = {"source": "wikimedia/wikipedia 20231101.el (Hugging Face)",
                "parquet_md5": {os.path.basename(f): md5(f) for f in files},
                "merges_md5": merges_md5, "vocab": len(tok.vocab),
                "split": {"by": "article id hash", "val_every": VAL_EVERY, "val_cap": VAL_CAP},
                "train_md5": md5(os.path.join(args.out, "train.bin")),
                "val_md5": md5(os.path.join(args.out, "val.bin")), **stats}
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

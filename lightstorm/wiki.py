"""Πείραμα 6: Wikipedia (ελληνική και Simple English) ως corpus, με τα παγωμένα merges του Γύρου 1.

Μόνο encode. Τα parquet και τα .bin μένουν στο corpus_wiki/, που δεν ανεβαίνει ποτέ στο GitHub.
Στο repo μένει μόνο το wiki_manifest.json: μετρήσεις και hashes, κανένα κείμενο.
Το split γίνεται ανά άρθρο (hash του id), ώστε κανένα άρθρο να μη μοιράζεται ανάμεσα σε train και val.

    python3 -m lightstorm.wiki --sample 2000                 # μόνο μέτρηση, δεν γράφει τίποτα
    python3 -m lightstorm.wiki --src el=corpus_wiki/parquet_el --src simple=corpus_wiki/parquet_simple \\
        --cap 50000000 --val-cap 500000 --muse corpus_wiki/muse_el-en.txt   # πείραμα 6
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
from lightstorm.data import is_sealed, md5, sha256  # noqa: E402

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


def encode_source(lang, files, args, ftrain, byte_ids, t0):
    """Ένα σώμα κειμένων (μία γλώσσα): train στο ftrain μέχρι το όριο, val σε δική του λίστα."""
    val_parts, n_val = [], 0
    st = {"articles": 0, "chars": 0, "train_tokens": 0, "val_tokens": 0, "val_articles": 0, "byte_tokens": 0}
    done = False

    def chunks(size=4096):
        buf = []
        for item in articles(files, args.sample):
            buf.append(item)
            if len(buf) == size:
                yield buf
                buf = []
        if buf:
            yield buf

    # κομμάτια των 4096 άρθρων: το όριο ελέγχεται ανάμεσα στα κομμάτια, χωρίς break μέσα στο imap
    # (το break μέσα στο imap κόλλαγε το Pool στο κλείσιμο, και το imap διάβαζε όλα τα άρθρα στη μνήμη)
    with Pool(args.workers, initializer=_init, initargs=(args.model,)) as pool:
        for chunk in chunks():
            for (art_id, text), (_, ids) in zip(chunk, pool.map(_encode, chunk, chunksize=64)):
                st["articles"] += 1
                st["chars"] += len(text)
                st["byte_tokens"] += int((ids < byte_ids).sum())
                if is_val(f"{lang}:{art_id}"):
                    if n_val < args.val_cap:
                        val_parts.append(ids)
                        n_val += len(ids)
                        st["val_articles"] += 1
                    continue                   # τα άρθρα του val πέρα από το όριο δεν πάνε ούτε στο train
                st["train_tokens"] += len(ids)
                if ftrain:
                    ids.tofile(ftrain)
                if args.cap and st["train_tokens"] >= args.cap:
                    done = True                # όριο ανά γλώσσα: ίσος όγκος από κάθε μία
                    break
            print(f"{lang}: {st['articles']:,} άρθρα · {st['train_tokens']:,} train tokens · "
                  f"{time.time() - t0:.0f}s", flush=True)
            if done:
                break
    val = np.concatenate(val_parts)[:args.val_cap] if val_parts else np.zeros(0, np.uint16)
    st["val_tokens"] = int(len(val))
    st["tokens_per_char"] = round((st["train_tokens"] + st["val_tokens"]) / max(1, st["chars"]), 4)
    return val, st


def elen_gold(tok, muse_path):
    """Ζευγάρια μετάφρασης el↔en από το λεξικό MUSE, όπου κάθε λέξη είναι ένα token (με ▁).
    Ένα προς ένα: κρατάμε την πρώτη (πιο συχνή) μετάφραση κάθε λέξης, σε κάθε κατεύθυνση."""
    def single(w):
        t = tok.tokenize(" " + w)
        return tok.token_to_id[t[0]] if len(t) == 1 else None
    rows, seen_g, seen_e = [], set(), set()
    for line in open(muse_path, encoding="utf-8"):
        parts = line.split()
        if len(parts) != 2 or parts[0] == parts[1] or parts[0] in seen_g or parts[1] in seen_e:
            continue
        g, e = parts
        gi, ei = single(g), single(e)
        if gi is None or ei is None:
            continue
        seen_g.add(g)
        seen_e.add(e)
        rows.append([g, e, str(gi), str(ei)])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", default=None,
                    help="γλώσσα=φάκελος με parquet, π.χ. el=corpus_wiki/parquet_el (επαναλαμβάνεται)")
    ap.add_argument("--cap", type=int, default=0, help="όριο train tokens ανά γλώσσα (0 = όλα)")
    ap.add_argument("--val-cap", type=int, default=VAL_CAP, help="όριο val tokens ανά γλώσσα")
    ap.add_argument("--out", default=os.path.join(ROOT, "corpus_wiki"))
    ap.add_argument("--model", default=os.path.join(ROOT, "merges.json"))
    ap.add_argument("--gold", default=os.path.join(ROOT, "corpus", "gold_open.tsv"))
    ap.add_argument("--muse", default="", help="λεξικό MUSE el-en για το gold_elen (προαιρετικό)")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "wiki_manifest.json"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--sample", type=int, default=0, help="μόνο τόσα άρθρα, για μέτρηση ταχύτητας")
    args = ap.parse_args()

    sources = dict(s.split("=", 1) for s in (args.src or [f"el={os.path.join(ROOT, 'corpus_wiki', 'parquet_el')}"]))
    files = {lang: sorted(glob.glob(os.path.join(d, "*.parquet"))) for lang, d in sources.items()}
    for lang, f in files.items():
        assert f, f"κανένα parquet για {lang}"
    merges_md5 = md5(args.model)
    tok = Tokenizer.load(args.model)
    byte_ids = len(BYTE_TOKENS)          # τα ids < 256 είναι byte fallback

    os.makedirs(args.out, exist_ok=True)
    write = not args.sample
    ftrain = open(os.path.join(args.out, "train.bin"), "wb") if write else None
    t0, vals, stats = time.time(), [], {}
    for lang, f in files.items():        # η σειρά στο αρχείο δεν πειράζει: το training παίρνει τυχαία παράθυρα
        val, stats[lang] = encode_source(lang, f, args, ftrain, byte_ids, t0)
        vals.append(val)
    secs = time.time() - t0
    print(json.dumps(stats, indent=1), f"\n{secs:.0f}s", flush=True)
    if not write:
        return

    ftrain.close()
    np.concatenate(vals).tofile(os.path.join(args.out, "val.bin"))
    shutil.copy(args.gold, os.path.join(args.out, "gold_open.tsv"))   # μόνο το ανοιχτό μισό, ΠΟΤΕ το sealed
    manifest = {"source": "wikimedia/wikipedia 20231101 (Hugging Face): " + ", ".join(sources),
                "parquet_md5": {f"{lang}/{os.path.basename(p)}": md5(p) for lang, fs in files.items() for p in fs},
                "merges_md5": merges_md5, "vocab": len(tok.vocab), "cap_per_lang": args.cap,
                "split": {"by": "article id hash", "val_every": VAL_EVERY, "val_cap_per_lang": args.val_cap},
                "train_md5": md5(os.path.join(args.out, "train.bin")),
                "val_md5": md5(os.path.join(args.out, "val.bin")), "per_lang": stats}
    if args.muse:
        rows = elen_gold(tok, args.muse)
        header = "greek\tenglish\tgreek_id\tenglish_id"
        open_rows = [r for r in rows if not is_sealed(r[0], r[1])]
        sealed_rows = [r for r in rows if is_sealed(r[0], r[1])]
        for name, part in (("gold_elen.tsv", open_rows), ("gold_elen_sealed.tsv", sealed_rows)):
            with open(os.path.join(args.out, name), "w", encoding="utf-8") as fo:
                fo.write("\n".join([header] + ["\t".join(r) for r in part]) + "\n")
        manifest.update({"muse_md5": md5(args.muse), "gold_elen": len(open_rows), "gold_elen_sealed": len(sealed_rows),
                         "gold_elen_sealed_sha256": sha256(os.path.join(args.out, "gold_elen_sealed.tsv"))})
    assert md5(args.model) == merges_md5, "το merges.json άλλαξε!"
    with open(args.manifest, "w", encoding="utf-8") as fo:
        json.dump(manifest, fo, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in manifest.items() if k != "parquet_md5"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

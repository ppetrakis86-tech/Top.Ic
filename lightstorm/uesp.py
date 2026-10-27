"""UESP (en.uesp.net, CC BY-SA 2.5): το XML dump σε καθαρό κείμενο, για το corpus του Γύρου 2.

Κρατάει σελίδες περιεχομένου (κύριος χώρος και οι χώροι παιχνιδιών/Lore, όχι συζητήσεις, χρήστες, templates),
χωρίς redirects. Από κάθε σελίδα μένουν μόνο παράγραφοι με πρόζα (>= MIN_WORDS λέξεις): οι πίνακες
δεδομένων (ESO items, stats) πέφτουν. Μοιράζει τις σελίδες με hash του τίτλου: μισές μένουν αγγλικές,
μισές πάνε για μετάφραση στα ελληνικά, ώστε κανένα κείμενο να μην υπάρχει και στις δύο γλώσσες.

Γράφει corpus_wiki/uesp/parquet_en/ και parquet_to_el/ (gitignored) και uesp_manifest.json (μόνο μετρήσεις).

    python3 -m lightstorm.uesp corpus_wiki/uesp/uespwiki-2026-10-18-current.xml.bz2
"""
import argparse
import bz2
import collections
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET

import mwparserfromhell
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_SALT = "Top.Ic-uesp-split"
MIN_WORDS = 8          # παράγραφος με λιγότερες λέξεις = πίνακας, λίστα ή τίτλος
MIN_CHARS = 400        # σελίδα με λιγότερη πρόζα πέφτει
IMAGE_OPTS = re.compile(r"(?:thumbnail|thumb)\||\b(?:frame|frameless|right|left|center|none|upright|\d+(?:x\d+)?px)\|")


def content_ns(key):
    return key == 0 or (100 <= key < 800 and key % 2 == 0)


def prose(wikitext):
    text = mwparserfromhell.parse(wikitext).strip_code(normalize=True, collapse=True)
    text = IMAGE_OPTS.sub("", text)    # οι λεζάντες εικόνων αφήνουν «thumb|right|» μπροστά
    keep = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = " ".join(para.split())
        if len(para.split()) >= MIN_WORDS and not para.startswith(("{|", "|", "!")):
            keep.append(para)
    return "\n\n".join(keep)


def to_greek(title):
    return hashlib.sha256(f"{SPLIT_SALT}\t{title}".encode()).digest()[0] % 2 == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", default=os.path.join(ROOT, "corpus_wiki", "uesp"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "uesp_manifest.json"))
    args = ap.parse_args()

    names, tag = {}, lambda e: e.tag.rsplit("}", 1)[-1]
    bufs = {"en": {"title": [], "text": []}, "to_el": {"title": [], "text": []}}
    st = collections.Counter()
    per_ns = collections.Counter()
    chars = collections.Counter()
    for _, el in ET.iterparse(bz2.open(args.dump), events=("end",)):
        t = tag(el)
        if t == "namespace":
            names[int(el.get("key"))] = el.text or ""
        elif t == "page":
            f = {tag(c): c for c in el.iter()}
            ns = int(f["ns"].text)
            st["pages"] += 1
            if not content_ns(ns) or "redirect" in f:
                el.clear()
                continue
            title, raw = f["title"].text, (f["text"].text if "text" in f else "") or ""
            body = prose(raw)
            if len(body) < MIN_CHARS:
                st["too_little_prose"] += 1
                el.clear()
                continue
            part = "to_el" if to_greek(title) else "en"
            bufs[part]["title"].append(title)
            bufs[part]["text"].append(body)
            per_ns[names.get(ns, str(ns))] += 1
            chars[part] += len(body)
            el.clear()
    for part, buf in bufs.items():
        d = os.path.join(args.out, f"parquet_{part}")
        os.makedirs(d, exist_ok=True)
        pq.write_table(pa.table(buf), os.path.join(d, "uesp.parquet"))
    stats = {"source": os.path.basename(args.dump) + " (UESP, CC BY-SA 2.5)", **st,
             "kept": {p: len(b["title"]) for p, b in bufs.items()}, "chars": dict(chars),
             "per_namespace": dict(per_ns.most_common())}
    json.dump(stats, open(args.manifest, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

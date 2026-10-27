"""UESP → ελληνικά με το Helsinki-NLP/opus-mt-en-el (Marian, Apache 2.0), τοπικά στη GPU.

Μεταφράζει μόνο το corpus_wiki/uesp/parquet_to_el/ (οι σελίδες που ΔΕΝ μένουν αγγλικές).
Ανά πρόταση, με τις προτάσεις ταξινομημένες κατά μήκος για γρήγορα batches. Γράφει ένα parquet
ανά κομμάτι σελίδων στο corpus_wiki/uesp/parquet_el/ και, αν κοπεί, συνεχίζει από το επόμενο κομμάτι.

    python3 -m lightstorm.translate_uesp --limit 200      # μέτρηση ταχύτητας, δεν γράφει τίποτα
    python3 -m lightstorm.translate_uesp                  # όλο
"""
import argparse
import os
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import MarianMTModel, MarianTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "Helsinki-NLP/opus-mt-en-el"
SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z])")
MAX_CHARS = 600        # πολύ μεγάλες «προτάσεις» (λίστες χωρίς τελεία) κόβονται σε κόμματα


def sentences(para):
    out = []
    for s in SENT.split(para):
        while len(s) > MAX_CHARS:
            cut = s.rfind(", ", 0, MAX_CHARS)
            cut = cut + 1 if cut > 0 else MAX_CHARS
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out


@torch.no_grad()
def translate(sents, tok, model, batch):
    order = sorted(range(len(sents)), key=lambda i: len(sents[i]))
    out = [None] * len(sents)
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        enc = tok([sents[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(model.device)
        gen = model.generate(**enc, num_beams=1, max_new_tokens=320)
        for j, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[j] = text
    return out


def translate_pages(titles, texts, tok, model, batch):
    """Κρατάει τη δομή: παράγραφοι χωρισμένες με κενή γραμμή, προτάσεις ενωμένες με κενό."""
    flat, shape = [], []
    for text in texts:
        paras = [sentences(p) for p in text.split("\n\n")]
        shape.append([len(p) for p in paras])
        for p in paras:
            flat.extend(p)
    done, k, pages = translate(flat, tok, model, batch), 0, []
    for lens in shape:
        paras = []
        for n in lens:
            paras.append(" ".join(done[k:k + n]))
            k += n
        pages.append("\n\n".join(paras))
    return pages, len(flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "corpus_wiki", "uesp", "parquet_to_el", "uesp.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "corpus_wiki", "uesp", "parquet_el"))
    ap.add_argument("--pages", type=int, default=500, help="σελίδες ανά κομμάτι/αρχείο")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="μόνο τόσες σελίδες, για μέτρηση (δεν γράφει)")
    args = ap.parse_args()

    tok = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL).to("cuda").half().eval()
    model.generation_config.max_length = None      # αλλιώς προειδοποίηση σε κάθε batch (μετράει το max_new_tokens)
    t = pq.read_table(args.src).to_pydict()
    titles, texts = t["title"], t["text"]
    if args.limit:
        t0 = time.time()
        out, n = translate_pages(titles[:args.limit], texts[:args.limit], tok, model, args.batch)
        secs = time.time() - t0
        chars = sum(len(x) for x in texts[:args.limit])
        total = sum(len(x) for x in texts)
        print(f"{args.limit} σελίδες · {n} προτάσεις · {chars:,} χαρ. · {secs:.0f}s · "
              f"{chars / secs:,.0f} χαρ./s → όλο ≈ {total / (chars / secs) / 3600:.1f} ώρες")
        for src, dst in zip(texts[:3], out[:3]):
            print("EN:", src[:300], "\nEL:", dst[:300], "\n")
        return
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    for part, start in enumerate(range(0, len(titles), args.pages)):
        path = os.path.join(args.out, f"uesp-el-{part:03d}.parquet")
        if os.path.exists(path):
            continue
        end = start + args.pages
        out, n = translate_pages(titles[start:end], texts[start:end], tok, model, args.batch)
        pq.write_table(pa.table({"title": titles[start:end], "text": out}), path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"κομμάτι {part} · σελίδες {end:,}/{len(titles):,} · {n} προτάσεις · {time.time() - t0:.0f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()

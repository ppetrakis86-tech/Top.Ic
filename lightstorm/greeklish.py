"""Ελληνικά → greeklish με τη γραφή του Πέτρου.

Κανόνες από τις μετρήσεις στο CLEAN.txt (27/10, όχι από τα gold pairs): ορθογραφικά, όχι φωνητικά.
χ→x, ξ→ks, ψ→ps, θ→th, η/ι/υ→i/i/u, ω→o, β→b, ου→ou, αυ/ευ→au/eu, χωρίς τόνους.
Για κάθε λέξη που ο Πέτρος έχει ήδη γράψει στο CLEAN.txt, κρατάμε τη δική του γραφή (λεξικό).
Για τις υπόλοιπες: κανόνες, και το υ γίνεται y σε ~20% των λέξεων (σταθερά ανά λέξη, όπως γράφει κι αυτός).

    python3 -m lightstorm.greeklish --clean corpus/CLEAN.txt --sample "Η γάτα κοιτάζει έξω από το παράθυρο."
"""
import argparse
import collections
import hashlib
import re
import unicodedata

DIGRAPHS = [("ου", "ou"), ("αι", "ai"), ("ει", "ei"), ("οι", "oi"), ("αυ", "au"), ("ευ", "eu"), ("ηυ", "iu"),
            ("μπ", "mp"), ("ντ", "nt"), ("γγ", "gg"), ("γκ", "gk"), ("τσ", "ts"), ("τζ", "tz")]
SINGLE = dict(zip("αβγδεζηθικλμνξοπρσςτυφχψω",
                  ["a", "b", "g", "d", "e", "z", "i", "th", "i", "k", "l", "m", "n", "ks", "o", "p", "r", "s", "s",
                   "t", "u", "f", "x", "ps", "o"]))
Y_SHARE = 20          # % των άγνωστων λέξεων με υ που γράφονται με y
GREEK_WORD = re.compile(r"[Ͱ-Ͽἀ-῿]+")


def strip_accents(w):
    return "".join(ch for ch in unicodedata.normalize("NFD", w) if unicodedata.category(ch) != "Mn")


def rule(word):
    """Μόνο κανόνες. Η λέξη σε πεζά, χωρίς τόνους."""
    w, out, i = strip_accents(word.lower()), [], 0
    while i < len(w):
        for g, l in DIGRAPHS:
            if w.startswith(g, i):
                out.append(l)
                i += 2
                break
        else:
            out.append(SINGLE.get(w[i], w[i]))
            i += 1
    return "".join(out)


def variants(lat):
    """Οι εναλλακτικές γραφές που χρησιμοποιεί ο Πέτρος: u/y, x/ch, b/v."""
    out = {lat}
    for a, b in (("u", "y"), ("x", "ch"), ("b", "v")):
        out |= {v.replace(a, b) for v in out}
    return out


def his_vocab(clean_path):
    text = open(clean_path, encoding="utf-8").read().lower()
    return collections.Counter(re.findall(r"\b[a-z]+\b", text))


class Greeklish:
    def __init__(self, vocab):
        self.vocab = vocab
        self.cache = {}
        self.stats = collections.Counter()

    def word(self, w):
        key = w.lower()
        if key not in self.cache:
            lat = rule(key)
            seen = [(self.vocab[v], v) for v in variants(lat) if self.vocab.get(v)]
            if seen:
                lat, src = max(seen)[1], "his"
            else:
                src = "rule"
                if "u" in lat and hashlib.sha256(key.encode()).digest()[0] % 100 < Y_SHARE:
                    lat = lat.replace("u", "y")
            self.cache[key] = (lat, src)
        lat, src = self.cache[key]
        self.stats[src] += 1
        if w[:1].isupper():
            lat = lat.upper() if w.isupper() and len(w) > 1 else lat[:1].upper() + lat[1:]
        return lat

    def text(self, t):
        return GREEK_WORD.sub(lambda m: self.word(m.group()), t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="corpus/CLEAN.txt")
    ap.add_argument("--sample", default="")
    args = ap.parse_args()
    g = Greeklish(his_vocab(args.clean))
    print(g.text(args.sample))
    print(dict(g.stats))


if __name__ == "__main__":
    main()

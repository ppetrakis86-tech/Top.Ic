"""Πείραμα 8: αφαίρεση διπλότυπων γραμμών πριν το encode.

Οι παγίδες βρήκαν 3,6% του val αυτούσιο στο train (διάλογοι ESO σε πολλές σελίδες, «=====», CSS, git log στο CLEAN).
Κανόνας (METRICS.md, «Πείραμα 8»): μια γραμμή με ≥ MIN_CHARS χαρακτήρες, ή χωρίς κανένα γράμμα/ψηφίο, κρατιέται
μόνο την πρώτη φορά που εμφανίζεται σε όλο το μείγμα. Οι κοντές γραμμές με κείμενο («ok», «Ναι») μένουν πάντα.
"""
import hashlib

MIN_CHARS = 40


class Dedup:
    def __init__(self, min_chars=MIN_CHARS):
        self.min_chars = min_chars
        self.seen = set()
        self.kept_chars = 0
        self.dropped_chars = 0
        self.dropped_lines = 0

    def __call__(self, text):
        out = []
        for line in text.split("\n"):
            s = line.strip()
            if s and (len(s) >= self.min_chars or not any(c.isalnum() for c in s)):
                k = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
                if k in self.seen:
                    self.dropped_chars += len(line) + 1
                    self.dropped_lines += 1
                    continue
                self.seen.add(k)
            out.append(line)
        text = "\n".join(out)
        self.kept_chars += len(text)
        return text

    def stats(self):
        tot = self.kept_chars + self.dropped_chars
        return {"min_chars": self.min_chars, "dropped_lines": self.dropped_lines, "dropped_chars": self.dropped_chars,
                "dropped_pct": round(100 * self.dropped_chars / max(1, tot), 3), "unique_lines": len(self.seen)}

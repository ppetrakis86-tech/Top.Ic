import os
import re
import json
import heapq
import argparse

SPACE = "\u2581"  # ▁ : το κενό γίνεται μέρος της λέξης που ακολουθεί
PIECE = re.compile(
    rf"\d"                                        # κάθε ψηφίο μόνο του, χωρίς κενό: 0-9 = 10 tokens
    rf"|{SPACE}*(?:[^\W\d_]+|_|[^\s\w{SPACE}])"  # κενά + λέξη, _ ή σύμβολο
    rf"|{SPACE}+|\s"                               # κενά μόνα τους, ή άλλο whitespace (\n, \t, \r)
)
BYTE_TOKENS = [f"<{b:02x}>" for b in range(256)]
BYTE_RE = re.compile(r"<[0-9a-f]{2}>")


def has_digit_merge(tok):
    """Token από merge που περιέχει ψηφίο (π.χ. '39', '▁3'). Τα ψηφία ζουν μόνο ως χαρακτήρες 0-9."""
    text = BYTE_RE.sub("", tok)
    return len(text) > 1 and bool(re.search(r"\d", text))


def pre_tokenize(text):
    return PIECE.findall(text.replace(" ", SPACE))

# --- 1. Συναρτήσεις I/O (Αρχείων) ---

def load_text(filename="text.txt"):
    """Διαβάζει το κείμενο εκπαίδευσης από το txt file."""
    if not os.path.exists(filename):
        default_text = "My vegetables list: bananas, red apples, green oranges, broccoli, beans"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(default_text)
        print(f"📄 Δημιουργήθηκε το αρχείο '{filename}' με προεπιλεγμένο κείμενο.")
        return default_text

    with open(filename, "r", encoding="utf-8") as f:
        print(f"📖 Διαβάστηκε το κείμενο από το '{filename}'")
        return f.read()


# --- 2. Ο Tokenizer (μοντέλο + ENCODE / DECODE) ---

class Tokenizer:
    """Κρατάει τα merges, το alphabet και το vocab (token -> id)."""

    def __init__(self, merges=None, alphabet=None, vocab=None):
        self.merges = merges or []
        self.alphabet = set(alphabet or [])
        # vocab: λίστα όπου index = token id. Μόνο προσθήκες στο τέλος, ώστε τα ids να μην αλλάζουν.
        self.vocab = vocab or list(BYTE_TOKENS)
        # Νεκρές θέσεις: παλιά tokens με ενωμένα ψηφία. Μένουν στο vocab για να μην αλλάξει κανένα id,
        # αλλά δεν βγαίνουν ποτέ από το encode.
        self.dead = {i for i, tok in enumerate(self.vocab) if i >= len(BYTE_TOKENS) and has_digit_merge(tok)}
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab) if i not in self.dead}
        self.ranks = {tuple(item["pair"]): item["id"] for item in self.merges
                      if not has_digit_merge(item["pair"][0] + item["pair"][1])}
        self.cache = {}

    def _add_token(self, tok):
        if tok not in self.token_to_id:
            self.token_to_id[tok] = len(self.vocab)
            self.vocab.append(tok)

    def add_alphabet(self, chars):
        for ch in sorted(set(chars) - self.alphabet):
            self.alphabet.add(ch)
            self._add_token(ch)
        self.cache = {}

    def add_merge(self, pair):
        self.ranks[pair] = len(self.merges)
        self.merges.append({"id": len(self.merges), "pair": list(pair)})
        self._add_token(pair[0] + pair[1])
        self.cache = {}

    @classmethod
    def load(cls, filename="merges.json"):
        """Φορτώνει το μοντέλο από το JSON (τα παλιά μοντέλα με </w> απορρίπτονται)."""
        if not os.path.exists(filename):
            print(f"🆕 Το αρχείο '{filename}' δεν βρέθηκε. Ξεκινάμε εκπαίδευση από το μηδέν!")
            return cls()

        print(f"📂 Φορτώθηκε το '{filename}'!")
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        merges = data if isinstance(data, list) else data["merges"]
        if isinstance(data, list) or any("</w>" in t for m in merges for t in m["pair"]):
            raise SystemExit(f"❌ Το '{filename}' είναι από το παλιό σχήμα με </w> και δεν ταιριάζει "
                             f"με το ▁. Σβήσ' το (ή δώσε άλλο --model) και εκπαίδευσε από την αρχή.")
        if "vocab" in data:
            tok = cls(data["merges"], data["alphabet"], data["vocab"])
            if tok.dead:
                print(f"🧹 {len(tok.dead)} tokens με ενωμένα ψηφία έγιναν νεκρές θέσεις "
                      f"(π.χ. {', '.join(repr(tok.vocab[i]) for i in sorted(tok.dead)[:3])}). Τα ids δεν άλλαξαν.")
            return tok

        # Χωρίς vocab: το χτίζουμε από alphabet + merges
        tok = cls()
        tok.add_alphabet(data["alphabet"])
        for item in data["merges"]:
            tok.add_merge(tuple(item["pair"]))
        return tok

    def save(self, filename="merges.json"):
        """Αποθηκεύει merges, alphabet και vocab στο JSON."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"alphabet": sorted(self.alphabet), "merges": self.merges, "vocab": self.vocab},
                      f, ensure_ascii=False, indent=1)
        print(f"💾 Το ανανεωμένο Tokenizer αποθηκεύτηκε στο '{filename}' ({len(self.vocab)} tokens)")

    def split_chars(self, word):
        """Χαρακτήρες που δεν είδαμε στο training σπάνε σε UTF-8 bytes."""
        out = []
        for ch in word:
            if ch in self.alphabet:
                out.append(ch)
            else:
                out.extend(BYTE_TOKENS[b] for b in ch.encode("utf-8"))
        return out

    def bpe(self, symbols):
        """Ενώνει κάθε φορά το ζευγάρι με το μικρότερο rank (= εκπαιδεύτηκε πιο νωρίς)."""
        while len(symbols) > 1:
            best = min(zip(symbols, symbols[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            symbols = encode_word(symbols, best)
        return symbols

    def tokenize(self, text):
        """Κείμενο -> λίστα από tokens (strings)."""
        tokens = []
        for word in pre_tokenize(text):
            if word not in self.cache:
                self.cache[word] = self.bpe(self.split_chars(word))
            tokens.extend(self.cache[word])
        return tokens

    def encode(self, text):
        """Κείμενο -> λίστα από token ids."""
        return [self.token_to_id[tok] for tok in self.tokenize(text)]

    def decode(self, ids):
        """Token ids -> κείμενο. Τα ids < 256 είναι bytes, όλα τα άλλα κείμενο με ▁ για τα κενά."""
        out, buf = [], bytearray()
        for i in ids:
            if i < len(BYTE_TOKENS):
                buf.append(i)
                continue
            if buf:
                out.append(buf.decode("utf-8", "replace"))
                buf = bytearray()
            out.append(self.vocab[i].replace(SPACE, " "))
        if buf:
            out.append(buf.decode("utf-8", "replace"))
        return "".join(out)


def encode_word(symbols, pair):
    """Εφαρμόζει ένα συγκεκριμένο ζευγάρι συγχώνευσης σε μια λίστα από symbols."""
    new_symbols = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            new_symbols.append(pair[0] + pair[1])
            i += 2
        else:
            new_symbols.append(symbols[i])
            i += 1
    return new_symbols


# --- 3. Training (BPE) ---

def build_vocab(text):
    """Μετράει πόσες φορές εμφανίζεται κάθε λέξη."""
    vocab = {}
    for word in pre_tokenize(text):
        vocab[word] = vocab.get(word, 0) + 1
    return vocab


def train_bpe(tok, text, num_merges, verbose=False):
    """Incremental training: εφαρμόζει τα υπάρχοντα merges και προσθέτει έως num_merges νέα.

    Αντί να ξαναμετράει όλα τα ζευγάρια σε κάθε βήμα, κρατάει τα counts και σε ποιες λέξεις
    εμφανίζεται κάθε ζευγάρι, και ενημερώνει μόνο τις λέξεις που αλλάζουν.
    """
    tok.add_alphabet("".join(pre_tokenize(text)))
    return train_bpe_freqs(tok, build_vocab(text), num_merges, verbose)


def train_bpe_freqs(tok, word_freqs, num_merges, verbose=False):
    """Το ίδιο training, αλλά από έτοιμες συχνότητες λέξεων {λέξη: count} (για μεγάλα, ζυγισμένα μείγματα).
    Το alphabet πρέπει να έχει μπει ήδη: ό,τι λείπει από αυτό γίνεται bytes."""
    words = [tok.bpe(tok.split_chars(w)) for w in word_freqs]
    freqs = list(word_freqs.values())

    pair_counts, where = {}, {}
    for idx, symbols in enumerate(words):
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + freqs[idx]
            where.setdefault(pair, set()).add(idx)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    for i in range(num_merges):
        # Lazy heap: πετάμε εγγραφές που δεν ταιριάζουν πια με το τρέχον count
        while heap and -heap[0][0] != pair_counts.get(heap[0][1], 0):
            heapq.heappop(heap)
        if not heap:
            print(f"⏹️  Δεν υπάρχουν άλλα ζευγάρια μετά από {i} νέα merges.")
            break
        count, best = heapq.heappop(heap)
        tok.add_merge(best)
        if verbose or i < 10 or (i + 1) % 1000 == 0:
            print(f"Merge {len(tok.merges)}: {best} ({-count})")

        changed = {}
        for idx in where.pop(best, ()):
            old = words[idx]
            new = encode_word(old, best)
            words[idx] = new
            for pair in zip(old, old[1:]):
                changed[pair] = changed.get(pair, 0) - freqs[idx]
                where.get(pair, set()).discard(idx)
            for pair in zip(new, new[1:]):
                changed[pair] = changed.get(pair, 0) + freqs[idx]
                where.setdefault(pair, set()).add(idx)
        pair_counts.pop(best, None)

        for pair, delta in changed.items():
            if pair == best or delta == 0:
                continue
            pair_counts[pair] = pair_counts.get(pair, 0) + delta
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
            else:
                heapq.heappush(heap, (-pair_counts[pair], pair))

    return tok


# --- 4. Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Mini BPE tokenizer")
    parser.add_argument("--text", default="text.txt", help="κείμενο εκπαίδευσης")
    parser.add_argument("--model", default="merges.json", help="αρχείο μοντέλου")
    parser.add_argument("--merges", type=int, default=14000, help="πόσα νέα merges να εκπαιδεύσει (0 = μόνο encode)")
    parser.add_argument("--test", default="Kalispera sas, me lene Petro kai auto einai ena dokimastiko keimeno gia training tou Tokenizer gia to vocab tou Topografic MoE pou prospatho na ftiakso. Euxaristo poli", help="πρόταση δοκιμής")
    parser.add_argument("--verbose", action="store_true", help="τύπωσε κάθε merge")
    args = parser.parse_args()

    tok = Tokenizer.load(args.model)

    # A. Εκπαίδευση (Incremental Training)
    if args.merges > 0:
        text = load_text(args.text)
        train_bpe(tok, text, args.merges, args.verbose)
        tok.save(args.model)

    # B. Δοκιμή της ENCODE / DECODE σε ΝΕΟ ΚΕΙΜΕΝΟ!
    print("\n" + "=" * 40)
    print(f"🧪 Νέο Κείμενο Δοκιμής: '{args.test}'")
    ids = tok.encode(args.test)
    print("\n🎯 Final Tokens:")
    print(tok.tokenize(args.test))
    print(f"🔢 IDs: {ids}")
    print(f"🔁 Decode: '{tok.decode(ids)}'")


if __name__ == "__main__":
    main()

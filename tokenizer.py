import os
import json
import re

PRE = re.compile(r"[^\W\d_]+|\d+|\S")

def pre_tokenize(text):
    return PRE.findall(text)

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


def load_model(filename="merges.json"):
    """Φορτώνει τα προηγούμενα merges και το alphabet από το JSON."""
    if os.path.exists(filename):
        print(f"📂 Φορτώθηκε το '{filename}'!")
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            # Παλιό format (μόνο λίστα merges): το alphabet βγαίνει από τα merges
            alphabet = {ch for item in data for tok in item["pair"] if tok != "</w>" for ch in tok}
            return data, alphabet
        return data["merges"], set(data["alphabet"])
    print(f"🆕 Το αρχείο '{filename}' δεν βρέθηκε. Ξεκινάμε εκπαίδευση από το μηδέν!")
    return [], set()


def save_model(merges, alphabet, filename="merges.json"):
    """Αποθηκεύει τα merges και το alphabet στο JSON."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"alphabet": sorted(alphabet), "merges": merges}, f, ensure_ascii=False, indent=2)
    print(f"💾 Το ανανεωμένο Tokenizer αποθηκεύτηκε στο '{filename}'")


# --- 2. Βασικές Συναρτήσεις Training (BPE) ---

def build_vocab(text):
    vocab = {}
    for word in pre_tokenize(text):
        symbols = tuple(word) + ("</w>",)
        vocab[symbols] = vocab.get(symbols, 0) + 1
    return vocab


def get_stats(vocab):
    pairs = {}
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


def merge_vocab(pair, vocab):
    v_out = {}
    for symbols, freq in vocab.items():
        new_symbols = tuple(encode_word(symbols, pair))
        v_out[new_symbols] = v_out.get(new_symbols, 0) + freq
    return v_out


def apply_existing_merges(vocab, merges):
    for item in merges:
        pair = tuple(item["pair"])
        vocab = merge_vocab(pair, vocab)
    return vocab


def train_bpe(vocab, merges, num_merges):
    offset = len(merges)
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges.append({"id": offset + i, "pair": list(best_pair)})
        print(f"Merge {offset + i + 1}: {best_pair}")
    return vocab, merges

def build_alphabet(text):
    return {ch for ch in text if not ch.isspace()}

def split_chars(word, alphabet):
    out = []
    for ch in word:
        if ch in alphabet:
            out.append(ch)
        else:
            out.extend(f"<{b:02x}>" for b in ch.encode("utf-8"))
    return out


# --- 3. Η Συνάρτηση ENCODE (Inference) ---

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


def encode(text, merges, alphabet, cache=None):
    """Μετατρέπει ένα εντελώς νέο string σε Tokens με βάση τα εκπαιδευμένα merges."""
    if cache is None:
        cache = {}
    encoded_tokens = []

    for word in pre_tokenize(text):
        if word not in cache:
            # 1. Αρχικό σπάσιμο σε χαρακτήρες (ή bytes για άγνωστους) + </w>
            symbols = split_chars(word, alphabet) + ["</w>"]

            # 2. Εφαρμογή των merges με τη σειρά που εκπαιδεύτηκαν (by ID)
            for item in merges:
                symbols = encode_word(symbols, tuple(item["pair"]))
            cache[word] = symbols
        encoded_tokens.extend(cache[word])

    return encoded_tokens


# --- 4. Main Execution ---

def main():
    json_file = "merges.json"
    txt_file = "text.txt"
    new_merges_to_train = 5

    # A. Διαβάζουμε το κείμενο εκπαίδευσης & τα παλιά merges
    text = load_text(txt_file)
    vocab = build_vocab(text)
    merges, alphabet = load_model(json_file)
    alphabet |= build_alphabet(text)

    # B. Εκπαίδευση (Incremental Training)
    vocab = apply_existing_merges(vocab, merges)
    vocab, merges = train_bpe(vocab, merges, new_merges_to_train)
    save_model(merges, alphabet, json_file)

    # C. Δοκιμή της ENCODE σε ΝΕΟ ΚΕΙΜΕΝΟ!
    print("\n" + "=" * 40)
    test_sentence = "I like green bananas and apples"
    print(f"🧪 Νέο Κείμενο Δοκιμής: '{test_sentence}'")
    
    tokens = encode(test_sentence, merges, alphabet)
    print("\n🎯 Final Tokens:")
    print(tokens)


if __name__ == "__main__":
    main()
import os
import json

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


def load_merges(filename="merges.json"):
    """Φορτώνει τα προηγούμενα merges από το JSON."""
    if os.path.exists(filename):
        print(f"📂 Φορτώθηκε το '{filename}'!")
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"🆕 Το αρχείο '{filename}' δεν βρέθηκε. Ξεκινάμε εκπαίδευση από το μηδέν!")
    return []


def save_merges(merges, filename="merges.json"):
    """Αποθηκεύει τα merges στο JSON."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(merges, f, ensure_ascii=False, indent=2)
    print(f"💾 Το ανανεωμένο Tokenizer αποθηκεύτηκε στο '{filename}'")


# --- 2. Βασικές Συναρτήσεις Training (BPE) ---

def build_vocab(text):
    words = text.split()
    vocab = {}
    for word in words:
        formatted_word = " ".join(list(word)) + " </w>"
        vocab[formatted_word] = vocab.get(formatted_word, 0) + 1
    return vocab


def get_stats(vocab):
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


def merge_vocab(pair, vocab):
    v_out = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        v_out[new_word] = vocab[word]
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


def encode(text, merges):
    """Μετατρέπει ένα εντελώς νέο string σε Tokens με βάση τα εκπαιδευμένα merges."""
    words = text.split()
    encoded_tokens = []

    for word in words:
        # 1. Αρχικό σπάσιμο σε χαρακτήρες + </w>
        symbols = list(word) + ["</w>"]

        # 2. Εφαρμογή των merges με τη σειρά που εκπαιδεύτηκαν (by ID)
        for item in merges:
            pair = tuple(item["pair"])
            symbols = encode_word(symbols, pair)

        encoded_tokens.extend(symbols)

    return encoded_tokens


# --- 4. Main Execution ---

def main():
    json_file = "merges.json"
    txt_file = "text.txt"
    new_merges_to_train = 5

    # A. Διαβάζουμε το κείμενο εκπαίδευσης & τα παλιά merges
    text = load_text(txt_file)
    vocab = build_vocab(text)
    merges = load_merges(json_file)

    # B. Εκπαίδευση (Incremental Training)
    vocab = apply_existing_merges(vocab, merges)
    vocab, merges = train_bpe(vocab, merges, new_merges_to_train)
    save_merges(merges, json_file)

    # C. Δοκιμή της ENCODE σε ΝΕΟ ΚΕΙΜΕΝΟ!
    print("\n" + "=" * 40)
    test_sentence = "I like green bananas and apples"
    print(f"🧪 Νέο Κείμενο Δοκιμής: '{test_sentence}'")
    
    tokens = encode(test_sentence, merges)
    print("\n🎯 Final Tokens:")
    print(tokens)


if __name__ == "__main__":
    main()
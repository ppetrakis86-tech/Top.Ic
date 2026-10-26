import json
import os
import tempfile
import unittest

from tokenizer import SPACE, Tokenizer, build_vocab, encode_word, has_digit_merge, pre_tokenize, train_bpe

TEXT = ("My vegetables list: bananas, red apples, green oranges, broccoli, beans.\n"
        "Καλημέρα, καλησπέρα!  my_var = 42\r\n\tτέλος ")


def naive_counts(text, merges):
    """Αργή αναφορά: εφαρμόζει τα merges ένα-ένα και ξαναμετράει όλα τα ζευγάρια."""
    words = {tuple(w): f for w, f in build_vocab(text).items()}
    counts = []
    for pair in merges:
        stats = {}
        for symbols, f in words.items():
            for p in zip(symbols, symbols[1:]):
                stats[p] = stats.get(p, 0) + f
        counts.append((stats.get(pair, 0), max(stats.values())))
        words = {tuple(encode_word(list(s), pair)): f for s, f in words.items()}
    return counts


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = train_bpe(Tokenizer(), TEXT, 60)

    def test_pre_tokenize(self):
        S = SPACE
        self.assertEqual(pre_tokenize("my_var = 42, γεια!\n"),
                         ["my", "_", "var", S + "=", S, "4", "2", ",", S + "γεια", "!", "\n"])

    def test_pre_tokenize_keeps_everything(self):
        self.assertEqual("".join(pre_tokenize(TEXT)), TEXT.replace(" ", SPACE))

    def test_training_picks_most_frequent_pair(self):
        pairs = [tuple(m["pair"]) for m in self.tok.merges]
        for chosen, best in naive_counts(TEXT, pairs):
            self.assertEqual(chosen, best)

    def test_roundtrip_exact(self):
        for text in [TEXT, "bananas, red apples", "  διπλά  κενά  ", "άγνωστα: 😀 ñ 中文",
                     "x_1 = 42", "<ce> δεν είναι byte", "γραμμή\r\nνέα\tγραμμή\n\n"]:
            self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_unknown_chars_fall_back_to_bytes(self):
        self.assertEqual(self.tok.tokenize("😀"), ["<f0>", "<9f>", "<98>", "<80>"])

    def test_no_byte_fragments_in_merges(self):
        for m in self.tok.merges:
            for t in m["pair"]:
                self.assertNotRegex(t, r"<[0-9a-f]{2}>")

    def test_ids_stable_after_save_load(self):
        path = os.path.join(tempfile.mkdtemp(), "m.json")
        self.tok.save(path)
        loaded = Tokenizer.load(path)
        self.assertEqual(loaded.encode(TEXT), self.tok.encode(TEXT))
        train_bpe(loaded, TEXT + " πορτοκάλια", 10)
        self.assertEqual(loaded.vocab[:len(self.tok.vocab)], self.tok.vocab)

    def test_old_model_rejected(self):
        path = os.path.join(tempfile.mkdtemp(), "old.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"id": 0, "pair": ["a", "</w>"]}], f)
        with self.assertRaises(SystemExit):
            Tokenizer.load(path)


    def test_digits_are_single_chars(self):
        tok = train_bpe(Tokenizer(), "39+27=66 και 100 + 2026 = 2126 " * 50, 200)
        self.assertEqual(tok.tokenize("39+27"), ["3", "9", "+", "2", "7"])
        self.assertFalse([t for t in tok.vocab if has_digit_merge(t)])
        self.assertEqual(tok.decode(tok.encode(" 15 + 15 = 30")), " 15 + 15 = 30")

    def test_digit_merges_dropped_on_load(self):
        path = os.path.join(tempfile.mkdtemp(), "digits.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"alphabet": ["3", "9", "a", "b", SPACE],
                       "merges": [{"id": 0, "pair": ["3", "9"]}, {"id": 1, "pair": ["a", "b"]},
                                  {"id": 2, "pair": [SPACE, "3"]}]}, f)
        tok = Tokenizer.load(path)
        self.assertEqual([m["pair"] for m in tok.merges], [["a", "b"]])
        self.assertFalse([t for t in tok.vocab if has_digit_merge(t)])
        self.assertIn("<39>", tok.vocab)   # τα byte tokens δεν είναι «ψηφία»


if __name__ == "__main__":
    unittest.main()

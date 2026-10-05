# mini-bpe-tokenizer

Ένας μικρός tokenizer **Byte Pair Encoding (BPE)** σε καθαρή Python, χωρίς εξαρτήσεις. Φτιαγμένος για να καταλάβω πώς δουλεύουν οι tokenizers των γλωσσικών μοντέλων.

## Πώς δουλεύει

1. **Pre-tokenization**: το κείμενο χωρίζεται σε λέξεις, αριθμούς και σύμβολα.
   `"bananas, 42!"` → `["bananas", ",", "42", "!"]`
2. **Training**: κάθε λέξη σπάει σε χαρακτήρες, με ένα `</w>` στο τέλος. Σε κάθε βήμα, το πιο συχνό ζευγάρι γειτονικών symbols ενώνεται σε ένα νέο token (π.χ. `a` + `n` → `an`).
3. **Encode**: ένα καινούργιο κείμενο σπάει σε χαρακτήρες και τα merges εφαρμόζονται με τη σειρά που εκπαιδεύτηκαν. Όσοι χαρακτήρες δεν εμφανίστηκαν ποτέ στο training (π.χ. emoji) σπάνε σε UTF-8 bytes (`<f0>`, `<9f>` …), οπότε ποτέ δεν υπάρχει "άγνωστο" σύμβολο.

Το training είναι **incremental**. Κάθε τρέξιμο φορτώνει τα προηγούμενα merges και προσθέτει καινούργια, χωρίς να ξεκινάει από το μηδέν.

## Χρήση

Χρειάζεται Python 3.8+.

```bash
python3 tokenizer.py
```

Σε κάθε τρέξιμο:
- διαβάζει το κείμενο εκπαίδευσης από το `text.txt` (αν δεν υπάρχει, το δημιουργεί με ένα κείμενο παράδειγμα)
- φορτώνει το `merges.json`, αν υπάρχει
- εκπαιδεύει 5 νέα merges (`new_merges_to_train` στο `main()`)
- αποθηκεύει το αποτέλεσμα στο `merges.json`
- κάνει encode σε μια πρόταση δοκιμής και τυπώνει τα tokens

Παράδειγμα εξόδου:

```
Merge 26: ('banan', 'a')
...
🎯 Final Tokens:
['I</w>', 'like</w>', 'green</w>', 'bananas</w>', 'and</w>', 'apples</w>']
```

Από κώδικα:

```python
from tokenizer import load_model, encode

merges, alphabet = load_model("merges.json")
print(encode("I like green bananas", merges, alphabet))
```

## Αρχεία

| Αρχείο | Περιεχόμενο |
|---|---|
| `tokenizer.py` | Όλος ο κώδικας: I/O, training, encode |
| `text.txt` | Το κείμενο εκπαίδευσης |
| `merges.json` | Το εκπαιδευμένο μοντέλο: `alphabet` (γνωστοί χαρακτήρες) και `merges` (με σειρά) |

## Επόμενα βήματα

- [ ] Token IDs: `encode` να επιστρέφει αριθμούς αντί για strings
- [ ] `decode`, και τεστ ότι `decode(encode(x)) == x`
- [ ] Πλήρες byte-level BPE (όπως στο GPT-2)
- [ ] Γρηγορότερο encode με ranks αντί για πέρασμα όλων των merges
- [ ] Παράμετροι από command line (αρχείο εισόδου, αριθμός merges)

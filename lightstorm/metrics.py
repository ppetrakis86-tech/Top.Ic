"""Μετρικές του Γύρου 1. Γράφτηκαν ΠΡΙΝ από το μοντέλο (κανόνας: οι μετρικές ορίζονται πριν).

Ορισμοί: METRICS.md. Κάθε σταθερά εδώ είναι κλειδωμένη για τον Γύρο 1.
"""
import numpy as np
import torch

SIDE = 8
N_EXPERTS = SIDE * SIDE
IDLE_ALARM = 6          # συναγερμός άνεργων, δοκιμαστικό (Πέτρος, 25/10)
ACTIVE_SHARE = 1e-3     # ένας expert «δουλεύει» αν παίρνει >= 0,1% των tokens στον ρόλο του
GOLD_RANDOM_SEED = 971
SHUFFLE_SEEDS = (0, 1, 2, 3, 4)


def torus_coords(side=SIDE):
    """Expert i κάθεται στο (i // side, i % side)."""
    return torch.tensor([(i // side, i % side) for i in range(side * side)])


def torus_distances(side=SIDE):
    """Πίνακας D (64×64): ευκλείδεια απόσταση πάνω στον τόρο. Κοινός για forward, SOM, μετρική."""
    c = torus_coords(side).float()
    diff = (c[:, None, :] - c[None, :, :]).abs()
    diff = torch.minimum(diff, side - diff)
    return diff.pow(2).sum(-1).sqrt()


def cross(side=SIDE):
    """Για κάθε expert: [ο ίδιος, πάνω, κάτω, αριστερά, δεξιά] στον τόρο (k=5)."""
    out = []
    for i in range(side * side):
        r, c = divmod(i, side)
        out.append([i, ((r - 1) % side) * side + c, ((r + 1) % side) * side + c,
                    r * side + (c - 1) % side, r * side + (c + 1) % side])
    return torch.tensor(out)


def usage(winner_counts, member_counts, total_tokens):
    """Νικητές / γείτονες / άνεργοι.

    winner_counts[j]: πόσες φορές ο j ήταν ο νικητής (top-1).
    member_counts[j]: πόσες φορές ο j ήταν μέσα στα k=5 που έτρεξαν.
    Νικητής: winner share >= ACTIVE_SHARE. Γείτονας: όχι νικητής, member share >= ACTIVE_SHARE.
    Άνεργος: ούτε το ένα ούτε το άλλο.
    """
    w = np.asarray(winner_counts, dtype=np.float64) / max(total_tokens, 1)
    m = np.asarray(member_counts, dtype=np.float64) / max(total_tokens, 1)
    winners = w >= ACTIVE_SHARE
    neighbours = ~winners & (m >= ACTIVE_SHARE)
    idle = ~winners & ~neighbours
    return {"winners": int(winners.sum()), "neighbours": int(neighbours.sum()),
            "idle": int(idle.sum()), "idle_alarm": bool(idle.sum() >= IDLE_ALARM)}


def home_positions(token_ids, winners, vocab_size, n_experts=N_EXPERTS):
    """Η «διεύθυνση» κάθε token στο φύλλο = ο expert που το κερδίζει πιο συχνά (mode)."""
    hist = np.zeros((vocab_size, n_experts), dtype=np.int64)
    np.add.at(hist, (np.asarray(token_ids), np.asarray(winners)), 1)
    home = hist.argmax(1)
    home[hist.sum(1) == 0] = -1          # token που δεν εμφανίστηκε
    return home


def gold_distance(home, pairs, D):
    """Μέση απόσταση στο φύλλο για τα αληθινά ζευγάρια ÷ μέση απόσταση για τυχαία.

    pairs: λίστα (greek_id, greeklish_id). Τα τυχαία: ίδιες λέξεις, ανακατεμένοι
    σύντροφοι (σταθερό seed), ίδιος αριθμός. < 1 σημαίνει ότι τα ζευγάρια κάθονται κοντά.
    """
    D = np.asarray(D)
    pairs = [(g, l) for g, l in pairs if home[g] >= 0 and home[l] >= 0]
    if not pairs:
        return {"pairs": 0, "gold": float("nan"), "random": float("nan"), "ratio": float("nan")}
    g = np.array([p[0] for p in pairs])
    l = np.array([p[1] for p in pairs])
    rng = np.random.default_rng(GOLD_RANDOM_SEED)
    l_rand = l[rng.permutation(len(l))]
    gold = D[home[g], home[l]].mean()
    rand = D[home[g], home[l_rand]].mean()
    return {"pairs": len(pairs), "gold": float(gold), "random": float(rand),
            "ratio": float(gold / rand) if rand > 0 else float("nan")}


def tokens_to_target(history, target):
    """Πρώτο σημείο (σε tokens) όπου το val loss έπεσε <= target. None αν δεν έφτασε ποτέ.

    history: λίστα (tokens_seen, val_loss). Το target ορίζεται πριν: το τελικό val loss του A.
    """
    for seen, loss in history:
        if loss <= target:
            return seen
    return None


def shuffle_permutations(n=N_EXPERTS, seeds=SHUFFLE_SEEDS):
    """Οι μεταθέσεις του τεστ ανακατέματος, σταθερές για όλα τα σκέλη."""
    return [torch.from_numpy(np.random.default_rng(s).permutation(n)) for s in seeds]

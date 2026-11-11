"""G2: νέα γνώση εκπαιδεύει μόνο τη γειτονιά της. Συνέχεια από checkpoint της Φάσης 1 ΜΟΝΟ στο train μιας νέας πηγής,
με εκπαιδεύσιμους μόνο 5 (ή όλους τους) ειδικούς. Όλα τα άλλα παγωμένα· τα κλειδιά του B παγωμένα (χωρίς som_step),
το σ στο τελικό του (model.load_checkpoint).

Σκέλη (--mode): local  B: νικητής στη νέα πηγή + σταυρός · A: οι 5 με τη μεγαλύτερη μάζα routing (top-5, κανονικοποιημένα βάρη)
                 shuffle B: ίδιος νικητής + 4 τυχαίοι ΜΗ γείτονες (seed του run)
                 all    όλοι οι ειδικοί
Επιλογή: ένα πέρασμα --select-tokens του train της νέας πηγής (ισαπέχοντα παράθυρα) στο παγωμένο μοντέλο, πριν την ενημέρωση.
Παγωμένα: requires_grad=False παντού εκτός από experts.{w1,b1,w2,b2}· μετά από κάθε βήμα οι γραμμές των μη επιλεγμένων
ειδικών επανέρχονται ακριβώς (ούτε weight decay τις αγγίζει).
Μέτρα: val νέας πηγής και val παλιών πηγών στα --eval-at tokens. Μ = νέο(0) − νέο(τέλος), Ξ = παλιό(τέλος) − παλιό(0).

    python -m lightstorm.finetune --ckpt RUN/B/epoch1.pt --mode local --arm B-local --out RUN/ft --corpus corpus_r3 \
        --manifest r3_data_manifest.json --new uesp_en --old wiki_el,wiki_en,greeklish,clean --seed 0
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import load_checkpoint
from .train import batches, evaluate, load_bin

EXPERT_PARAMS = ("w1", "b1", "w2", "b2")


def segments(manifest):
    """όνομα πηγής → (train_από, train_έως, val_από, val_έως), με τη σειρά του manifest (όπως το train.subset)."""
    out, a, b = {}, 0, 0
    for k, v in json.load(open(manifest))["sources"].items():
        if isinstance(v, dict) and "train_tokens" in v:
            out[k] = (a, a + v["train_tokens"], b, b + v["val_tokens"])
            a += v["train_tokens"]
            b += v["val_tokens"]
    return out


@torch.no_grad()
def routing(m, data, n_tokens, device, batch=32):
    """Νίκες ανά ειδικό και μάζα routing (άθροισμα των βαρών με τα οποία κάθε ειδικός μπαίνει στην έξοδο)."""
    ctx, n = m.cfg.ctx, m.cfg.side ** 2
    starts = np.linspace(0, len(data) - ctx - 1, max(1, n_tokens // ctx)).astype(np.int64)
    grab = {}
    h = m.moe.register_forward_hook(lambda mod, inp, out: grab.__setitem__("x", inp[0].reshape(-1, inp[0].shape[-1])))
    wins, mass = torch.zeros(n), torch.zeros(n)
    for i in range(0, len(starts), batch):
        x = torch.stack([data[s:s + ctx] for s in starts[i:i + batch]]).to(device)
        _, _, aux = m(x)
        wins += torch.bincount(aux["winner"].cpu(), minlength=n).float()
        if m.cfg.arm == "A":
            probs = m.moe.gate(grab["x"]).softmax(-1)
            w, idx = probs.topk(m.cfg.k, dim=-1)
            w = w / w.sum(-1, keepdim=True)
        else:
            mem = aux["members"]
            d = m.moe.D[mem[:, :1], mem]
            w = torch.exp(-d.pow(2) / (2 * m.moe.sigma ** 2))
            w, idx = w / w.sum(-1, keepdim=True), mem
        mass += torch.zeros(n, device=w.device).index_add_(0, idx.reshape(-1), w.reshape(-1).float()).cpu()
    h.remove()
    return wins, mass


def choose(m, mode, wins, mass, seed):
    n = m.cfg.side ** 2
    if mode == "all":
        return list(range(n))
    if m.cfg.arm == "A":
        assert mode == "local", "το A έχει μόνο local/all"
        return mass.topk(5).indices.tolist()
    win = int(wins.argmax())
    cross = m.moe.cross0[win].tolist()                     # [νικητής, πάνω, κάτω, αριστερά, δεξιά]
    if mode == "local":
        return cross
    others = [e for e in range(n) if e not in cross]
    g = torch.Generator().manual_seed(9000 + seed)
    return [win] + [others[i] for i in torch.randperm(len(others), generator=g)[:4].tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=("local", "shuffle", "all"), required=True)
    ap.add_argument("--arm", required=True, help="όνομα σκέλους, π.χ. B-local (φάκελος εξόδου)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="corpus_r3")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--new", default="uesp_en")
    ap.add_argument("--old", default="wiki_el,wiki_en,greeklish,clean")
    ap.add_argument("--tokens", type=float, default=5e6)
    ap.add_argument("--eval-at", default="0,1e6,2.5e6,5e6")
    ap.add_argument("--select-tokens", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=1e-4, help="σταθερό· = το τελικό lr της Φάσης 1 (0,1 × 1e-3)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--trace", default=None, help="για τη συμβατότητα με την ουρά (γράφει ένα βήμα ανά γραμμή)")
    ap.add_argument("--max-steps", type=int, default=0, help="μόνο για δοκιμές")
    a = ap.parse_args()
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, ck = load_checkpoint(a.ckpt, device)
    assert a.mode != "shuffle" or m.cfg.arm != "A", "το shuffle είναι μόνο για το B"
    seg = segments(a.manifest)
    train, val = load_bin(os.path.join(a.corpus, "train.bin")), load_bin(os.path.join(a.corpus, "val.bin"))
    assert seg and max(s[1] for s in seg.values()) == len(train) and max(s[3] for s in seg.values()) == len(val), \
        "το corpus δεν ταιριάζει με το manifest"
    t0_, t1_, v0_, v1_ = seg[a.new]
    new_train, new_val = train[t0_:t1_], val[v0_:v1_]
    old_val = torch.cat([val[seg[k][2]:seg[k][3]] for k in a.old.split(",")])
    del train

    wins, mass = routing(m, new_train, a.select_tokens, device)
    chosen = choose(m, a.mode, wins, mass, a.seed)
    win = int(wins.argmax())
    cross = m.moe.cross0[win].tolist() if m.cfg.arm != "A" else None
    info = {"arm": a.arm, "mode": a.mode, "base_arm": m.cfg.arm, "ckpt": a.ckpt, "chosen": chosen, "lr": a.lr,
            "sigma": getattr(m.moe, "sigma", None), "win_share": (wins / wins.sum()).tolist(),
            "mass_share": (mass / mass.sum()).tolist(),
            "chosen_win_share": float(wins[chosen].sum() / wins.sum()),
            "chosen_mass_share": float(mass[chosen].sum() / mass.sum()),
            "cross_win_share": float(wins[cross].sum() / wins.sum()) if cross else None}
    print(f"{a.arm}: ειδικοί {chosen} · νίκες τους {info['chosen_win_share']:.1%} · μάζα {info['chosen_mass_share']:.1%}", flush=True)

    for p in m.parameters():
        p.requires_grad_(False)
    ex = m.moe.experts
    params = [getattr(ex, k) for k in EXPERT_PARAMS]
    keep = torch.ones(m.cfg.side ** 2, dtype=torch.bool, device=device)
    keep[chosen] = False                                    # keep = παγωμένοι
    frozen = [p.detach().clone() for p in params]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))

    tps = a.batch * m.cfg.ctx
    marks = sorted({math.ceil(float(t) / tps) for t in a.eval_at.split(",")})
    total = a.max_steps or marks[-1]
    hist = []

    def measure(step):
        r_new = evaluate(m, new_val, m.cfg.ctx, device, {})["val_loss"]
        r_old = evaluate(m, old_val, m.cfg.ctx, device, {})["val_loss"]
        hist.append({"step": step, "tokens": step * tps, "val_new": r_new, "val_old": r_old})
        print(f"{a.arm} βήμα {step:5d} · {step * tps:,} tok · νέο {r_new:.4f} · παλιό {r_old:.4f}", flush=True)

    measure(0)
    m.train()
    trace = open(a.trace, "w") if a.trace else None
    t0 = time.time()
    for step, (x, y) in enumerate(batches(new_train, m.cfg.ctx, a.batch, total, a.seed), start=1):
        x, y = x.to(device), y.to(device)
        logits, _, _ = m(x, y)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))   # χωρίς lb: το gate είναι παγωμένο
        opt.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():                               # οι μη επιλεγμένοι ειδικοί μένουν ακριβώς ίδιοι
            for p, f in zip(params, frozen):
                p[keep] = f[keep]
        if trace is not None:
            trace.write(f"{step}\t{ce.item():.4f}\n")
            trace.flush()
        if step in marks or step == total:
            measure(step)
            m.train()
    m.eval()
    with torch.no_grad():
        changed = [e for e in range(len(keep)) if any(bool((p[e] != f[e]).any()) for p, f in zip(params, frozen))]
    first, last = hist[0], hist[-1]
    info.update({"history": hist, "M": first["val_new"] - last["val_new"], "X": last["val_old"] - first["val_old"],
                 "experts_changed": changed, "seconds": time.time() - t0})
    assert set(info["experts_changed"]) <= set(chosen), "άλλαξε ειδικός που έπρεπε να μείνει παγωμένος"
    os.makedirs(os.path.join(a.out, a.arm), exist_ok=True)
    json.dump(info, open(os.path.join(a.out, a.arm, "finetune.json"), "w"), indent=1, ensure_ascii=False)
    print(f"✅ {a.arm}: Μ {info['M']:+.4f} · Ξ {info['X']:+.4f} · {info['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Training ενός σκέλους. Ίδια σειρά batches σε όλα τα σκέλη (σταθερό seed δεδομένων).

    python3 -m lightstorm.train --arm B            # 2 εποχές, checkpoint κάθε εποχή
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from . import metrics
from .model import ARMS, Config, Lightstorm, count_params, schedule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_bin(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))


def batches(data, ctx, batch, steps, seed):
    """Τυχαία παράθυρα, ίδια για κάθε σκέλος αφού το seed είναι ίδιο."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        start = torch.randint(0, len(data) - ctx - 1, (batch,), generator=g)
        x = torch.stack([data[s:s + ctx] for s in start])
        y = torch.stack([data[s + 1:s + ctx + 1] for s in start])
        yield x, y


def read_gold(path):
    rows = [l.split("\t") for l in open(path, encoding="utf-8").read().splitlines()[1:] if l.strip()]
    return [(int(r[2]), int(r[3])) for r in rows]


@torch.no_grad()
def evaluate(model, val, ctx, device, gold_sets, batch=32):
    """Val loss + νικητές/γείτονες/άνεργοι + gold. Όλο το val, μη επικαλυπτόμενα παράθυρα.
    gold_sets: {όνομα: ζευγάρια}, π.χ. gold_open (ελληνικά↔greeklish) και gold_elen (πείραμα 6)."""
    model.eval()
    n = (len(val) - 1) // ctx
    xs = val[:n * ctx].view(n, ctx)
    ys = val[1:n * ctx + 1].view(n, ctx)
    tot, cnt = 0.0, 0
    win_all, tok_all = [], []
    side = model.cfg.side
    members = torch.zeros(side ** 2, dtype=torch.long)
    for i in range(0, n, batch):
        x, y = xs[i:i + batch].to(device), ys[i:i + batch].to(device)
        _, _, aux = model(x, y)
        tot += aux["ce"].item() * x.numel()
        cnt += x.numel()
        win_all.append(aux["winner"].cpu())
        tok_all.append(x.reshape(-1).cpu())
        members += torch.bincount(aux["members"].reshape(-1).cpu(), minlength=side ** 2)
    win = torch.cat(win_all)
    toks = torch.cat(tok_all)
    out = {"val_loss": tot / cnt}
    out.update(metrics.usage(torch.bincount(win, minlength=side ** 2).numpy(),
                             members.numpy(), len(win)))
    home = metrics.home_positions(toks.numpy(), win.numpy(), model.cfg.vocab, side ** 2)
    for name, pairs in gold_sets.items():
        out[name] = metrics.gold_distance(home, pairs, metrics.torus_distances(side).numpy())
    model.train()
    return out


@torch.no_grad()
def shuffle_test(model, val, ctx, device, base_loss, batch=32):
    """Ανακατεύει τις θέσεις στο φύλλο. Αν το val loss δεν αλλάζει, η τοπογραφία είναι διακοσμητική."""
    if model.cfg.arm == "A":
        return {"note": "το A δεν χρησιμοποιεί θέσεις, Δ = 0 εξ ορισμού"}
    deltas = []
    n = model.cfg.side ** 2
    for perm in metrics.shuffle_permutations(n):
        model.moe.set_permutation(perm)
        deltas.append(evaluate(model, val, ctx, device, {}, batch)["val_loss"] - base_loss)
    model.moe.set_permutation(torch.arange(n))
    return {"delta_mean": float(np.mean(deltas)), "delta_std": float(np.std(deltas)), "deltas": deltas}


# πείραμα 3: (d_model, layers, heads). FFN 1× και expert d→d/4→d όπως στο 7M.
SIZES = {"7M": (256, 4, 4), "25M": (512, 6, 8), "58M": (768, 8, 12), "110M": (1024, 10, 16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=25, help="σε βήματα")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--freeze-keys-after", type=int, default=0,
                    help="πείραμα 1α: μετά από τόσες εποχές τα κλειδιά παγώνουν (η=0). 0 = ποτέ")
    ap.add_argument("--moe-layer", type=int, default=2, help="πείραμα 2: 0-based θέση του MoE (2 = το 3ο layer)")
    ap.add_argument("--key-seed", type=int, default=None,
                    help="πείραμα 2β: seed μόνο για τα αρχικά κλειδιά του SOM (προεπιλογή = --seed)")
    ap.add_argument("--size", choices=list(SIZES), default="7M", help="πείραμα 3: μέγεθος μοντέλου")
    ap.add_argument("--side", type=int, default=8,
                    help="πείραμα 4: πλευρά του φύλλου. Ο συνολικός όγκος των experts μένει ίδιος")
    ap.add_argument("--max-steps", type=int, default=0, help="μόνο για δοκιμές")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key_seed = args.seed if args.key_seed is None else args.key_seed
    d, L, h = SIZES[args.size]
    cfg = Config(arm=args.arm, seed=key_seed, moe_layer=args.moe_layer, side=args.side,
                 d_model=d, n_layers=L, n_heads=h, ffn=d, expert_hidden=d // 4 * 64 // args.side ** 2)
    if args.size != "7M" and args.lr == 1e-3:
        args.lr = 1e-3 * 256 / d          # πείραμα 3: lr ∝ 1/πλάτος, ορισμένο στο METRICS.md
    model = Lightstorm(cfg).to(device)
    train = load_bin(os.path.join(args.corpus, "train.bin"))
    val = load_bin(os.path.join(args.corpus, "val.bin"))
    gold = {"gold_open": read_gold(os.path.join(args.corpus, "gold_open.tsv"))}     # ΠΟΤΕ το sealed εδώ
    if os.path.exists(os.path.join(args.corpus, "gold_elen.tsv")):                # πείραμα 6: el↔en
        gold["gold_elen"] = read_gold(os.path.join(args.corpus, "gold_elen.tsv"))

    tokens_per_step = args.batch * cfg.ctx
    steps_per_epoch = math.ceil(len(train) / tokens_per_step)
    total = steps_per_epoch * args.epochs if not args.max_steps else args.max_steps
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    lr_at = lambda s: args.lr * min(1, (s + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / total)))

    run = os.path.join(args.out, args.arm)
    os.makedirs(run, exist_ok=True)
    print(f"arm {args.arm} · {count_params(model):,} params · {len(train):,} train / {len(val):,} val tokens "
          f"· {steps_per_epoch} steps/εποχή · {device}")
    history, t0 = [], time.time()
    for step, (x, y) in enumerate(batches(train, cfg.ctx, args.batch, total, args.data_seed)):
        seen = step * tokens_per_step
        if args.arm != "A":
            model.moe.sigma, model.moe.eta = schedule(seen, steps_per_epoch * tokens_per_step,
                                                      sigma=(2.5 * cfg.side / 8, 0.7))   # αρχικό σ ∝ πλευρά
            if args.freeze_keys_after and step >= args.freeze_keys_after * steps_per_epoch:
                model.moe.eta = 0.0
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        _, loss, aux = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if args.arm != "A":
            model.moe.som_step()
        seen += tokens_per_step
        if (step + 1) % args.eval_every == 0 or step + 1 == total:
            ev = evaluate(model, val, cfg.ctx, device, gold)
            history.append((seen, ev["val_loss"]))
            alarm = " ⚠️ ΣΥΝΑΓΕΡΜΟΣ άνεργων" if ev["idle_alarm"] else ""
            print(f"step {step + 1:5d} · {seen:,} tok · train {aux['ce'].item():.3f} · val {ev['val_loss']:.3f} "
                  f"· νικ/γειτ/άνεργ {ev['winners']}/{ev['neighbours']}/{ev['idle']} "
                  f"· gold {ev['gold_open']['ratio']:.3f}"
                  + (f" · el↔en {ev['gold_elen']['ratio']:.3f}" if "gold_elen" in ev else "")
                  + f" · {time.time() - t0:.0f}s{alarm}")
        if (step + 1) % steps_per_epoch == 0 or step + 1 == total:
            epoch = math.ceil((step + 1) / steps_per_epoch)
            ev = evaluate(model, val, cfg.ctx, device, gold)
            ev["shuffle"] = shuffle_test(model, val, cfg.ctx, device, ev["val_loss"])
            ev.update({"arm": args.arm, "epoch": epoch, "step": step + 1, "tokens": seen, "history": history})
            if args.arm != "A":
                ev["sigma"], ev["eta"] = model.moe.sigma, model.moe.eta
            keep = epoch <= 2 or epoch % 10 == 0 if args.size == "7M" else False   # μεγάλα μοντέλα: μόνο το τελευταίο
            if keep or step + 1 == total:     # τα .pt είναι ~90 MB στο 7M, ~1,3 GB στο 110M
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg.__dict__,
                            "args": vars(args), "bench": ev}, os.path.join(run, f"epoch{epoch}.pt"))
            with open(os.path.join(run, f"epoch{epoch}.json"), "w", encoding="utf-8") as f:
                json.dump(ev, f, indent=1, ensure_ascii=False)
            print(f"💾 checkpoint εποχή {epoch} · shuffle Δ {ev['shuffle'].get('delta_mean', 0):+.4f}")


if __name__ == "__main__":
    main()

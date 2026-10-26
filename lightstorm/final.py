"""ΤΕΛΙΚΗ μέτρηση στο κλειστό κομμάτι των gold pairs. Τρέχει ΜΙΑ φορά, στο τέλος του Γύρου 1.

Ελέγχει ότι το gold_sealed.tsv είναι αυτό που δεσμεύτηκε στο data_manifest.json.

    python3 -m lightstorm.final --i-am-done runs/A/epoch2.pt runs/B/epoch2.pt ...
"""
import argparse
import hashlib
import json
import os

import torch

from .model import Config, Lightstorm
from .train import ROOT, evaluate, load_bin, read_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-am-done", action="store_true", required=True)
    ap.add_argument("ckpts", nargs="+")
    args = ap.parse_args()
    manifest = json.load(open(os.path.join(ROOT, "data_manifest.json")))
    sealed = os.path.join(ROOT, "corpus", "gold_sealed.tsv")
    assert hashlib.sha256(open(sealed, "rb").read()).hexdigest() == manifest["gold_sealed_sha256"], \
        "το κλειστό κλειδί άλλαξε από τη δέσμευση"
    gold = read_gold(sealed)
    val = load_bin(os.path.join(ROOT, "corpus", "val.bin"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for path in args.ckpts:
        ck = torch.load(path, map_location=device)
        model = Lightstorm(Config(**ck["cfg"])).to(device)
        model.load_state_dict(ck["model"])
        ev = evaluate(model, val, model.cfg.ctx, device, {"gold_sealed": gold})
        print(path, json.dumps(ev["gold_sealed"]), sep="\t")


if __name__ == "__main__":
    main()

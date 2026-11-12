#!/usr/bin/env bash
# Πείραμα 2β: B, MoE στο layer 1, 20 εποχές.
#   (i)  ίδιος χάρτης (κλειδιά seed 0), βάρη+batches seed 0–4  -> runs_e20_L1_samemap_s{0..4}/B/
#   (ii) μόνο ο χάρτης αλλάζει (κλειδιά seed 0–4), βάρη+batches seed 0 -> runs_e20_L1_onlymap_k{0..4}/B/
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
for s in 0 1 2 3 4; do
    out=runs_e20_L1_samemap_s$s; mkdir -p "$out"
    "$PY" -m lightstorm.train --arm B --epochs 20 --seed "$s" --data-seed "$s" --key-seed 0 \
        --moe-layer 0 --out "$out" > "$out/B.log" 2>&1
    out=runs_e20_L1_onlymap_k$s; mkdir -p "$out"
    "$PY" -m lightstorm.train --arm B --epochs 20 --seed 0 --data-seed 0 --key-seed "$s" \
        --moe-layer 0 --out "$out" > "$out/B.log" 2>&1
done
echo done > runs_e20_L1_onlymap_k4/done.txt

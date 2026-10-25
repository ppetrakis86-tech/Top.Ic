#!/usr/bin/env bash
# Πείραμα 2: MoE στο layer 1, σκέλη A και B, 20 εποχές, seeds 0 1 2.
#   ./run_layer1.sh   -> runs_e20_L1_s{0,1,2}/{A,B}/
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
for seed in 0 1 2; do
    out=runs_e20_L1_s$seed
    mkdir -p "$out"
    for arm in A B; do
        "$PY" -m lightstorm.train --arm "$arm" --epochs 20 --seed "$seed" --data-seed "$seed" \
            --moe-layer 0 --out "$out" > "$out/$arm.log" 2>&1
    done
done
echo done > runs_e20_L1_s2/done.txt

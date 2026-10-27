#!/usr/bin/env bash
# Πείραμα 2: MoE στο layer 1, σκέλη A και B, 20 εποχές, seeds 0 1 2.
#   ./run_layer1.sh   -> runs_e20_L1_s{0,1,2}/{A,B}/
#   SEEDS="3 4" ./run_layer1.sh   -> επέκταση σε 5 seeds
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
SEEDS=${SEEDS:-0 1 2}
for seed in $SEEDS; do
    out=runs_e20_L1_s$seed
    mkdir -p "$out"
    for arm in A B; do
        "$PY" -m lightstorm.train --arm "$arm" --epochs 20 --seed "$seed" --data-seed "$seed" \
            --moe-layer 0 --out "$out" > "$out/$arm.log" 2>&1
    done
done
echo done > "runs_e20_L1_s$seed/done.txt"

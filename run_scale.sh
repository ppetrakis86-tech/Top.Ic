#!/usr/bin/env bash
# Πείραμα 3: A και B σε 25M, 58M, 110M (το 7M είναι το runs_e20), seed 0, 20 εποχές, MoE στο 3ο layer.
#   ./run_scale.sh   -> runs_e20_size{25M,58M,110M}/{A,B}/
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # αλλιώς το 110M κάνει OOM στο WSL
for size in 25M 58M 110M; do
    out=runs_e20_size$size
    mkdir -p "$out"
    for arm in A B; do
        "$PY" -m lightstorm.train --arm "$arm" --epochs 20 --seed 0 --data-seed 0 --size "$size" \
            --out "$out" > "$out/$arm.log" 2>&1
    done
done
echo done > runs_e20_size110M/done.txt

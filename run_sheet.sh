#!/usr/bin/env bash
# Πείραμα 4: 110M με φύλλο 16×16 (ίδιος όγκος experts), A και B, seed 0, 20 εποχές.
#   ./run_sheet.sh   -> runs_e20_size110M_16x16/{A,B}/
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # αλλιώς το 110M κάνει OOM στο WSL
out=runs_e20_size110M_16x16
mkdir -p "$out"
for arm in A B; do
    "$PY" -m lightstorm.train --arm "$arm" --epochs 20 --seed 0 --data-seed 0 --size 110M --side 16 \
        --out "$out" > "$out/$arm.log" 2>&1
done
echo done > "$out/done.txt"

#!/usr/bin/env bash
# Πείραμα 5: σκάλα χάρτη στο 7M, 4×4 και 16×16 (το 8×8 υπάρχει ήδη), A και B, seeds 0–2, 20 εποχές.
#   ./run_sheet7m.sh   -> runs_e20_7M_{4x4,16x16}_s{0,1,2}/{A,B}/ και runs_e20_7M_sheets_done.txt
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PYTHONUNBUFFERED=1
for side in 4 16; do
    for s in 0 1 2; do
        out=runs_e20_7M_${side}x${side}_s$s
        mkdir -p "$out"
        for arm in A B; do
            [ -f "$out/$arm/epoch20.json" ] && continue   # ξανατρέχει μόνο ό,τι λείπει
            "$PY" -m lightstorm.train --arm "$arm" --epochs 20 --seed "$s" --data-seed "$s" --side "$side" \
                --out "$out" > "$out/$arm.log" 2>&1
        done
    done
done
echo done > runs_e20_7M_sheets_done.txt

#!/usr/bin/env bash
# Γύρος 1: τα 4 σκέλη το ένα μετά το άλλο, ίδια σειρά batches.
#   ./run_round1.sh                  2 εποχές, seed 0 -> runs/
#   EPOCHS=20 ./run_round1.sh                         -> runs_e20/
#   EPOCHS=20 SEED=1 ./run_round1.sh                  -> runs_e20_s1/
# Το SEED αλλάζει init, αρχικά κλειδιά SOM και σειρά batches (ίδια για όλα τα σκέλη του seed).
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
EPOCHS=${EPOCHS:-2}
SEED=${SEED:-0}
OUT=runs
[ "$EPOCHS" != 2 ] && OUT=runs_e$EPOCHS
[ "$SEED" != 0 ] && OUT=${OUT}_s$SEED
mkdir -p "$OUT"
rm -f "$OUT/done.txt"
for arm in A B C1 C2; do
    "$PY" -m lightstorm.train --arm "$arm" --epochs "$EPOCHS" --seed "$SEED" --data-seed "$SEED" --out "$OUT" > "$OUT/$arm.log" 2>&1
done
echo done > "$OUT/done.txt"

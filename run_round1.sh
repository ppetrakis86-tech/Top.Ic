#!/usr/bin/env bash
# Γύρος 1: τα 4 σκέλη το ένα μετά το άλλο, ίδια σειρά batches.
#   ./run_round1.sh          2 εποχές  -> runs/
#   EPOCHS=20 ./run_round1.sh          -> runs_e20/
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
EPOCHS=${EPOCHS:-2}
OUT=runs
[ "$EPOCHS" != 2 ] && OUT=runs_e$EPOCHS
mkdir -p "$OUT"
rm -f "$OUT/done.txt"
for arm in A B C1 C2; do
    "$PY" -m lightstorm.train --arm "$arm" --epochs "$EPOCHS" --out "$OUT" > "$OUT/$arm.log" 2>&1
done
echo done > "$OUT/done.txt"

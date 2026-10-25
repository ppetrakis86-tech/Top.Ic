#!/usr/bin/env bash
# Γύρος 1: τα 4 σκέλη το ένα μετά το άλλο, ίδια σειρά batches. Logs στο runs/<σκέλος>.log
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
mkdir -p runs
rm -f runs/done.txt
for arm in A B C1 C2; do
    "$PY" -m lightstorm.train --arm "$arm" > "runs/$arm.log" 2>&1
done
echo done > runs/done.txt

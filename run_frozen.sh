#!/usr/bin/env bash
# Πείραμα 1α: B με παγωμένα κλειδιά μετά την 1η εποχή, 20 εποχές, seeds 0 1 2.
#   ./run_frozen.sh   -> runs_e20_frozen_s{0,1,2}/B/
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
for seed in 0 1 2; do
    out=runs_e20_frozen_s$seed
    mkdir -p "$out"
    "$PY" -m lightstorm.train --arm B --epochs 20 --seed "$seed" --data-seed "$seed" \
        --freeze-keys-after 1 --out "$out" > "$out/B.log" 2>&1
done
echo done > runs_e20_frozen_s2/done.txt

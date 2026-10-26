#!/usr/bin/env bash
# Πείραμα 6: encode (μία φορά) + A και B στο 7M πάνω σε ελληνική + Simple English Wikipedia.
#   ./run_wiki.sh   -> corpus_wiki/{train,val}.bin, runs_wiki_elen/{A,B}/
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PYTHONUNBUFFERED=1
out=runs_wiki_elen
mkdir -p "$out"
if [ ! -s corpus_wiki/gold_elen.tsv ]; then                 # τα αρχεία της Wikipedia μένουν στο PC, δεν ξαναγίνονται
    "$PY" -m lightstorm.wiki --src el=corpus_wiki/parquet_el --src simple=corpus_wiki/parquet_simple \
        --cap 50000000 --val-cap 500000 --muse corpus_wiki/muse_el-en.txt > "$out/encode.log" 2>&1
fi
for arm in A B; do
    [ -f "$out/$arm/epoch1.json" ] && continue
    "$PY" -m lightstorm.train --arm "$arm" --epochs 1 --seed 0 --data-seed 0 --eval-every 1000 \
        --corpus corpus_wiki --out "$out" > "$out/$arm.log" 2>&1
done
echo done > "$out/done.txt"

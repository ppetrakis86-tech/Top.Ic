#!/usr/bin/env bash
# Πείραμα 6: encode (μία φορά) + A και B στο 7M πάνω σε ελληνική + Simple English Wikipedia.
#   ./run_wiki.sh              -> corpus_wiki/{train,val}.bin, runs_wiki_elen/{A,B}/
#   SEEDS="1 2" ./run_wiki.sh  -> runs_wiki_elen_s1/, runs_wiki_elen_s2/ (πείραμα 6β)
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PYTHONUNBUFFERED=1
mkdir -p runs_wiki_elen
if [ ! -s corpus_wiki/gold_elen.tsv ]; then                 # τα αρχεία της Wikipedia μένουν στο PC, δεν ξαναγίνονται
    "$PY" -m lightstorm.wiki --src el=corpus_wiki/parquet_el --src simple=corpus_wiki/parquet_simple \
        --cap 50000000 --val-cap 500000 --muse corpus_wiki/muse_el-en.txt > runs_wiki_elen/encode.log 2>&1
fi
for seed in ${SEEDS:-0}; do
    out=runs_wiki_elen
    [ "$seed" = 0 ] || out=runs_wiki_elen_s$seed
    mkdir -p "$out"
    for arm in A B; do
        [ -f "$out/$arm/epoch1.json" ] && continue
        "$PY" -m lightstorm.train --arm "$arm" --epochs 1 --seed "$seed" --data-seed "$seed" --eval-every 1000 \
            --corpus corpus_wiki --out "$out" > "$out/$arm.log" 2>&1
    done
    echo done > "$out/done.txt"
done

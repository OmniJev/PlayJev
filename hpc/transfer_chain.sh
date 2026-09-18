#!/bin/bash
# Data-efficiency chain for one held-out game: fine-tune on 1k / 3k / 10k random frames (hpc/transfer_ft.sh).
#   bash hpc/transfer_chain.sh racer 1000+3000+10000 base+hold8      (plus-separated lists: qsub -v splits on commas and strips quotes)
set -u
GAME=$1; SIZES=${2:-1000+3000+10000}; INITS=${3:-base+hold8}; SIZES=${SIZES//+/ }; INITS=${INITS//+/ }
for init in $INITS; do for n in $SIZES; do bash hpc/transfer_ft.sh $GAME $n $init; done; done
echo "[transfer chain $GAME] done"

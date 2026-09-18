#!/bin/bash
# System One / System Two sweep for one game: the model plays, and below a Jev-confidence threshold the teacher
# decides. tau 0 = the model alone, 1.01 = the teacher alone; the curve in between is what the confidence buys.
#   bash hpc/handover_sweep.sh snake [ckpt] [taus]
# Results: runs/play/<game>_local_handover<tau>.json (score_mean, handover_rate, ...), one per tau.
set -u
GAME=$1
CKPT=${2:-$WORK/ckpt/sft_all1/final}
TAUS=${3:-"0 0.2 0.4 0.6 0.8 1.01"}
for tau in $TAUS; do
    echo "[handover $GAME] tau $tau"
    python -u -m playjev.play $GAME --policy local --ckpt $CKPT --episodes 16 --pages 8 --seed0 5000 --max-steps 1500 --handover $tau
done
echo "[handover $GAME] done"

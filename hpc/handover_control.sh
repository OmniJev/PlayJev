#!/bin/bash
# Control for the handover curve: the same share of steps handed to the teacher at random (no confidence).
#   bash hpc/handover_control.sh snake "0.004 0.38 0.45 0.53"
set -u
GAME=$1; RATES=$2
CKPT=${3:-$WORK/ckpt/sft_all1/final}
for r in $RATES; do
    echo "[handover-random $GAME] rate $r"
    python -u -m playjev.play $GAME --policy local --ckpt $CKPT --episodes 16 --pages 8 --seed0 5000 --max-steps 1500 --handover-random $r
done
echo "[handover-random $GAME] done"

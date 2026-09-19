#!/bin/bash
# Control for the handover curve: the same share of steps handed to the teacher at random (no confidence).
#   bash hpc/handover_control.sh snake 0.004 0.38 0.45 0.53          (rates as separate arguments; qsub -v strips quotes)
set -u
GAME=$1; shift
WORK=${WORK:-$(cd "$(dirname "$0")/../.." && pwd)}   # the repo sits at $WORK/repo
CKPT_ROOT=${CKPT_ROOT:-$WORK/ckpt}
CKPT=${CKPT:-$CKPT_ROOT/sft_all1/final}
for r in "$@"; do
    echo "[handover-random $GAME] rate $r"
    python -u -m playjev.play $GAME --policy local --ckpt $CKPT --episodes 16 --pages 8 --seed0 5000 --max-steps 1500 --handover-random $r
done
echo "[handover-random $GAME] done"

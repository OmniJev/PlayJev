#!/bin/bash
# System One / System Two sweep for one game: the model plays, and below a Jev-confidence threshold the teacher
# decides. tau 0 = the model alone, 1.01 = the teacher alone; the curve in between is what the confidence buys.
#   bash hpc/handover_sweep.sh snake [ckpt] [taus]
# Results: runs/play/<game>_local_handover<tau>.json (score_mean, handover_rate, ...), one per tau.
set -u
GAME=$1
WORK=${WORK:-$(cd "$(dirname "$0")/../.." && pwd)}   # the repo sits at $WORK/repo
CKPT_ROOT=${CKPT_ROOT:-$WORK/ckpt}
CKPT=${2:-$CKPT_ROOT/sft_all1/final}
TAUS=${3:-"0 0.2 0.4 0.6 0.8 1.01"}
RUN=$(basename $(dirname $CKPT))   # ckpt/<run>/final -> <run>; results are also kept under runs/play/<run>_<game>_handover<tau>.json
for tau in $TAUS; do
    echo "[handover $GAME] tau $tau ($RUN)"
    python -u -m playjev.play $GAME --policy local --ckpt $CKPT --episodes 16 --pages 8 --seed0 5000 --max-steps 1500 --handover $tau
    cp runs/play/${GAME}_local_handover${tau}.json runs/play/${RUN}_${GAME}_handover${tau}.json 2>/dev/null
done
echo "[handover $GAME] done"

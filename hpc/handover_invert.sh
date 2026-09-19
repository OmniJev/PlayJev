#!/bin/bash
# Control for the handover result: hand the same share of steps to the teacher, chosen by HIGH confidence instead of
# low. If the trigger carries signal, this has to come out below the random control, not just below the real one.
#   bash hpc/handover_invert.sh <ckpt> <tag> <game> <tau> [<game> <tau> ...]
set -u
WORK=${WORK:-$(cd "$(dirname "$0")/../.." && pwd)}   # the repo sits at $WORK/repo
CKPT=$1; TAG=$2; shift 2
while [ $# -ge 2 ]; do
  GAME=$1; TAU=$2; shift 2
  python -u -m playjev.play $GAME --policy local --ckpt $CKPT --episodes 16 --pages 8 --seed0 5000 --max-steps 1500 \
      --handover $TAU --handover-invert \
      --out $WORK/runs/play/hinv_${TAG}_${GAME}_$TAU.json
  echo "[hinv $TAG $GAME tau=$TAU] done"
done
echo "[handover invert] done"

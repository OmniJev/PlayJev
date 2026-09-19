#!/bin/bash
# Record System Two episodes for the demo: the model plays, and below a per-game confidence threshold (chosen so that
# roughly a third of the steps are handed over) the teacher decides; steps decided by the teacher carry "h": 1.
#   bash hpc/handover_record.sh [ckpt] [policy-name]
set -u
WORK=${WORK:-$(cd "$(dirname "$0")/../.." && pwd)}   # the repo sits at $WORK/repo
CKPT_ROOT=${CKPT_ROOT:-$WORK/ckpt}
CKPT=${1:-$CKPT_ROOT/dagger1/final}
NAME=${2:-playjev-0.8b-dagger1-s2}
declare -A TAU=( [breakout]=0.4 [tetris]=0.6 [2048]=0.2 [pacman]=0.6 [flappy]=0.8 [snake]=0.4 [racer]=0.6 [sokoban]=0.8 [mario]=0.4 [invaders]=0.4 )
for g in breakout tetris pacman flappy 2048 snake mario racer sokoban invaders; do
    echo "[record-s2 $g] tau ${TAU[$g]}"
    python -u -m playjev.play $g --policy local --ckpt $CKPT --episodes 16 --pages 8 --seed0 5000 --max-steps 1500 \
        --handover ${TAU[$g]} --record runs/replays --policy-name $NAME
    cp runs/play/${g}_local_handover${TAU[$g]}.json runs/play/s2_${g}_handover${TAU[$g]}.json 2>/dev/null
done
echo "[record-s2] done"

#!/bin/bash
# Transfer test: fine-tune an initial checkpoint on N random frames of one game, then play it.
#   bash hpc/transfer_ft.sh <game> <N> <init: base|hold8|path> [epochs=3] [lr=2e-5]
# init "base" = Qwen/Qwen3.5-0.8B-Base (no game knowledge), "hold8" = the eight-game model that never saw this game,
# "shuf" = the same eight games with their targets dealt out at random (answer format and option text, no game skill).
# Output checkpoint ckpt/tr_<game>_<N>_<init>, eval on the full validation split every epoch, closed loop at the end.
set -u
GAME=$1; N=$2; INIT=$3; EPOCHS=${4:-3}; LR=${5:-2e-5}
WORK=${WORK:-$(cd "$(dirname "$0")/../.." && pwd)}   # the repo sits at $WORK/repo
case $INIT in
  base) MODEL=Qwen/Qwen3.5-0.8B-Base ;;
  hold8) MODEL=$WORK/ckpt/sft_hold8b/final ;;
  shuf) MODEL=$WORK/ckpt/sft_hold8_shuf/final ;;
  *) MODEL=$INIT ;;
esac
TAG=""; [ "$LR" != "2e-5" ] && TAG=_lr${LR}
RUN=tr_${GAME}_${N}_${INIT}${TAG}; OUT=$WORK/ckpt/$RUN; mkdir -p $OUT
STEPS=$(( (N * EPOCHS + 63) / 64 ))
python -u -m playjev.train_sft --games $GAME --shards sft_all1_a --subsample $N --model $MODEL --out $OUT \
    --epochs $EPOCHS --lr $LR --batch 64 --micro-batch 32 --workers 8 --eval-every $(( STEPS / EPOCHS )) --save-every 100000 \
    --eval-samples 2000 --keep 1
python -u -m playjev.play $GAME --policy local --ckpt $OUT/final --episodes 16 --pages 8 --seed0 5000 --max-steps 1500 \
    --record runs/replays --policy-name transfer-${GAME}-$(( N / 1000 ))kf-${INIT} --out $OUT/play_${GAME}_local_delay0.json
echo "[transfer $RUN] done"

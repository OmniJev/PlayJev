#!/bin/bash
# The left end of the transfer curve: play a held-out game with no fine-tuning at all, from each initial checkpoint.
#   bash hpc/transfer_zeroshot.sh racer pacman
# Results land in ckpt/tr_<game>_0_<init>/play_<game>_local_delay0.json, the shape scripts/collect_transfer.py reads.
set -u
WORK=${WORK:-$WORK}
for GAME in "$@"; do
  for INIT in base hold8 shuf; do
    case $INIT in
      base) MODEL=Qwen/Qwen3.5-0.8B-Base ;;
      hold8) MODEL=$WORK/ckpt/sft_hold8b/final ;;
      shuf) MODEL=$WORK/ckpt/sft_hold8_shuf/final ;;
    esac
    case $MODEL in /*) [ -d "$MODEL" ] || { echo "[tr_${GAME}_0_${INIT}] no checkpoint at $MODEL, skipped"; continue; } ;; esac
    OUT=$WORK/ckpt/tr_${GAME}_0_${INIT}; mkdir -p $OUT
    [ -f $OUT/play_${GAME}_local_delay0.json ] && { echo "[tr_${GAME}_0_${INIT}] already played"; continue; }
    python -u -m playjev.play $GAME --policy local --ckpt $MODEL --episodes 16 --pages 8 --seed0 5000 --max-steps 1500 \
        --out $OUT/play_${GAME}_local_delay0.json
    echo "[tr_${GAME}_0_${INIT}] played"
  done
done
echo "[transfer zeroshot] done"

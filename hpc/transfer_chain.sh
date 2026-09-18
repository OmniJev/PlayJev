#!/bin/bash
# Data-efficiency chain for one held-out game: fine-tune from the base model and from the eight-game model on
# 1k / 3k / 10k random frames each (hpc/transfer_ft.sh), base runs first, the hold8 runs once its checkpoint exists.
#   bash hpc/transfer_chain.sh racer [sizes="1000 3000 10000"]
set -u
GAME=$1; SIZES=${2:-"1000 3000 10000"}
WORK=$WORK
for n in $SIZES; do bash hpc/transfer_ft.sh $GAME $n base; done
for i in $(seq 1 480); do [ -f $WORK/ckpt/sft_hold8/final/config.json ] && break; sleep 60; done   # up to 8 h
[ -f $WORK/ckpt/sft_hold8/final/config.json ] || { echo "no hold8 checkpoint"; exit 1; }
for n in $SIZES; do bash hpc/transfer_ft.sh $GAME $n hold8; done
echo "[transfer chain $GAME] done"

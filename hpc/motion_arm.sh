#!/bin/bash
# One motion arm end to end on a hopper node: composite the shards, one SFT epoch, closed loop on held-out seeds.
#
#   cd $WORK && qsub -v 'NAME=mo_ghost,CMD=ARM=ghost bash hpc/motion_arm.sh' repo/hpc/hopper_task.pbs
#
# ARM: plain (single frame, the control), ghost (threshold-masked trail), rgbt (three frames as R, G, B).
# The three arms share everything else: same game, same source shards, same hyperparameters, same 16 held-out
# seeds, same cap. The composites are pre-rendered into new shards, so the trainer reads them with --shards and
# needs no flag of its own; scripts/play_motion.py composes the same way at play time.
set -e
GAME=${GAME:-breakout}
ARM=${ARM:-ghost}
SRC_SHARDS=${SRC_SHARDS:-sft_all1_a sft_all1_b sft_all1_c}
RUN=${RUN:-motion_${GAME}_${ARM}}
CAP=${CAP:-1500}
CKPT_ROOT=$WORK/ckpt
OUT=$CKPT_ROOT/$RUN
MODEL=${MODEL:-$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68}
mkdir -p $OUT
echo "[arm] GAME=$GAME ARM=$ARM RUN=$RUN SRC_SHARDS=$SRC_SHARDS CAP=$CAP"

# 1. Composite (the control trains on the raw frames, so it skips this).
if [ "$ARM" = plain ]; then
    TRAIN_SHARDS="$SRC_SHARDS"
else
    TRAIN_SHARDS=""
    for s in $SRC_SHARDS; do
        d=m_${ARM}_${s}
        if [ -f data/$GAME/$d/records.jsonl ]; then
            echo "[compose] $d exists, reusing"
        else
            t0=$(date +%s)
            python -u scripts/make_motion_shard.py $GAME --src $s --dst $d --mode $ARM --workers 12
            echo "[compose] $d $(( $(date +%s) - t0 )) s"
        fi
        TRAIN_SHARDS="$TRAIN_SHARDS $d"
    done
fi
echo "[train] shards:$TRAIN_SHARDS"

# 2. One epoch, the same hyperparameters as hpc/train_sft.pbs.
python -u -m playjev.train_sft --games $GAME --shards $TRAIN_SHARDS --model $MODEL --out $OUT \
    --epochs 1 --batch 64 --micro-batch 64 --workers 10 --eval-every 400 --save-every 400 \
    --eval-samples 2000 --keep 2 $EXTRA

# 3. Closed loop: 16 held-out episodes, argmax, the no-op rule, cap CAP.
if [ "$ARM" = plain ]; then
    python -u -m playjev.play $GAME --policy local --ckpt $OUT/final --episodes 16 --pages 8 --seed0 5000 \
        --max-steps $CAP --out runs/play/${RUN}.json --record runs/replays --policy-name playjev-0.8b-$RUN
else
    python -u scripts/play_motion.py $GAME --ckpt $OUT/final --motion $ARM --episodes 16 --pages 8 --seed0 5000 \
        --max-steps $CAP --out runs/play/${RUN}.json --record runs/replays --policy-name playjev-0.8b-$RUN
fi
cp runs/play/${RUN}.json $OUT/play_${GAME}.json
echo "[final eval]"; cat $OUT/final_eval.json
echo "[closed loop]"; cat runs/play/${RUN}.json
echo MOTION_ARM_DONE

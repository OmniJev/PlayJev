#!/bin/bash
# Per-game DAgger specialist on one vanda A40 (run through hpc/vanda_task.pbs, cwd = the repo):
#   bash hpc/vanda_specialist.sh breakout [frames=40000] [epochs=1] [lr=1e-5]
# 1. the ten-game checkpoint plays the game (argmax, no-op rule, eps 0.05); the teacher labels every visited frame,
#    three collectors in parallel on the node's cores (OMP threads capped, as the hopper job does);
# 2. one epoch from that checkpoint on its teacher-driven shard (sft_all1_a) plus the three on-policy shards;
# 3. closed loop on the held-out seeds, recorded (policy name playjev-0.8b-spec_<game>).
set -u
GAME=$1; FRAMES=${2:-40000}; EPOCHS=${3:-1}; LR=${4:-1e-5}
WORK=${WORK:-$(cd "$(dirname "$0")/../.." && pwd)}   # the repo sits at $WORK/repo
CKPT=$WORK/ckpt/sft_all1/final
RUN=spec_$GAME; OUT=$WORK/ckpt/$RUN; PER=$(( FRAMES / 3 ))
mkdir -p $OUT
t0=$(date +%s)
if [ ! -f data/$GAME/${RUN}_c/records.jsonl ]; then
    export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
    ACT="--actor local --ckpt $CKPT --epsilon 0.05"
    python -u -m playjev.collect $GAME --steps $PER --pages 8 $ACT --shard ${RUN}_a --seed0 700000 > $WORK/logs/collect_${RUN}_a.log 2>&1 &
    P1=$!
    python -u -m playjev.collect $GAME --steps $PER --pages 8 $ACT --shard ${RUN}_b --seed0 800000 > $WORK/logs/collect_${RUN}_b.log 2>&1 &
    P2=$!
    python -u -m playjev.collect $GAME --steps $PER --pages 8 $ACT --shard ${RUN}_c --seed0 900000 > $WORK/logs/collect_${RUN}_c.log 2>&1 &
    P3=$!
    wait $P1 $P2 $P3
    unset OMP_NUM_THREADS MKL_NUM_THREADS
    echo "[collect $GAME] $(( $(date +%s) - t0 )) s"; for s in a b c; do tail -1 $WORK/logs/collect_${RUN}_$s.log; done
fi
python -u -m playjev.data $GAME
python -u -m playjev.train_sft --games $GAME --shards sft_all1_a ${RUN}_a ${RUN}_b ${RUN}_c --model $CKPT --out $OUT \
    --epochs $EPOCHS --lr $LR --batch 64 --micro-batch 32 --workers 10 --eval-every 400 --save-every 400 --eval-samples 2000 --keep 1
echo "[train $GAME] $(( $(date +%s) - t0 )) s"
PLAY="python -u -m playjev.play"
$PLAY $GAME --policy local --ckpt $OUT/final --episodes 16 --pages 8 --seed0 5000 --max-steps 1500 --record runs/replays --policy-name playjev-0.8b-$RUN
cp runs/play/${GAME}_local.json $OUT/play_${GAME}_local_delay0.json
echo "[play $GAME] done $(( $(date +%s) - t0 )) s"

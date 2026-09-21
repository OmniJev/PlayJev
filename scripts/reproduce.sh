#!/usr/bin/env bash
# The whole model from nothing: the teachers collect, the base model clones them, two DAgger
# rounds, then version 3 starts over from the base model on everything those rounds produced with
# the replay mix and plays a third round, then the closed-loop score. No game dataset to download;
# the games are in this repository and the collector drives them in headless Chromium, so the
# frames are regenerated here. The replay pools are public multiple-choice sets that
# scripts/build_aux.py fetches from Hugging Face.
#
#   bash scripts/reproduce.sh                        # all ten games, the released recipe
#   GAMES=snake bash scripts/reproduce.sh            # one game
#   STAGES="score" bash scripts/reproduce.sh         # only the closed loop, on an existing checkpoint
#
# One CUDA GPU (17 GB at batch 64 with gradient checkpointing; 60 GB for the version 3 stages, which
# run micro-batch 16 without it), Python 3.12, `playwright install chromium`. Collection is about
# three minutes per game per round and writes ~5 GB of 448 px JPEGs under data/.
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES=${GAMES:-"tetris snake pacman racer invaders sokoban mario flappy breakout 2048"}
BASE=${BASE:-Qwen/Qwen3.5-0.8B-Base}
FRAMES=${FRAMES:-100000}          # per game for the cloning round
DFRAMES=${DFRAMES:-40000}         # per game for each DAgger round
STAGES=${STAGES-"collect clone dagger1 dagger2 aux version3 dagger3 score"}
PY=${PY:-python3}
mkdir -p logs ckpt

# Three shards per game, collected in parallel. Shards a and b use the game's own random-action
# rate (games/<id>/pj.json "collect": {"epsilon": x}); shard c raises it to 0.3 so the data
# covers recoveries, except in sokoban where a random push deadlocks the level for good. The
# seed ranges are far apart so no two shards ever share an episode, and every seed0 is a
# multiple of ten because the validation split is seed % 10 == 0.
collect () {                      # collect <run> [actor checkpoint] [seed0 seed0 seed0]
  local run=$1 ckpt=${2:-} act="" total=$FRAMES seeds=(100000 300000 500000)
  [ -n "$ckpt" ] && total=$DFRAMES
  local per=$(( total / 3 ))
  if [ -n "$ckpt" ]; then         # DAgger: the model plays, the teacher labels what it visited
    act="--actor local --ckpt $ckpt --epsilon 0.05"; seeds=(${3:-700000} ${4:-800000} ${5:-900000})
    export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3   # three model copies share the cores with Chromium
  fi
  for g in $GAMES; do
    local t0=$SECONDS i=0 pids=() logs=()
    for sh in a b c; do
      local eps=""
      [ -z "$ckpt" ] && [ "$sh" = c ] && [ "$g" != sokoban ] && eps="--epsilon 0.3"
      logs+=("logs/collect_${run}_${g}_$sh.log")
      $PY -u -m playjev.collect "$g" --steps "$per" --pages 8 $act $eps \
        --shard "${run}_$sh" --seed0 "${seeds[$i]}" > "logs/collect_${run}_${g}_$sh.log" 2>&1 &
      pids+=($!); i=$(( i + 1 ))
    done
    # a bare `wait` returns 0 whatever the children did, so wait on each one: a collector that
    # dies has to stop the run, not leave a short shard behind for the trainer to find
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || { echo "collector failed, ${logs[$i]}:" >&2; tail -5 "${logs[$i]}" >&2; exit 1; }
    done
    echo "[collect $run $g] $(( SECONDS - t0 )) s"
  done
  unset OMP_NUM_THREADS MKL_NUM_THREADS
  $PY -u -m playjev.data $GAMES
}

train () {                        # train <out> <model> <lr> <eval-every> <extra> <shard>...
  local out=$1 model=$2 lr=$3 every=$4 extra=$5; shift 5
  $PY -u -m playjev.train_sft --games $GAMES --shards "$@" --model "$model" --out "$out" \
    --epochs 1 --batch 64 --micro-batch 64 --lr "$lr" --workers 10 \
    --eval-every "$every" --save-every "$every" --eval-samples 4000 --keep 2 $extra
}

# Version 3: 60 percent of the batches are game frames, 20 percent general image questions and 20
# percent text questions (the pools from scripts/build_aux.py), and 30 percent of the game samples
# get one of the four option rewrites. Micro-batch 16 without gradient checkpointing is 1.3x faster
# than 64 with it and changes no arithmetic; batch, step count and mixture are the same.
MIX="--aux runs/aux/text.jsonl runs/aux/image.jsonl --mix 0.6 0.2 0.2 --aug 0.12 0.08 0.05 0.05"
train_mix () {                    # train_mix <out> <model> <lr> <extra> <shard>...
  local out=$1 model=$2 lr=$3 extra=$4; shift 4
  $PY -u -m playjev.train_sft --games $GAMES --shards "$@" --model "$model" --out "$out" \
    --batch 64 --micro-batch 16 --no-grad-ckpt --lr "$lr" --workers 10 \
    --eval-every 2000 --save-every 5000 --eval-samples 2000 --keep 2 $MIX $extra
}

has () { [[ " $STAGES " == *" $1 "* ]]; }

has collect  && collect clone
has clone    && train ckpt/clone   "$BASE"              2e-5 400  "" clone_a clone_b clone_c
# Each round starts from the previous checkpoint at half the learning rate and keeps one
# teacher-driven shard in the mix so the earlier states do not fall out of the model.
has dagger1  && { collect dagger1 ckpt/clone/final
                  train ckpt/dagger1 ckpt/clone/final   1e-5 1500 "" clone_a dagger1_a dagger1_b dagger1_c; }
# Round 2 also repeats the last ten records of every episode shorter than 500 steps four times:
# a short episode ended in a mistake, and those are the steps worth weighting.
has dagger2  && { collect dagger2 ckpt/dagger1/final
                  train ckpt/dagger2 ckpt/dagger1/final 1e-5 1500 "--boost-last 10 4" \
                        clone_a dagger1_a dagger2_a dagger2_b dagger2_c; }
# Version 3 goes back to the base model and takes every frame the first two rounds left behind in
# one run of 45000 steps (a little over one epoch of the 1.6M-frame pool), with the replay mix so
# the general ability the base model has survives the games. Then one more round on its own play,
# on fresh seeds, with the same mix carried through.
has aux      && $PY -u scripts/build_aux.py --out runs/aux
has version3 && train_mix ckpt/version3 "$BASE" 2e-5 "--max-steps 45000" \
                  clone_a clone_b clone_c dagger1_a dagger1_b dagger1_c dagger2_a dagger2_b dagger2_c
has dagger3  && { collect dagger3 ckpt/version3/final 1100000 1200000 1300000
                  train_mix ckpt/dagger3 ckpt/version3/final 1e-5 "--epochs 1 --boost-last 10 4" \
                        clone_a dagger1_a dagger2_a dagger3_a dagger3_b dagger3_c; }

# 16 held-out episodes per game on seeds the training never saw, and the two reference rows on
# the same seeds. `playjev.play` prints the mean score and writes the replays the demo reads.
if has score; then
  for g in $GAMES; do
    for p in "--policy local --ckpt ${CKPT:-ckpt/dagger3/final}" "--policy random" "--policy teacher"; do
      $PY -u -m playjev.play "$g" $p --episodes 16 --pages 8 --seed0 5000 --max-steps 1500
    done
  done
fi

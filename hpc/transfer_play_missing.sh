#!/bin/bash
# Play the transfer checkpoints whose closed loop never got written (the copy line postdates the first chain).
#   bash hpc/transfer_play_missing.sh tr_racer_1000_base tr_pacman_1000_base
set -u
WORK=$WORK
for run in "$@"; do
  OUT=$WORK/ckpt/$run; GAME=$(echo $run | cut -d_ -f2)
  if [ ! -d $OUT/final ]; then echo "[$run] no final checkpoint, skipped"; continue; fi
  if [ -f $OUT/play_${GAME}_local_delay0.json ]; then echo "[$run] already played"; continue; fi
  python -u -m playjev.play $GAME --policy local --ckpt $OUT/final --episodes 16 --pages 8 --seed0 5000 --max-steps 1500 \
      --out $OUT/play_${GAME}_local_delay0.json
  echo "[$run] played"
done

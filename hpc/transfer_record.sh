#!/bin/bash
# Record the transfer checkpoints so the demo page can replay them side by side on the same seeds.
#   bash hpc/transfer_record.sh racer 1000+10000 base+hold8
# Policy name: transfer-<game>-<N>f-<init>, replays under runs/replays/<game>/.
set -u
GAME=$1; SIZES=${2:-1000+10000}; INITS=${3:-base+hold8}; SIZES=${SIZES//+/ }; INITS=${INITS//+/ }
WORK=${WORK:-$WORK}
for init in $INITS; do for n in $SIZES; do
  OUT=$WORK/ckpt/tr_${GAME}_${n}_${init}
  if [ ! -d $OUT/final ]; then echo "[tr_${GAME}_${n}_${init}] no final checkpoint, skipped"; continue; fi
  NAME=transfer-${GAME}-$(( n / 1000 ))kf-${init}
  python -u -m playjev.play $GAME --policy local --ckpt $OUT/final --episodes 16 --pages 8 --seed0 5000 \
      --max-steps 1500 --record runs/replays --policy-name $NAME --out $OUT/play_${GAME}_local_delay0.json
  echo "[$NAME] recorded"
done; done
echo "[transfer record $GAME] done"

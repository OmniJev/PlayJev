#!/bin/bash
# Learning-rate sweep for the transfer baseline: the raw base init gets several learning rates, the eight-game
# init keeps the single default, so the comparison is generous to the arm the claim argues against.
#   bash hpc/transfer_lr_sweep.sh racer 10000 5e-5+1e-4
set -u
GAME=$1; N=${2:-10000}; LRS=${3:-5e-5+1e-4}; LRS=${LRS//+/ }
for lr in $LRS; do bash hpc/transfer_ft.sh $GAME $N base 3 $lr; done
echo "[lr sweep $GAME $N] done"

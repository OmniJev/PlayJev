#!/bin/bash
# qsub wrapper: adds this site's project code and log directory, which live outside the repo.
#   cd $WORK && bash repo/hpc/q.sh -v 'NAME=x,CMD=...' repo/hpc/hopper_task.pbs
# Reads $WORK/site.env (gitignored; copy hpc/site.env.example and fill it in).
set -euo pipefail
WORK=${PLAYJEV_WORK:-$PWD}
if [ -f "$WORK/site.env" ]; then . "$WORK/site.env"; fi
exec qsub ${PBS_PROJECT:+-P "$PBS_PROJECT"} ${PLAYJEV_LOGS:+-o "$PLAYJEV_LOGS"} "$@"

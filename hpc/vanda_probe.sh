#!/bin/bash
# Smoke test for a fresh vanda node: torch sees the GPU, Chromium runs, the checkpoint loads, one closed loop plays.
set -x
WORK=${WORK:-$(cd "$(dirname "$0")/../.." && pwd)}   # the repo sits at $WORK/repo
CKPT_ROOT=${CKPT_ROOT:-$WORK/ckpt}
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
PY
python -m playjev.bench snake --pages 4 --steps 50
python -m playjev.play snake --policy local --ckpt $CKPT_ROOT/sft_all1/final --episodes 4 --pages 4 --max-steps 300

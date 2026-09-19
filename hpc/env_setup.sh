#!/bin/bash
# Build the PlayJev Python environment on hopper. Run this ON THE LOGIN NODE: compute nodes have no
# network, so every wheel has to land here first.
#
#   PLAYJEV_WORK=<work tree> bash hpc/env_setup.sh        # -> $PLAYJEV_WORK/.venv
#
# The stack is the one that runs Qwen3.5 on jvp (driver 550, H200): torch 2.10.0+cu128,
# transformers 5.17.0, flash-linear-attention 0.5.2 for the linear-attention layers (without it
# Qwen3.5 is about 2x slower), triton 3.7.1 (see below). causal-conv1d is optional: it needs a prebuilt wheel or nvcc, and
# the login node has neither for torch 2.10, so the script tries once and moves on.
# Playwright is pinned to 1.63 because the Chromium build already on scratch is 1243
# ($PLAYWRIGHT_BROWSERS_PATH/chromium_headless_shell-1243), which is what 1.63 expects.
set -euo pipefail

WORK=${PLAYJEV_WORK:?set PLAYJEV_WORK to the work tree (repo/, .venv/, logs/)}
VENV=${PLAYJEV_VENV:-$WORK/.venv}
UV=${UV:-$HOME/.local/bin/uv}
PYVER=3.12

# The login node's /tmp is 5 GB and ~/.bashrc points UV_CACHE_DIR there; the cu128 torch stack alone
# downloads about 4 GB of wheels, so cache and temp files go to scratch for this build.
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WORK/.cache/uv}   # both need a few GB; point them at a disk that has it
export TMPDIR=${TMPDIR:-$WORK/tmp}
mkdir -p "$WORK/logs" "$UV_CACHE_DIR" "$TMPDIR"

log() { echo "[env_setup $(date +%H:%M:%S)] $*"; }

if [ ! -x "$VENV/bin/python" ]; then
    log "creating venv $VENV (python $PYVER)"
    "$UV" venv --python "$PYVER" "$VENV"
fi
PY="$VENV/bin/python"

log "torch 2.10.0+cu128 and torchvision 0.25.0 (the Qwen image and video processors need torchvision)"
"$UV" pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0

log "transformers stack"
"$UV" pip install --python "$PY" \
    transformers==5.17.0 accelerate==1.15.0 peft==0.21.0 \
    flash-linear-attention==0.5.2 \
    tokenizers==0.23.2 safetensors==0.8.0 numpy==2.5.3 \
    pillow playwright==1.63.0 einops

# torch 2.10 pins triton 3.6.0, but fla 0.5.2 refuses its gated chunk *backward* kernel on Hopper with triton 3.4 to
# 3.7.0 (wrong results, fla issue #640); training needs 3.7.1. The forward path is fine either way. Installed last so
# nothing downgrades it; playjev.train_sft checks the kernel against the torch reference before training.
log "triton 3.7.1 (fla backward on Hopper)"
"$UV" pip install --python "$PY" triton==3.7.1

log "causal-conv1d (optional)"
if "$UV" pip install --python "$PY" --no-build-isolation causal-conv1d >"$WORK/logs/causal_conv1d_install.log" 2>&1; then
    log "causal-conv1d installed"
else
    log "causal-conv1d unavailable (see logs/causal_conv1d_install.log); Qwen3.5 uses the torch conv fallback"
fi

log "versions"
"$PY" - <<'EOF'
import importlib, importlib.metadata as m
for name in ["torch", "torchvision", "transformers", "accelerate", "peft", "flash-linear-attention", "triton", "tokenizers",
             "safetensors", "numpy", "pillow", "playwright", "causal-conv1d"]:
    try:
        print(f"  {name:24s} {m.version(name)}")
    except m.PackageNotFoundError:
        print(f"  {name:24s} (not installed)")
import torch
print("  torch.version.cuda      ", torch.version.cuda)
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor  # noqa: F401  (import check only)
print("  Qwen3_5ForConditionalGeneration import ok")
EOF
log "done: $VENV"

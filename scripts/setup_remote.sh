#!/usr/bin/env bash
# Provision a freshly rented GPU box for training.
#
#   curl -fsSL https://raw.githubusercontent.com/GokulEdakkePuram/ObjectDetection-DroneImagery/main/scripts/setup_remote.sh | bash
#
# Assumes an Ubuntu image with an NVIDIA driver already present (any of the
# PyTorch or CUDA templates on vast.ai / RunPod). Everything Python-side comes
# from uv.lock, so the environment matches the one the results were produced
# on.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/GokulEdakkePuram/ObjectDetection-DroneImagery.git}"
WORKDIR="${WORKDIR:-$HOME/aerialdet}"

echo "==> Checking for a GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "    nvidia-smi not found. This box has no usable GPU driver." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "==> Checking disk space"
# The dataset is ~3.7 GB unpacked, the CUDA-enabled venv is ~8 GB, and runs
# and checkpoints grow from there. Rentals default to 10 GB, which is not
# enough -- this is the most common way an instance gets wasted.
AVAIL_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "    ${AVAIL_GB} GB available"
if [ "${AVAIL_GB}" -lt 30 ]; then
    echo "    WARNING: under 30 GB free. Destroy this instance and re-rent with more disk." >&2
fi

echo "==> Installing uv"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> Cloning into ${WORKDIR}"
if [ -d "${WORKDIR}/.git" ]; then
    git -C "${WORKDIR}" pull --ff-only
else
    git clone "${REPO_URL}" "${WORKDIR}"
fi
cd "${WORKDIR}"

echo "==> Installing dependencies (CUDA wheels come from uv.lock)"
uv sync --extra wandb

echo "==> Verifying CUDA is visible to torch"
uv run python -c "
import torch
assert torch.cuda.is_available(), 'torch cannot see the GPU'
p = torch.cuda.get_device_properties(0)
print(f'    {p.name}, {p.total_memory / 1024**3:.0f} GB, torch {torch.__version__}')
"

echo "==> Downloading VisDrone (~2 GB)"
uv run aerialdet download

cat <<'NEXT'

Ready. Suggested next steps:

  uv run aerialdet probe baseline_640 finetune_960 finetune_1280
      Calibrate before committing rental hours. Confirms the profile's batch
      size fits and projects how long the real runs take.

  wandb login
      Only needed once per instance, if tracking to W&B.

  uv run aerialdet train finetune_960 --track wandb
      The profile is auto-detected from the GPU.

NEXT

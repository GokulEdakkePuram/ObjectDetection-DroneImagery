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

echo "==> Checking the driver is new enough"
# The pinned torch wheels bundle a CUDA 13 runtime (nvidia-cudnn-cu13, nccl-cu13),
# which needs a host driver that supports CUDA 13.0 or later. A 12.x driver fails
# with "NVIDIA driver on your system is too old" -- but only after uv sync has
# spent five paid minutes downloading 5 GB, so check it up front.
REQUIRED_CUDA_MAJOR=13
HOST_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
if [ -z "${HOST_CUDA}" ]; then
    echo "    WARNING: could not read the driver's CUDA version; continuing anyway." >&2
else
    echo "    driver supports CUDA ${HOST_CUDA}"
    if [ "${HOST_CUDA%%.*}" -lt "${REQUIRED_CUDA_MAJOR}" ]; then
        cat >&2 <<EOF
    This host supports CUDA ${HOST_CUDA}, but the locked torch build needs
    CUDA ${REQUIRED_CUDA_MAJOR}.0 or newer.

    Destroy this instance and rent one filtered to CUDA ${REQUIRED_CUDA_MAJOR}.0+.
    That is cheaper than the time it takes to work around it, and keeps the
    environment identical to the one in uv.lock.
EOF
        exit 1
    fi
fi

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

if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "==> WANDB_API_KEY is set; W&B needs no interactive login"
else
    echo "==> WANDB_API_KEY is not set"
    echo "    Either export it before training, or run: uv run wandb login"
fi

cat <<'NEXT'

Ready. Suggested next steps:

  uv run aerialdet probe baseline_640 finetune_960 finetune_1280 --fractions 0.1 0.3
      Calibrate before committing rental hours. Confirms the profile's batch
      size fits and projects how long the real runs take. Two fractions, so
      fixed per-epoch overhead is separated from the per-image rate.

  export WANDB_API_KEY=<key from wandb.ai/authorize>
      Preferred over an interactive login on an instance you will destroy.
      Every CLI here lives in the project venv, so the interactive form is
      `uv run wandb login` -- a bare `wandb` is not on PATH.

  tmux new -s train
  uv run aerialdet train finetune_960 --track wandb
      Inside tmux, so a dropped SSH connection does not kill the run.
      The hardware profile is auto-detected from the GPU.

NEXT

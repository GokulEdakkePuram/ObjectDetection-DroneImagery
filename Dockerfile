# Reproducible training image.
#
# Deliberately built on plain Ubuntu rather than an nvidia/cuda base: the
# torch wheels pinned in uv.lock ship their own CUDA runtime (nvidia-cudnn,
# nccl, cublas ...), so a CUDA base image would mean two copies of the
# toolkit and a chance of them disagreeing. The only host requirement is an
# NVIDIA driver new enough for the CUDA version those wheels were built
# against -- check with `nvidia-smi` before pulling.
#
#   docker build -t aerialdet .
#   docker run --gpus all -v $PWD/runs:/app/runs aerialdet \
#       uv run aerialdet train finetune_960 --track wandb
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:/app/.venv/bin:$PATH"

# libgl and libglib are OpenCV's runtime dependencies; without them the
# import succeeds at build time and fails on the first image read.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Dependencies resolve from the lockfile alone, so this layer is cached until
# the lock actually changes -- source edits do not trigger a 5 GB reinstall.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --extra wandb

COPY . .
RUN uv sync --frozen --extra wandb

# Datasets and runs belong on a mounted volume, not baked into the image.
VOLUME ["/app/data", "/app/runs"]

CMD ["uv", "run", "aerialdet", "--help"]

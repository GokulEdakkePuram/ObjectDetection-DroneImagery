# Running on rented GPUs

The laptop measurements in [doc 02](02-resolution-and-stride.md) made the full
sweep impractical — 13.9 h at 640, 34.5 h at 960, and worse at 1280, so about
five days of continuous compute. Renting a GPU turns that into a few hours and
a few dollars, and it removes the ablation's biggest weakness as a side effect.

## The confound it fixes

On a 16 GB laptop, batch size had to shrink as resolution grew (8 → 4 → 2).
That is a real confound: smaller batches mean noisier gradients, so the
comparison was never purely about resolution.

On one GPU with enough memory, **batch stays constant across the whole sweep**
and the comparison isolates a single variable. That is the difference between
an ablation and three runs that happen to be next to each other.

This is why hardware lives in [`configs/profiles/`](../configs/profiles)
rather than in the experiment configs:

| | describes | may set |
| --- | --- | --- |
| experiment config | what to train | `model`, `imgsz`, `epochs`, augmentation |
| hardware profile | what the machine can hold | `batch`, `workers`, `device`, `amp`, `cache` |

Precedence is base → experiment → profile, and `load_profile` **rejects** a
profile that tries to set anything else. If a profile could change `epochs` or
`imgsz`, two runs with the same label could quietly mean different things.

```bash
uv run aerialdet train finetune_960                    # profile auto-detected
uv run aerialdet train finetune_960 --profile cuda24   # or pinned explicitly
```

The chosen profile is recorded in `runs/train/<name>/aerialdet_config.json`,
so a result always says which machine produced it.

## Measure before paying

The batch sizes in the profiles are estimates. Confirm them on the actual card
before committing rental hours:

```bash
uv run aerialdet probe baseline_640 finetune_960 finetune_1280
```

This runs a few epochs on 5% of the data and projects the full schedule,
reporting peak memory on CUDA. An OOM or an unwelcome time estimate then costs
four minutes instead of three hours. The projection uses the *fastest* epoch
observed, because the first one always carries label caching and warmup.

## Choosing a box

**vast.ai** is the cheapest option and fine for this workload. **RunPod**
costs slightly more and is more reliable, with persistent volumes so the
dataset survives between sessions.

A 24 GB card (RTX 3090/4090, A5000) is the right tier — `yolo11s` at 1280 does
not need more, and 48 GB mainly buys a larger batch.

**Filter for CUDA 13.0 or newer.** The pinned torch wheels bundle a CUDA 13
runtime, so a host with a 12.x driver fails with *"NVIDIA driver on your
system is too old"* — after `uv sync` has already spent five paid minutes
pulling 5 GB. Plenty of rental hosts still run 12.x drivers, so this is worth
setting as a search filter rather than discovering on the box.
`setup_remote.sh` now checks it up front and exits in seconds.

Three things that waste rentals:

- **Set disk to 40 GB.** The 10 GB default is not enough: ~3.7 GB dataset,
  ~8 GB for the CUDA venv, plus checkpoints. `setup_remote.sh` warns, but by
  then you have already paid for the instance.
- **Prefer on-demand over interruptible** until resume-from-checkpoint is
  wired up. Interruptible is roughly half price, but losing a two-hour run to
  reclaim $0.30 is a bad trade.
- **Check the driver before anything else.** `nvidia-smi` reports the CUDA
  version the host supports; if it is below 13.0, destroy and re-rent.

## Provisioning

On any Ubuntu image with an NVIDIA driver:

```bash
curl -fsSL https://raw.githubusercontent.com/GokulEdakkePuram/ObjectDetection-DroneImagery/main/scripts/setup_remote.sh | bash
```

It verifies the driver, warns about disk, installs uv, syncs from `uv.lock`
and downloads the dataset. Because everything Python-side comes from the
lockfile — which already resolves CUDA wheels for Linux — the environment
matches the one the results were produced on.

For byte-identical reproducibility there is a [`Dockerfile`](../Dockerfile),
usable directly as a custom image on vast.ai:

```bash
docker build -t aerialdet .
docker run --gpus all -v $PWD/runs:/app/runs aerialdet \
    uv run aerialdet train finetune_960 --track wandb
```

Honestly, the script is the better day-to-day path: no image build, no
registry push, one command on a fresh box. The image earns its place when you
need the environment pinned exactly, or want to hand someone a single
artifact.

## Tracking

```bash
uv sync --extra wandb
wandb login
uv run aerialdet train finetune_960 --track wandb
```

W&B is the default because its run pages are shareable — a reader can be
linked straight at the training curves rather than asked to take a number on
trust. `--track mlflow` is there when the data should stay local; with no
`MLFLOW_TRACKING_URI` set it writes a file store under `runs/mlflow`.

Either way the tracker raises immediately if it is not installed, rather than
letting a long run finish having logged nothing.

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

This trains briefly on a fraction of the data and projects the full schedule,
reporting peak memory on CUDA. An OOM or an unwelcome time estimate then costs
a few minutes instead of three hours.

On the rented 4090, with the corrected two-fraction method:

| config | fitted rate | per epoch | 50 epochs | peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| `baseline_640` | 13.7 ms/img | 1.5 min | 1.2 h | 3.3 GB |
| `finetune_960` | 14.5 ms/img | 1.6 min | 1.3 h | 8.1 GB |
| `finetune_1280` | 19.5 ms/img | 2.1 min | 1.8 h | 13.6 GB |

Fitted overhead came out at 0.6-1.2 s/epoch — small, and the reason the naive
method went so wrong at 5% of the data.

`batch: 8` holds at 1280 with 10 GB to spare, so the constant-batch ablation
is safe on this card.

### What the scaling says about the bottleneck

Normalised against 640:

| imgsz | pixels | VRAM | time |
| ---: | ---: | ---: | ---: |
| 640 | 1.00x | 1.00x | 1.00x |
| 960 | 2.25x | 2.45x | 1.06x |
| 1280 | 4.00x | 4.12x | 1.42x |

Memory follows pixel count almost exactly, which confirms the model really is
processing 4x the data at 1280. Time does not: quadrupling the pixels costs
42% more wall clock, not 300%.

A GPU-bound pipeline could not do that. The 4090 is finishing each batch and
waiting, so the constraint is upstream — decoding and augmenting images on
CPU, with mosaic (which composites four source images per sample) the obvious
suspect.

The capacity ablation then confirmed it independently. At fixed 960 px:

| model | params | GFLOPs | train ms/img | peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| yolo11n | 2.6 M | 6.5 | 15.0 | 6.1 GB |
| yolo11s | 9.4 M | 21.4 | 14.5 | 8.1 GB |
| yolo11m | 20.0 M | 67.8 | 19.1 | 12.2 GB |

`yolo11n` is *slower* than `yolo11s` despite 3.6x fewer parameters — the two
are indistinguishable because neither is the bottleneck. Across the three
models a **10.4x span in FLOPs produces a 1.32x span in wall clock**. That is
about as clean a signature of an input-bound pipeline as you could ask for,
and it arrived from a different direction than the resolution result.

Memory behaves differently and usefully: it tracks compute faithfully in both
ablations. So the models genuinely are doing the extra work — they are simply
not what you are waiting for.

### Two consequences

**For reading the results.** The usual argument against high resolution or a
bigger model is that it costs too much. Here resolution costs 42% and capacity
costs 32%, so a result favouring either is a weaker claim than the same result
on hardware where the GPU is saturated.

**For deployment, the opposite.** Training throughput is dominated by
augmentation that inference never performs. In the validation pass, with no
mosaic in the path, `yolo11s` and `yolo11m` run at 1.6 ms and 2.2 ms per
image. On a power- and thermally-constrained drone the GPU *is* the
constraint, so FLOPs matter there in a way they do not here. Never quote a
training-time cost as if it were a deployment cost.

### Fixing it, if you want to

The bottleneck is CPU-side, so more dataloader workers should help. `workers`
lives in the hardware profile precisely because it is a machine property:
raise it in `configs/profiles/cuda24.yaml` and re-probe. If the 640 rate
improves, the diagnosis is confirmed and the sweep gets cheaper. Watching
`nvidia-smi` during a run is the other cheap check — utilisation well under
100% says the same thing.

## Calibrating honestly

The first version of `probe` timed one short run and divided by the fraction
of data it used. That produced this, on the 4090:

```
baseline_640   projected: 2.3 min/epoch
finetune_960   projected: 1.8 min/epoch     <- faster at 2.25x the pixels?
finetune_1280  projected: 2.6 min/epoch
```

`finetune_960` cannot be faster than `baseline_640` at the same batch size and
more than twice the pixels. The numbers were wrong, and wrong in the most
dangerous way: plausible enough to write down.

The bug is an assumption in the arithmetic — that epoch time is proportional
to dataset size. It is not:

```
epoch_seconds = overhead + rate * n_images
```

`overhead` is dataloader spin-up, cuDNN autotuning and epoch teardown, and it
does not shrink with the data. Dividing by `fraction` scales it up too, so at
`fraction=0.05` the estimate carries **twenty times** the real overhead:

```
projection - truth = (1/fraction - 1) * overhead = 19 * overhead
```

At 5% of VisDrone with batch 8 an epoch is only ~40 steps, which on a 4090 is
a couple of seconds of actual compute against several seconds of fixed cost.
Overhead dominated the measurement, so the differences between configs were
noise, and the ranking was meaningless.

The fix is to measure at **two** fractions and solve for both terms, then
extrapolate with the rate alone. The extra run costs a couple of minutes and
is the difference between an estimate and a guess. `_fit` also refuses to
extrapolate from a non-positive fitted rate — two runs lost in the noise
should raise, not return a confident wrong number.

The lesson generalises past this repo: an extrapolation is only as good as the
model behind it, and *dividing by a fraction* is a model. It is worth stating
the model out loud, because this one is wrong in a way that survives a glance
at the output.

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

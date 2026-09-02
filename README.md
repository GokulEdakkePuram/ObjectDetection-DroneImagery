# aerialdet — fine-tuning YOLO for small-object detection in drone imagery

Fine-tuning YOLO11 on [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset),
where the objects of interest are ~20 px across in a 1400 px frame.

The interesting part of this problem is not calling `.train()`. It is that a
stock-configured detector fails on aerial imagery for a specific, diagnosable
reason — the detection head's stride-8 feature map cannot resolve objects that
small once the frame has been downscaled to 640 px — and that the fix follows
from the diagnosis. This repo is a controlled attempt to demonstrate that,
with the reasoning written down in [`docs/`](docs/).

## Results

> **Status:** pipeline complete and tested; training runs in progress. Numbers
> below are filled in from `reports/results.md` as runs finish, and the
> commands that produce them are the ones in this README.

All five runs: 50 epochs, VisDrone `val`, constant `batch: 8` on one RTX 4090.

**Resolution** (`yolo11s`, only `imgsz` varies):

| run | imgsz | mAP50-95 | mAP50 | vs control | train time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_640` (control) | 640 | 0.2218 | 0.3797 | — | 1.2 h |
| `finetune_960` | 960 | 0.2873 | 0.4739 | **+29.5%** | 1.3 h |
| `finetune_1280` | 1280 | **0.3260** | 0.5284 | **+47.0%** | 1.8 h |

**Capacity** (960 px, only the model varies):

| run | model | params | mAP50-95 | mAP50 | train time |
| --- | --- | ---: | ---: | ---: | ---: |
| `size_n_960` | yolo11n | 2.6 M | 0.2324 | 0.3928 | 1.4 h |
| `finetune_960` | yolo11s | 9.4 M | 0.2873 | 0.4739 | 1.3 h |
| `size_m_960` | yolo11m | 20.0 M | **0.3203** | 0.5201 | 1.7 h |

Each checkpoint is scored at the resolution it was trained for.

### The mechanism holds up

Monotonic curves are not enough — a bigger input could help for any number of
reasons. [docs/01](docs/01-why-small-objects-are-hard.md) made a *falsifiable*
prediction: if resolution works by lifting objects above the detector's
stride-8 floor, the gain must concentrate in the smallest classes. If `bus`
improved as much as `pedestrian`, the explanation was wrong.

Per-class AP50, 640 → 1280, ordered by median object size:

| class | median side | AP50 @640 | AP50 @1280 | absolute gain |
| --- | ---: | ---: | ---: | ---: |
| people | 16.7 px | 0.3037 | 0.4837 | +0.180 |
| pedestrian | 17.2 px | 0.4134 | 0.6291 | +0.216 |
| motor | 20.0 px | 0.4427 | 0.6314 | +0.189 |
| bicycle | 20.8 px | 0.1241 | 0.3315 | +0.207 |
| tricycle | 28.4 px | 0.2683 | 0.4200 | +0.152 |
| awning-tricycle | 29.3 px | 0.1267 | 0.2305 | +0.104 |
| car | 32.5 px | 0.7761 | 0.8705 | +0.094 |
| van | 33.6 px | 0.4299 | 0.5504 | +0.121 |
| bus | 35.2 px | 0.5270 | 0.6707 | +0.144 |
| truck | 38.6 px | 0.3849 | 0.4661 | +0.081 |

**Absolute AP50 gain correlates with object size at r = −0.88.** The four
smallest classes gain **1.71×** what the six largest do. The prediction holds.

One trap worth naming. Ranked by *percentage* gain, `bicycle` looks
spectacular at +167% — but relative gain correlates with the *baseline* score
at r = −0.77, and only −0.50 with size. Classes that started badly show large
percentage gains regardless of why. Absolute gain removes that confound, and
it is the measure that gives r = −0.88. The headline survives the stricter
test; it would have been overstated by the looser one.

**The two axes buy almost the same thing.** `yolo11m` at 960 (0.3272) and
`yolo11s` at 1280 (0.3246) are within 0.8% of each other, for 1.7 h and 1.8 h
respectively. Spending on resolution and spending on capacity are close to
interchangeable here — which is not what the small-object argument alone would
predict, and is the most interesting thing in the table.

> These are `val` numbers, and `best.pt` is selected on `val`, so they are
> mildly optimistic. `test-dev` is held out for a single final measurement —
> see [docs/04](docs/04-reading-the-metrics.md).

**Tiled inference makes this model worse**: 0.2799 against 0.3112 whole-frame
at 1280, for 1.6x the latency. On a weakly-trained checkpoint the same code
gave +45%. Together those say tiling *substitutes* for resolution rather than
adding to it — a 1280 model has already bought the pixels tiling would supply,
so only tiling's seam artifacts remain. See
[docs/03](docs/03-tiled-inference.md).

Hardware: Apple M2 Pro (16 GB, MPS) for development; a rented RTX 4090
(24 GB, CUDA 13) for training. The profile that produced each result is
recorded alongside it.

### Compute budget, measured on the 4090

The `cuda24` profile's constant `batch: 8` was an estimate. `aerialdet probe`
confirms it holds across the whole sweep — which is what makes the resolution
ablation single-variable:

| config | imgsz | batch | per image | per epoch | 50 epochs | peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_640` | 640 | 8 | 13.7 ms | 1.5 min | 1.2 h | 3.3 GB |
| `finetune_960` | 960 | 8 | 14.5 ms | 1.6 min | 1.3 h | 8.1 GB |
| `finetune_1280` | 1280 | 8 | 19.5 ms | 2.1 min | 1.8 h | 13.6 GB |
| **full sweep** | | | | | **4.3 h** | |

Those VRAM figures are `max_memory_allocated`, and they **understate what the
card must supply**: PyTorch's caching allocator reserves more than it allocates,
and reserved memory is what triggers an OOM. The real `finetune_1280` run
reserved **17.5 GB** against the 13.6 GB measured here — a 29% gap. `probe` now
reports reserved for this reason.

Against ~5 days for the same work on the laptop, at a rental cost of a few
dollars.

A second axis, capacity at fixed 960 px:

| config | model | params | per image | 50 epochs | peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| `size_n_960` | yolo11n | 2.6 M | 15.0 ms | 1.4 h | 6.1 GB |
| `finetune_960` | yolo11s | 9.4 M | 14.5 ms | 1.3 h | 8.1 GB |
| `size_m_960` | yolo11m | 20.0 M | 19.1 ms | 1.7 h | 12.2 GB |

### Training here is input-bound, not compute-bound

Both ablations say the same thing. Normalised against `yolo11s` at 960:

| axis | | compute | train time |
| --- | --- | ---: | ---: |
| resolution | 640 → 1280 | 4.00× pixels | 1.42× |
| capacity | yolo11n → yolo11m | 10.4× FLOPs | 1.32× |

`yolo11n` is *slower* than `yolo11s` (15.0 vs 14.5 ms) despite 3.6× fewer
parameters. A GPU-bound pipeline cannot do that. A **10× span in compute
producing a 1.3× span in wall clock** means the 4090 spends most of its time
waiting on CPU-side decode and augmentation — mosaic composites four source
images per sample, which is expensive at 960 px.

Memory, by contrast, tracks compute faithfully (VRAM scales 0.75 / 1.00 / 1.51
across the models, and 1.00 / 2.45 / 4.12 across resolutions), which confirms
the models really are doing the extra work — they just are not the bottleneck.

**This changes how to read every result below.** "Higher resolution wins" and
"more capacity wins" are weak claims on hardware where both are nearly free.
It also does *not* transfer to deployment: training throughput is dominated by
augmentation that inference never runs. The validation pass, with no mosaic in
the path, separates `yolo11s` and `yolo11m` at 1.6 ms and 2.2 ms per image —
so on a power-constrained drone, FLOPs would matter a great deal more than
they do here.

## The argument in one table

Why resolution is the first thing to change, before any hyperparameter:

| input size | scale on a 1400 px frame | a 20 px car becomes | cells on the stride-8 map |
| ---: | ---: | ---: | ---: |
| 640 | 0.46 | 9 px | 1.1 |
| 960 | 0.69 | 14 px | 1.7 |
| 1280 | 0.91 | 18 px | 2.3 |

At 640 px most VisDrone objects occupy roughly one cell of the finest feature
map the detector has. Full reasoning in
[docs/01](docs/01-why-small-objects-are-hard.md).

## Quickstart

```bash
make setup      # uv sync
make data       # ~2 GB VisDrone download + conversion to YOLO format
make profile    # measure class balance and object scale -> reports/
make smoke      # 1-epoch run to verify the pipeline end to end
make baseline   # the control run
make finetune   # the 960 px run
make eval       # comparison table -> reports/results.md
```

On a rented GPU, provision with one command and calibrate before spending
hours — see [docs/06](docs/06-running-on-rented-gpus.md):

```bash
curl -fsSL https://raw.githubusercontent.com/GokulEdakkePuram/ObjectDetection-DroneImagery/main/scripts/setup_remote.sh | bash
uv run aerialdet probe baseline_640 finetune_960 finetune_1280
uv run aerialdet train finetune_960 --track wandb
```

Everything is also reachable directly:

```bash
uv run aerialdet train finetune_960 --profile cuda24 --track wandb
uv run aerialdet probe finetune_960          # cost a run before starting it
uv run aerialdet eval runs/train/finetune_960/weights/best.pt --imgsz 960
uv run aerialdet tiled-predict <weights> <image.jpg> --tile 640 --overlap 0.2
uv run aerialdet tiled-eval <weights>          # does tiling actually help?
uv run aerialdet export <weights> --format onnx
```

## How it is put together

```
configs/          one YAML per experiment, composed via `extends:`
  profiles/       what each machine can hold — batch, workers, device, amp
src/aerialdet/
  config.py       config loading, inheritance, validation
  hardware.py     accelerator detection and profile selection
  stats.py        dataset profiling — class balance, box-area distribution
  train.py        fine-tuning; writes the resolved config next to the weights
  probe.py        short calibration run that projects the full schedule
  evaluate.py     validation and the cross-run comparison table
  tiling.py       overlapping sliced inference + class-aware NMS merge
  tiled_eval.py   scores tiled vs whole-frame through the same metric code
  tracking.py     W&B / MLflow wiring
  export.py       ONNX / CoreML / TorchScript export
scripts/          provisioning for a freshly rented GPU box
docs/             the reasoning, written as it was worked out
tests/            geometry, config and profile tests (`make test`)
Dockerfile        pinned training image
```

Four deliberate choices:

**Hardware is not part of an experiment.** An experiment config says what to
train; a profile in `configs/profiles/` says what the machine can hold, and
`load_profile` rejects any profile that tries to set `epochs` or `imgsz`. The
payoff is a constant batch size across the resolution sweep — on the laptop
batch had to shrink as `imgsz` grew, so the comparison was never purely about
resolution. On one GPU it now is.

**Configs compose.** `finetune_960` inherits from `base.yaml` and overrides two
lines. An ablation where the configs differ in one place is an experiment; one
where they differ in twelve places is an anecdote. A test asserts the ablation
configs stay comparable, so the claim in the README cannot rot silently.

**Runs are traceable.** Each run writes its fully-resolved config to
`runs/train/<name>/aerialdet_config.json`, and Ultralytics' global data/output
directories are pinned into the repo rather than scattered across `$HOME`.

**Tiling is implemented, not imported.** [`tiling.py`](src/aerialdet/tiling.py)
is ~150 lines rather than a SAHI dependency, because the two things that make
sliced inference work — flush edge tiles and class-aware NMS across seams — are
the two things worth being able to explain.

## Notes

Written as the work was done, not reconstructed afterwards:

- [00 — The dataset](docs/00-the-dataset.md) — VisDrone, its label format, and
  the ignored-region trap in the conversion
- [01 — Why small objects are hard](docs/01-why-small-objects-are-hard.md) —
  stride floors, and why `IoU = (s-d)/(s+d)` makes small boxes brutal
- [02 — Resolution and stride](docs/02-resolution-and-stride.md) — the main
  ablation, including the batch-size confound
- [03 — Tiled inference](docs/03-tiled-inference.md) — the geometry, the
  overlap inequality, and what it costs
- [04 — Reading the metrics](docs/04-reading-the-metrics.md) — mAP50 vs
  mAP50-95, `max_det` truncation, and how to not oversell a result
- [05 — Experiment log](docs/05-experiment-log.md) — running journal, with
  expectations recorded before each run
- [06 — Running on rented GPUs](docs/06-running-on-rented-gpus.md) — hardware
  profiles, calibrating a run before paying for it, and what wastes a rental

## Licence

Code: MIT. Ultralytics YOLO is AGPL-3.0 — relevant if you build on this
commercially. VisDrone is released for academic use; check upstream terms.

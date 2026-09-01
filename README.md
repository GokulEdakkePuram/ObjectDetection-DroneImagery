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

| run | imgsz | mAP50-95 | mAP50 | train time |
| --- | ---: | ---: | ---: | ---: |
| `baseline_640` (control) | 640 | _pending_ | | |
| `finetune_960` | 960 | _pending_ | | |
| `finetune_1280` | 1280 | _pending_ | | |
| `finetune_960` + tiled inference | 640/tile | _pending_ | | |

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

Against ~5 days for the same work on the laptop, at a rental cost of a few
dollars.

**Resolution is nearly free here, and that is a finding rather than a
footnote.** Normalised against 640:

| imgsz | pixels | VRAM | time |
| ---: | ---: | ---: | ---: |
| 640 | 1.00× | 1.00× | 1.00× |
| 960 | 2.25× | 2.45× | 1.06× |
| 1280 | 4.00× | 4.12× | **1.42×** |

Memory tracks pixel count almost exactly, so the model really is doing 4× the
work at 1280. Time does not follow, which means the GPU is not the constraint
— the input pipeline is, and mosaic augmentation on CPU is the obvious
suspect. The practical consequence: if higher resolution wins on mAP, the
usual reason to compromise (it costs too much) does not apply on this
hardware.

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

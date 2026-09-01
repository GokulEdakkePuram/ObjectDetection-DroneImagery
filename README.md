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

Hardware: Apple M2 Pro (16 GB, MPS) for development; rented 24 GB CUDA for
training runs. The profile that produced each result is recorded alongside it.

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

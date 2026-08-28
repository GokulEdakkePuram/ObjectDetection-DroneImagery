# Experiment 1: input resolution

## Hypothesis

On VisDrone, input resolution dominates every other training hyperparameter,
because most objects sit near the resolution floor imposed by the detector's
stride-8 feature map ([doc 01](01-why-small-objects-are-hard.md)).

If that is true, raising `imgsz` from 640 to 960 should produce a larger mAP
gain than any augmentation or learning-rate change would — and the gain should
be concentrated in the smallest classes (`pedestrian`, `people`, `motor`)
rather than spread evenly.

## Design

Three configs that differ in exactly one meaningful line:

| config | imgsz | batch | everything else |
| --- | ---: | ---: | --- |
| [`baseline_640`](../configs/baseline_640.yaml) | 640 | 8 | inherited from [`base.yaml`](../configs/base.yaml) |
| [`finetune_960`](../configs/finetune_960.yaml) | 960 | 4 | identical |
| [`finetune_1280`](../configs/finetune_1280.yaml) | 1280 | 2 | identical |

`batch` has to shrink as resolution grows to fit in memory. That is a genuine
confound — smaller batches mean noisier gradients — and it is worth being
honest about rather than hiding. It is also mostly unavoidable on a single
machine, which is itself a useful thing to be able to explain in an interview.

A test asserts the configs stay comparable, so the ablation claim cannot rot
silently as the configs are edited:

```python
# tests/test_config.py
assert baseline.train_args == finetune.train_args
assert (baseline.imgsz, finetune.imgsz) == (640, 960)
```

## Running it

```bash
make baseline    # ~? h on an M2 Pro
make finetune
make eval        # writes reports/results.md
```

## Results

> Fill in from `reports/results.md` once both runs finish. Record wall-clock
> time alongside mAP — a 2% mAP gain that costs 3x the training time is a
> different result than one that is free.

| run | imgsz | mAP50-95 | mAP50 | train time | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline_640 | 640 | | | | |
| finetune_960 | 960 | | | | |
| finetune_1280 | 1280 | | | | |

### Per-class AP50

> The interesting cut. If the hypothesis holds, `pedestrian` / `people` /
> `motor` should improve much more than `bus` / `truck`. If every class
> improves by a similar amount, the gain is coming from something other than
> object scale and the explanation above is wrong.

## What I'd check next

- Whether the gain tracks *object size* or just *more pixels of context* —
  separable by evaluating the 640-trained model at 960 without retraining.
- Whether P2 (stride-4) detection head helps more than raw resolution, at
  lower cost.

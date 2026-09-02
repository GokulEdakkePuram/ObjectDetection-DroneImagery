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

## What it costs

Measured on a rented RTX 4090 with `aerialdet probe` — the batch-size confound
is gone, since `batch: 8` holds at every resolution on a 24 GB card:

| config | per epoch | 50 epochs | peak VRAM |
| --- | ---: | ---: | ---: |
| `baseline_640` | 1.5 min | 1.2 h | 3.3 GB |
| `finetune_960` | 1.6 min | 1.3 h | 8.1 GB |
| `finetune_1280` | 2.1 min | 1.8 h | 13.6 GB |

Note how little resolution costs here: 4x the pixels for 1.42x the time. The
GPU is not the bottleneck at these sizes ([doc 06](06-running-on-rented-gpus.md)),
so the usual accuracy-versus-cost tradeoff is much flatter than expected. Any
conclusion below should be read with that in mind — a result that says "1280 is
worth it" is a weaker claim on hardware where 1280 is nearly free.

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


## Experiment 3: is it the grid, or the pixels?

The resolution result has two candidate mechanisms, and mAP alone cannot
separate them:

1. **Grid density.** More input pixels means more cells on the finest feature
   map, so small objects stop sharing a cell.
2. **Object scale through the whole hierarchy.** More input pixels means the
   object is physically larger at *every* layer, so deeper features — the ones
   with the semantic content — can actually see it.

A P2 (stride-4) head separates them cleanly. At 640 px it produces a 160x160
finest grid: exactly what a 1280 px model reaches at stride 8. Same grid,
a quarter of the input area. If mechanism (1) were the story, they should
score alike.

| run | model | imgsz | finest grid | mAP50-95 | ms/img | peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_640` | yolo11s | 640 | 80x80 | 0.2218 | 13.7 | 3.3 GB |
| `p2_640` | yolo11s-p2 | 640 | **160x160** | 0.2204 | 15.4 | 14.3 GB |
| `finetune_1280` | yolo11s | 1280 | **160x160** | **0.3264** | 19.5 | 17.5 GB |

**They do not score alike. It is not close.**

`p2_640` matched the same-resolution baseline to within 0.6% — the extra
detection level bought nothing at all — and fell 32.5% short of the model with
the same finest grid. So grid density is not the mechanism, and the version of
the stride-floor argument in [doc 01](01-why-small-objects-are-hard.md) that
leans on cell counts is wrong.

### Why it fails

The P2 map is fine, but its features are shallow. It branches off backbone
layer 2 — two convolutions and one C3k2 block deep — so it has a small
receptive field and little semantic content. A stride-4 grid can *localise*
precisely and still be unable to tell a pedestrian from a pole.

Raising the input resolution does something different in kind. The object
grows at every level of the hierarchy, so the deep, semantically rich P3
features finally have enough spatial extent to work with. Fine grids are not
the scarce resource; **fine grids carrying deep features** are.

This also explains why tiled inference behaved the way it did in
[doc 03](03-tiled-inference.md). Tiling raises effective resolution — objects
land on the network at native scale — so it helps a resolution-starved model.
P2 does not raise resolution, only grid density, and so it helps nothing.

### The confound, stated plainly

`p2_640` transfers **297/593** pretrained items against 499/499 for the stock
head: adding the P2 branch shifts every downstream layer index, so the neck
does not match by name. It therefore starts with a partly random neck and gets
the same fixed 50-epoch budget.

That could account for some of the gap and it should be said out loud. But it
does not obviously account for the *shape* of the result. If weak
initialisation were dominating, `p2_640` should have landed clearly *below*
`baseline_640`; instead it landed on top of it, which reads more like an
architecture that contributed nothing than one that was held back.

Separating the two is a further experiment, not an inference: train `p2_640`
for 150 epochs, or initialise `baseline_640` from the same partial transfer
and see whether it loses the same amount.

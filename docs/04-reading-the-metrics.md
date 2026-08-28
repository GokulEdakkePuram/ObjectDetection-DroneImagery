# Reading the metrics honestly

## What mAP actually averages

**AP** for one class is the area under its precision-recall curve, swept by
varying the confidence threshold. **mAP** averages that over classes. The two
numbers reported everywhere differ only in the IoU threshold used to decide
whether a prediction counts as a hit:

- **mAP50** — a box counts if IoU >= 0.5. Measures *"did you find the thing?"*
- **mAP50-95** — averaged over IoU thresholds 0.5, 0.55, ... 0.95. Measures
  *"did you find it and draw the box precisely?"*

On VisDrone the gap between them is unusually wide, and [doc 01's
`IoU = (s-d)/(s+d)`](01-why-small-objects-are-hard.md) explains why: on a 15 px
box, a 5 px error already fails the 0.5 threshold, so the stricter thresholds
are measuring annotation precision as much as model quality.

**Report both.** Quoting only mAP50 on a small-object dataset is the standard
way to make a result look better than it is.

## Traps specific to this dataset

**`max_det` can truncate, but not for the obvious reason.** My first guess was
that VisDrone frames simply contain more than the default cap of 300 objects.
The profile says otherwise: only 3 of 548 validation images exceed 300
annotated boxes, and the densest has 317. On ground-truth density alone the
cap is nearly irrelevant.

The cap still matters, on the *prediction* side. Ultralytics validates at
`conf=0.001`, not the `0.25` used for prediction — deliberately, because mAP
sweeps the confidence threshold and needs the low-confidence tail to trace out
the high-recall end of the PR curve. At that threshold a detector emits far
more candidate boxes than there are objects, so on a 70-object image the 300
cap can bind even though ground truth is nowhere near it. What gets discarded
is exactly the tail that the recall sweep depends on.

So it is worth measuring rather than either ignoring or "fixing" blind:

```bash
uv run aerialdet eval <weights> --imgsz 960   # then re-run with max_det raised
```

If mAP does not move, the cap was not binding and the default stands. That is
a cheap experiment and a better answer than an assumption in either direction.

**Class imbalance hides in the mean.** mAP weights every class equally, so
`awning-tricycle` (3,246 boxes) moves the headline number exactly as much as
`car` (144,867) — a 45:1 imbalance. That cuts both ways: it is why mAP is a fairer summary than
accuracy, and why a jump in mAP can come entirely from one rare class getting
slightly luckier. Always read the per-class AP50 table next to it.

**Evaluating at a different resolution than you trained at** changes the
number. Comparisons are only meaningful when `imgsz` is held fixed, or when the
change in `imgsz` *is* the thing being measured and is stated.

## The comparison this repo produces

`aerialdet eval` writes both a JSON record and a markdown table to `reports/`,
covering every checkpoint passed to it:

```bash
uv run aerialdet eval \
    runs/train/baseline_640/weights/best.pt \
    runs/train/finetune_960/weights/best.pt
```

Keeping the resolved config next to each run's weights
(`runs/train/<name>/aerialdet_config.json`) is what makes a row in that table
traceable back to the exact settings that produced it.

## What a good result looks like here

Published VisDrone numbers vary widely with input resolution and model size,
and single-model results in the ~0.2-0.3 mAP50-95 range at moderate resolution
are respectable — this is a hard benchmark and a 0.9 is not on the table. The
defensible claim for a portfolio is not a leaderboard position. It is: *here is
a controlled ablation, here is the mechanism I expected to see, and here is
whether the data agreed with me.*

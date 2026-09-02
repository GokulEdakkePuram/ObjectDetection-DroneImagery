# Tiled (sliced) inference

Raising `imgsz` during training has a ceiling: memory grows quadratically, and
at some point a batch no longer fits. Tiled inference gets the same effect at
*test* time for a model trained at ordinary resolution — cut the frame into
overlapping tiles, detect in each at native scale, and stitch the results back
together. This is the idea popularised as SAHI; it is implemented directly in
[`tiling.py`](../src/aerialdet/tiling.py) so the mechanics stay visible.

## The geometry

With tile size `T` and overlap fraction `o`, tiles start every
`S = T(1 - o)` pixels. Two things follow.

**Coverage.** Stepping by `S` from 0 leaves a trailing strip narrower than `S`
that no tile reaches. The fix is to place one final tile flush against the far
edge, which is why `_axis_positions` ends with:

```python
if positions[-1] != length - tile:
    positions.append(length - tile)
```

That off-by-one is the most common bug in hand-rolled tiling, and it fails
quietly — you just never detect anything down the right edge of the frame.
`test_tiles_cover_every_pixel` rasterises the tiles onto a mask and asserts
full coverage, including for images smaller than one tile.

**How much overlap is enough.** An object of width `w` is guaranteed to fall
entirely inside *some* tile only when

```
w <= T - S = T * o
```

At `T = 640, o = 0.2` that guarantees objects up to 128 px. Since essentially
every VisDrone object is smaller than that, 0.2 is sufficient here — and going
higher would only add cost. On a dataset with larger objects, this inequality
is how you would pick the number instead of guessing it.

## The merge step

Tiles overlap, so the same object gets detected two or four times, with boxes
in tile-local coordinates. The fix is: offset boxes back to full-frame
coordinates, pool them, then run NMS once globally.

The subtlety is that NMS must be **class-aware**. In a dense aerial scene a
pedestrian's box legitimately overlaps the car they are standing beside; plain
NMS would delete one. The standard trick is to shift each class into its own
coordinate band before the IoU test, so boxes of different classes can never
overlap numerically:

```python
offsets = classes.to(boxes.dtype) * (boxes.max() + 1)
keep = nms(boxes + offsets[:, None], scores, iou_threshold)
```

`test_merge_keeps_overlapping_boxes_of_different_classes` pins this down.

## The cost

Tiling is not free. A 1400x1080 frame at `T=640, o=0.2` produces **6 tiles**
(3 across, 2 down), so inference costs ~6 forward passes instead of one, plus
the merge. Throughput matters for drone workloads, so this is a tradeoff to
measure, not a free win:

```bash
uv run aerialdet tiled-predict runs/train/finetune_960/weights/best.pt \
    data/VisDrone/images/val/<some>.jpg --tile 640 --overlap 0.2
```

## Measuring it

```bash
uv run aerialdet tiled-eval runs/train/finetune_960/weights/best.pt
```

This scores both modes over the same images and reports mAP next to
latency. Both go through Ultralytics' own `match_predictions` and
`DetMetrics` — scoring tiled predictions with a hand-rolled metric would
confound *what* is being measured with *how*, and the point is to compare
the two modes, not two metric implementations. As a side effect the numbers
are comparable to `aerialdet eval`.

Both modes also run at `conf=0.001`, the threshold Ultralytics validates at
([doc 04](04-reading-the-metrics.md)). Using the prediction default of 0.25
would truncate the low-confidence tail that the recall end of the PR curve
depends on, and it would truncate it differently for the two modes.

## Results: tiling loses, and that is the interesting part

`finetune_1280`, full 548-image val split, tiles of 640 px at 0.2 overlap:

| mode | mAP50-95 | mAP50 | precision | recall | ms/frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| whole frame @1280 | **0.3112** | 0.5029 | 0.6326 | 0.5115 | 16 |
| tiled 640 / 0.2 | 0.2799 | 0.4617 | 0.5562 | 0.4981 | 25 |

**−10.1% mAP50-95 for 1.6x the latency.** Tiling made this model worse.

That is worth keeping rather than burying, because the mechanism is legible in
the precision/recall split. Tiling did not fail to find things — it found more
and was wrong more often. Its detections rose from 148k to 425k, and the loss
is concentrated in precision (0.633 → 0.556), which is what tile-seam
duplicates and edge-truncated boxes produce.

### Why an earlier check said the opposite

On a weakly-trained checkpoint (`yolo11s` at 0.10 mAP50-95), tiling scored
**+45%**. On a well-trained 1280 model it costs 10%. Both are real, and
together they say something neither says alone:

> Tiling *substitutes* for resolution rather than adding to it.

A 640 px model cannot resolve a 20 px car — tiling gives it the pixels and
helps enormously. A model already trained at 1280 has bought those pixels
already, so tiling contributes nothing new and keeps only its costs. This is
the same stride-floor argument from [doc 01](01-why-small-objects-are-hard.md)
seen from the other side, and it is consistent with the r = −0.88 correlation
between object size and resolution gain in
[doc 02](02-resolution-and-stride.md).

### The test, and it passed

Running the same comparison on `baseline_640` — the model that *cannot*
resolve small objects — flips the sign:

| checkpoint | mode | mAP50-95 | precision | recall | ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline_640` | whole @640 | 0.2056 | 0.534 | 0.372 | 13 |
| `baseline_640` | tiled 640 | **0.2309** | 0.493 | 0.436 | 26 |
| `finetune_1280` | whole @1280 | **0.3112** | 0.633 | 0.511 | 16 |
| `finetune_1280` | tiled 640 | 0.2799 | 0.556 | 0.498 | 25 |

**+12.3% where the model is resolution-starved, −10.1% where it is not.**

The mechanism is identical in both cases — tiling trades precision for recall,
because seams create duplicates and truncated boxes while native-scale crops
expose objects the downscaled frame lost. Only the *balance* changes. At 640
the recall gain is large (+17%) and outweighs the precision cost. At 1280 the
model has already found those objects, so recall barely moves (−3%) and the
precision cost is all that is left.

### But do not tile — raise the resolution instead

The substitution is real but partial, and it loses on cost:

```
tiled 640   mAP 0.2309 at 26 ms
whole 1280  mAP 0.3112 at 16 ms     <- better on both axes
```

Tiling recovers only **24%** of what moving 640 → 1280 buys, while costing
*more* wall clock than the 1280 model does. It is Pareto-dominated: there is
no operating point on this hardware where tiling is the right answer.

That conclusion is specific and worth qualifying. Tiling earns its place when
resolution is not a free variable — a fixed-input exported model, an edge
accelerator with a hard input cap, or frames so large that the whole image
cannot fit in memory at native scale. None of those apply here, and on a
laptop or a rented GPU raising `imgsz` is simply the better instrument.

### Two caveats on the numbers

**The comparison is skewed toward tiling, and tiling still loses.** The
whole-frame arm averages 271 detections per image against a `max_det` cap of
300 — it is running into the ceiling described in
[doc 04](04-reading-the-metrics.md). The tiled arm gets up to 300 *per tile*
before merging, so roughly six times the budget. Removing that asymmetry would
likely widen the gap, not close it.

**The absolute whole-frame figure here (0.3112) is below what `aerialdet eval`
reports for the same checkpoint (0.3260).** Ultralytics' validator uses
rectangular batching and its own preprocessing; this module drives
`model.predict` per image. Both arms of *this* comparison go through the
identical path, so the comparison between them is sound — but the absolute
number is not directly interchangeable with the eval table.

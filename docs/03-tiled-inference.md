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

## Results

> Pending a properly trained checkpoint. An early check on 12 val images with
> a weakly-trained `yolo11s` gave **+45% mAP50-95 for 1.5x the latency** —
> enough to show the machinery works and that the effect points the right way,
> but far too few images and too weak a model to quote as a result.

| mode | tiles/frame | mAP50-95 | mAP50 | ms/frame |
| --- | ---: | ---: | ---: | ---: |
| whole frame @960 | 1 | | | |
| tiled 640 / 0.2 | 6 | | | |
| tiled 512 / 0.3 | 12 | | | |

The number worth reporting is not "tiling wins" but the exchange rate: how
much mAP per millisecond, and whether the gain concentrates in the smallest
classes (`pedestrian`, `people`, `motor`) as the argument in
[doc 01](01-why-small-objects-are-hard.md) predicts. If tiling helps `bus`
as much as `pedestrian`, the stated mechanism is wrong.

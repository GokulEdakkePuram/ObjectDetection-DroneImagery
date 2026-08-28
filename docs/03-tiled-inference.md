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

## Results

> Compare whole-frame vs tiled inference with the *same* checkpoint. Report
> mAP50-95 and latency per frame for both. The honest framing is a
> accuracy-per-millisecond curve, not a single number.

| mode | tiles/frame | mAP50-95 | ms/frame |
| --- | ---: | ---: | ---: |
| whole frame @960 | 1 | | |
| tiled 640 / 0.2 | 6 | | |
| tiled 512 / 0.3 | | | |

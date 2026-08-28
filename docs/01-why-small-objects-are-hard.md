# Why small-object detection is a different problem

Detectors are usually benchmarked on COCO, where the median object covers a
meaningful fraction of the frame. Aerial imagery inverts that. A drone at 80 m
sees cars as ~20 px boxes in a 1400 px-wide frame. Three separate things break
at once, and they compound.

## 1. The network's stride sets a resolution floor

YOLO11 predicts from three feature maps, at strides 8, 16 and 32 relative to
the network input (conventionally P3, P4, P5). The finest of these, P3, has one
cell per 8x8 input pixels. An object has to be at least a few cells across
before there is enough spatial signal to regress a box from.

That gives a hard floor. An object smaller than ~8 px **in network input
space** is essentially invisible to the head, no matter how good the backbone
is. And "network input space" is not the same as the original image:

| input size | scale factor on a 1400 px frame | a 20 px car becomes | cells on P3 |
| ---: | ---: | ---: | ---: |
| 640 | 0.46 | 9 px | 1.1 |
| 960 | 0.69 | 14 px | 1.7 |
| 1280 | 0.91 | 18 px | 2.3 |

At 640 px, most VisDrone objects land on roughly a *single* P3 cell. This is
the single biggest reason a stock-configured YOLO does badly here, and it is
why [the resolution ablation](02-resolution-and-stride.md) is the first
experiment in this repo rather than a hyperparameter sweep.

## 2. IoU is unforgiving at small scales

Localisation error is roughly constant in pixels, but IoU is scale-relative.
For a square box of side `s` displaced by `d` pixels, the overlap works out to
a strikingly simple expression:

```
IoU = (s - d) / (s + d)
```

Setting that equal to the 0.5 detection threshold gives `d = s / 3`. **A
displacement of one third of the box's side is enough to turn a detection into
both a false positive and a false negative.**

In practice:

| box side | displacement that drops IoU below 0.5 |
| ---: | ---: |
| 15 px | 5 px |
| 50 px | 17 px |
| 150 px | 50 px |

Five pixels. That is within the noise of human annotation. So on this dataset a
large share of the apparent "error" at high IoU thresholds is really the metric
resolving finer than the labels do — which is why mAP50-95 sits so far below
mAP50 here, and why both numbers are reported in
[the metrics note](04-reading-the-metrics.md).

## 3. Scenes are dense, so NMS becomes a real decision

VisDrone images routinely contain hundreds of annotated objects, packed at
parking-lot density. Two consequences:

- The default `max_det=300` silently truncates predictions on the busiest
  frames, capping recall before the model is even at fault.
- NMS IoU thresholds tuned for sparse COCO scenes will merge genuinely
  distinct adjacent cars.

## What this repo does about it

Each of the three problems gets a concrete, measured response rather than an
assertion:

| problem | response | where |
| --- | --- | --- |
| stride resolution floor | train and evaluate at higher `imgsz` | [`configs/`](../configs), [doc 02](02-resolution-and-stride.md) |
| stride floor, at inference time | overlapping tiled inference | [`tiling.py`](../src/aerialdet/tiling.py), [doc 03](03-tiled-inference.md) |
| dense scenes | raise `max_det`, tune NMS IoU | [doc 04](04-reading-the-metrics.md) |

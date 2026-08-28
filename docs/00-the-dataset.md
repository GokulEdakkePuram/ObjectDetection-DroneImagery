# VisDrone2019-DET

[VisDrone](https://github.com/VisDrone/VisDrone-Dataset) is a drone-captured
detection benchmark from Tianjin University: 10 classes of road users, shot
across 14 Chinese cities, at varying altitudes, weather and times of day.

| split | images | used for |
| --- | ---: | --- |
| train | 6,471 | fine-tuning |
| val | 548 | model selection and every number reported here |
| test-dev | 1,610 | held out; touched once, at the end |

Ten classes: `pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`,
`tricycle`, `awning-tricycle`, `bus`, `motor`.

## Why this dataset

Three properties make it a genuinely hard fine-tuning target rather than a
transfer-learning formality:

- **Scale.** Most objects are "small" by the COCO convention (area < 32x32 px).
  See [doc 01](01-why-small-objects-are-hard.md).
- **Density.** Hundreds of objects per frame is normal, not exceptional.
- **Label semantics that COCO does not share.** `pedestrian` vs `people` is a
  *pose* distinction (walking/standing vs any other posture), and
  `awning-tricycle` has no COCO analogue at all. So the pretrained head cannot
  simply be re-used — this is real fine-tuning, not a class remap.

## Label format and the ignored-region trap

VisDrone ships annotations as CSV:
`x, y, w, h, score, category, truncation, occlusion`, with pixel coordinates
and 1-indexed categories. YOLO wants normalised `cls xc yc w h` with 0-indexed
classes.

The conversion Ultralytics ships handles both, plus one thing that is easy to
miss:

```python
if row[4] != "0":       # score == 0 marks an *ignored region*
    ...
    cls = int(row[5]) - 1
```

Rows with `score == 0` are regions the benchmark explicitly excludes from
evaluation — crowds too dense to annotate individually, and areas outside the
region of interest. Training on them as background teaches the model to
suppress exactly the objects it is supposed to find. Dropping them is correct;
a stricter treatment would mask those pixels out of the loss entirely.

## Getting it

```bash
make data      # ~2 GB download, then converts annotations to YOLO format
make profile   # writes reports/dataset_profile.md
```

`make profile` is the step worth not skipping. It reports class balance and the
small/medium/large split of box areas, in pixels — because "small" only means
something in pixels, and the argument in doc 01 depends on the actual numbers
rather than on the claim that VisDrone is hard.

## Profile

> Paste the table from `reports/dataset_profile.md` here once it has run.

| split | images | boxes | boxes/img | small % | medium % | large % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | | | | | | |
| val | | | | | | |

### Class balance

> The long tail matters. If `car` is an order of magnitude more common than
> `awning-tricycle`, then overall mAP is largely a `car` score, and the
> per-class table is the one to read.

## Licence

VisDrone is released for academic use. Check the upstream terms before using
anything trained on it commercially.

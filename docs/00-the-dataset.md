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
- **Density.** 53 objects per training image on average, and up to 902.
- **Label semantics that COCO does not share.** `pedestrian` vs `people` is a
  *pose* distinction (walking/standing vs any other posture), and
  `awning-tricycle` has no COCO analogue at all. So the pretrained head cannot
  simply be re-used — this is real fine-tuning, not a class remap. See below
  for how much of the head actually survives.

## What the pretrained weights actually transfer

Starting a run prints one line that is easy to scroll past:

```
Remapped 4/10 cls head rows from pretrained weights by class name
Transferred 451/499 items from pretrained weights
```

Ultralytics matches target class names against the checkpoint's names
(case-insensitively) and copies just those rows of the classification head;
every unmatched row is randomly initialised. Only four VisDrone names exist
verbatim in COCO:

| inherits its COCO row | starts from scratch | why not |
| --- | --- | --- |
| `car`, `bus`, `truck`, `bicycle` | `pedestrian`, `people` | COCO calls it `person`, and neither VisDrone class means quite that |
| | `motor` | COCO calls it `motorcycle` |
| | `van`, `tricycle`, `awning-tricycle` | no COCO analogue |

So **6 of 10 classifier rows begin as noise** — and two of them,
`pedestrian` and `people`, account for 31% of all training boxes. The backbone
transfers nearly intact (451/499 tensors), which is where the real value of
pretraining lies; the head largely does not.

Two practical consequences. Freezing the backbone would be defensible here;
freezing the head would not. And a plain zero-shot COCO baseline is not a
meaningful control on this dataset — the classes do not line up — which is why
the control run in this repo is a *fine-tuned* model at stock resolution
rather than an off-the-shelf one.

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

Measured with `make profile` (full output in `reports/dataset_profile.md`).
Box areas are in pixels, bucketed by the COCO convention.

| split | images | boxes | boxes/img | small % | medium % | large % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 6,471 | 343,205 | 53.0 | 60.5 | 34.0 | 5.5 |
| val | 548 | 38,759 | 70.7 | 68.6 | 28.7 | 2.8 |
| test | 1,610 | 75,102 | 46.6 | 67.7 | 29.1 | 3.2 |

Three things fall out of this that were worth measuring rather than assuming:

**The small-object claim holds, but it is 60%, not 90%.** A clear majority of
boxes are under 32x32 px, and another third are under 96x96 -- only 5.5% of
training boxes are "large". So resolution should matter a lot, but there is
also a real medium-scale population that a 640 px model can already handle.

**Val is denser and harder than train.** 70.7 boxes per image against 53.0,
and 68.6% small against 60.5%. The validation split is not a random sample of
train, so val mAP is a slightly pessimistic estimate — fine for comparing runs
against each other, worth remembering before comparing against a paper.

**No empty images in any split.** Every frame has at least one object, so
there is no background-only imagery to calibrate the false-positive rate on.

### Class balance (train)

| class | boxes | share |
| --- | ---: | ---: |
| car | 144,867 | 42.2% |
| pedestrian | 79,337 | 23.1% |
| motor | 29,647 | 8.6% |
| people | 27,059 | 7.9% |
| van | 24,956 | 7.3% |
| truck | 12,875 | 3.8% |
| bicycle | 10,480 | 3.1% |
| bus | 5,926 | 1.7% |
| tricycle | 4,812 | 1.4% |
| awning-tricycle | 3,246 | 0.9% |

`car` outnumbers `awning-tricycle` 45:1. Since mAP weights every class
equally, `awning-tricycle` -- with 0.9% of the boxes -- moves the headline
number exactly as much as `car` does. Expect the rare classes to be both the
worst-scoring and the noisiest between runs, and read the per-class table
before believing any change in the mean. See
[doc 04](04-reading-the-metrics.md).

## Licence

VisDrone is released for academic use. Check the upstream terms before using
anything trained on it commercially.

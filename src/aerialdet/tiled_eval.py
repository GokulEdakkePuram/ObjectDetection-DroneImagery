"""Measuring what tiled inference actually buys.

`tiling.py` implements sliced inference and `test_tiling.py` proves its
geometry, but neither says whether it *helps*. This module answers that with
an mAP number, by running both modes over the same split and scoring them
through the same code path.

That last part matters. Comparing a hand-rolled metric against Ultralytics'
would confound the thing being measured with the way it is measured, so both
modes here go through Ultralytics' own ``match_predictions`` and
``DetMetrics``. The numbers are therefore comparable to each other *and* to
``aerialdet eval``.

Tiling is not free -- a 1400x1080 frame becomes six 640px tiles -- so latency
is reported alongside accuracy. The honest summary of sliced inference is a
cost/benefit pair, never a single number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .paths import configure_ultralytics
from .stats import IMAGE_SUFFIXES
from .tiling import predict_tiled

# Ultralytics validates at this confidence, not the 0.25 used for prediction:
# mAP sweeps the threshold and needs the low-confidence tail to trace the
# high-recall end of the PR curve. Both modes must use the same value.
VAL_CONF = 0.001


class _Matcher:
    """Borrows Ultralytics' IoU matching so both modes are scored identically."""

    def __init__(self) -> None:
        self.iouv = torch.linspace(0.5, 0.95, 10)

    from ultralytics.engine.validator import BaseValidator as _BV

    match_predictions = _BV.match_predictions
    del _BV


@dataclass
class ModeResult:
    """Accuracy and cost for one inference mode."""

    mode: str
    images: int
    map50_95: float
    map50: float
    precision: float
    recall: float
    ms_per_image: float
    detections: int

    def row(self) -> str:
        return (
            f"| {self.mode} | {self.map50_95:.4f} | {self.map50:.4f} "
            f"| {self.precision:.4f} | {self.recall:.4f} "
            f"| {self.ms_per_image:.0f} | {self.detections:,} |"
        )


def _load_ground_truth(
    label_path: Path, width: int, height: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a YOLO label file as pixel xyxy boxes and class ids."""
    boxes, classes = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid, xc, yc, w, h = (float(v) for v in parts[:5])
            xc, w = xc * width, w * width
            yc, h = yc * height, h * height
            boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
            classes.append(cid)

    if not boxes:
        return torch.zeros((0, 4)), torch.zeros(0)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.float32)


def _predict_whole(model, image, imgsz: int, conf: float, device: str):
    result = model.predict(image, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    det = result.boxes
    if det is None or len(det) == 0:
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0)
    return det.xyxy.cpu(), det.conf.cpu(), det.cls.cpu()


def evaluate_mode(
    model,
    images: list[Path],
    labels_dir: Path,
    names: dict[int, str],
    mode: str,
    imgsz: int = 960,
    tile: int = 640,
    overlap: float = 0.2,
    conf: float = VAL_CONF,
    iou: float = 0.7,
    device: str = "auto",
) -> ModeResult:
    """Score one inference mode over a list of images."""
    import cv2
    from ultralytics.utils.metrics import DetMetrics, box_iou

    from .config import resolve_device

    device = resolve_device(device)
    matcher = _Matcher()
    metrics = DetMetrics(names=names)

    elapsed, n_det = 0.0, 0

    for image_path in images:
        array = cv2.imread(str(image_path))
        if array is None:
            continue
        height, width = array.shape[:2]

        start = time.perf_counter()
        if mode == "tiled":
            out = predict_tiled(
                model, array, tile=tile, overlap=overlap, conf=conf, iou=iou, device=device
            )
            p_boxes, p_conf, p_cls = out["boxes"], out["scores"], out["classes"]
        else:
            p_boxes, p_conf, p_cls = _predict_whole(model, array, imgsz, conf, device)
        elapsed += time.perf_counter() - start
        n_det += len(p_boxes)

        t_boxes, t_cls = _load_ground_truth(labels_dir / f"{image_path.stem}.txt", width, height)

        if len(t_boxes) == 0 or len(p_boxes) == 0:
            tp = np.zeros((len(p_boxes), 10), dtype=bool)
        else:
            tp = matcher.match_predictions(p_cls, t_cls, box_iou(t_boxes, p_boxes)).cpu().numpy()

        target_cls = t_cls.numpy()
        metrics.update_stats(
            {
                "tp": tp,
                "conf": p_conf.numpy(),
                "pred_cls": p_cls.numpy(),
                "target_cls": target_cls,
                "target_img": np.unique(target_cls),
                "im_name": image_path.name,
            }
        )

    metrics.process()
    r = metrics.results_dict
    return ModeResult(
        mode=mode,
        images=len(images),
        map50_95=float(r.get("metrics/mAP50-95(B)", 0.0)),
        map50=float(r.get("metrics/mAP50(B)", 0.0)),
        precision=float(r.get("metrics/precision(B)", 0.0)),
        recall=float(r.get("metrics/recall(B)", 0.0)),
        ms_per_image=1000 * elapsed / max(len(images), 1),
        detections=n_det,
    )


def compare(
    weights: str,
    data: str = "VisDrone.yaml",
    split: str = "val",
    limit: int | None = None,
    **kwargs: Any,
) -> list[ModeResult]:
    """Score whole-frame and tiled inference on the same images."""
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    configure_ultralytics()
    spec = check_det_dataset(data)
    images_dir = Path(spec[split])
    labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if limit:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    model = YOLO(weights)
    results = [
        evaluate_mode(model, images, labels_dir, spec["names"], mode, **kwargs)
        for mode in ("whole", "tiled")
    ]

    print(f"\n{len(images)} images from the {split} split, conf={kwargs.get('conf', VAL_CONF)}")
    print("\n| mode | mAP50-95 | mAP50 | precision | recall | ms/img | dets |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        print(r.row())

    whole, tiled = results
    if whole.map50_95 > 0:
        gain = 100 * (tiled.map50_95 / whole.map50_95 - 1)
        slowdown = tiled.ms_per_image / max(whole.ms_per_image, 1e-9)
        print(f"\ntiled vs whole-frame: {gain:+.1f}% mAP50-95 for {slowdown:.1f}x the latency")

    return results

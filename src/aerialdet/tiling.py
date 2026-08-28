"""Sliced (tiled) inference for small objects.

A 1400x1080 drone frame fed to a detector at 640 px is downscaled ~2.2x, so a
20 px car becomes 9 px -- below what the network's stride-8 feature map can
resolve. Tiled inference sidesteps this: cut the frame into overlapping tiles,
run the detector on each tile at native resolution, map the boxes back to
full-frame coordinates, then de-duplicate across tile seams with NMS.

This is the same idea as SAHI, implemented directly so the mechanics are
visible rather than hidden behind a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms


@dataclass(frozen=True)
class Tile:
    """A tile's pixel window in the source image, as (x0, y0, x1, y1)."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def offset(self) -> tuple[int, int]:
        return self.x0, self.y0


def _axis_positions(length: int, tile: int, stride: int) -> list[int]:
    """Tile start offsets along one axis, with the final tile flush to the edge."""
    if length <= tile:
        return [0]

    positions = list(range(0, length - tile + 1, stride))
    # Without this, a trailing strip up to `stride` px wide is never seen.
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def compute_tiles(width: int, height: int, tile: int = 640, overlap: float = 0.2) -> list[Tile]:
    """Lay overlapping tiles over a WxH image.

    ``overlap`` is the fraction of a tile shared with its neighbour. Some
    overlap is mandatory: an object straddling a seam is truncated in both
    tiles, and only a generous overlap guarantees it lands whole in one of them.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")

    stride = max(1, int(round(tile * (1.0 - overlap))))
    return [
        Tile(x, y, min(x + tile, width), min(y + tile, height))
        for y in _axis_positions(height, tile, stride)
        for x in _axis_positions(width, tile, stride)
    ]


def merge_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Class-aware NMS over detections pooled from every tile.

    Boxes are offset by a per-class stride before NMS so that overlapping
    detections of *different* classes never suppress each other -- the standard
    batched-NMS trick.
    """
    if boxes.numel() == 0:
        return boxes, scores, classes

    max_coord = boxes.max()
    offsets = classes.to(boxes.dtype) * (max_coord + 1)
    keep = nms(boxes + offsets[:, None], scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]


def predict_tiled(
    model,
    image: str | Path | np.ndarray,
    tile: int = 640,
    overlap: float = 0.2,
    conf: float = 0.25,
    iou: float = 0.5,
    device: str = "auto",
    batch_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Run tiled inference on one image.

    Returns full-image ``xyxy`` boxes, confidence scores, and class ids.
    """
    import cv2

    from .config import resolve_device

    if isinstance(image, (str, Path)):
        array = cv2.imread(str(image))
        if array is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        array = image

    height, width = array.shape[:2]
    tiles = compute_tiles(width, height, tile=tile, overlap=overlap)
    device = resolve_device(device)

    all_boxes: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    all_classes: list[torch.Tensor] = []

    for start in range(0, len(tiles), batch_size):
        chunk = tiles[start : start + batch_size]
        crops = [array[t.y0 : t.y1, t.x0 : t.x1] for t in chunk]
        results = model.predict(crops, conf=conf, imgsz=tile, device=device, verbose=False)

        for t, result in zip(chunk, results, strict=True):
            det = result.boxes
            if det is None or len(det) == 0:
                continue
            xyxy = det.xyxy.cpu().clone()
            # Tile-local -> full-image coordinates.
            xyxy[:, [0, 2]] += t.x0
            xyxy[:, [1, 3]] += t.y0
            all_boxes.append(xyxy)
            all_scores.append(det.conf.cpu())
            all_classes.append(det.cls.cpu())

    if not all_boxes:
        return {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "classes": torch.zeros(0),
            "n_tiles": len(tiles),
        }

    boxes, scores, classes = merge_detections(
        torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_classes), iou_threshold=iou
    )
    return {"boxes": boxes, "scores": scores, "classes": classes, "n_tiles": len(tiles)}

"""Tiling geometry is pure arithmetic, so it is worth pinning down exactly.

These are the tests that catch the classic sliced-inference bugs: a trailing
strip of the image that no tile covers, and cross-class suppression during the
merge step.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aerialdet.tiling import Tile, compute_tiles, merge_detections


def _coverage_mask(width: int, height: int, tiles) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for t in tiles:
        mask[t.y0 : t.y1, t.x0 : t.x1] = True
    return mask


@pytest.mark.parametrize(
    ("width", "height", "tile", "overlap"),
    [
        (1400, 1080, 640, 0.2),
        (1920, 1080, 512, 0.25),
        (1000, 1000, 640, 0.0),
        (300, 200, 640, 0.2),  # image smaller than one tile
        (641, 641, 640, 0.2),  # one pixel past a tile boundary
    ],
)
def test_tiles_cover_every_pixel(width, height, tile, overlap):
    tiles = compute_tiles(width, height, tile=tile, overlap=overlap)
    assert _coverage_mask(width, height, tiles).all(), "tiling left a gap"


def test_tiles_stay_inside_the_image():
    tiles = compute_tiles(1400, 1080, tile=640, overlap=0.2)
    assert all(0 <= t.x0 < t.x1 <= 1400 and 0 <= t.y0 < t.y1 <= 1080 for t in tiles)


def test_small_image_yields_single_tile():
    assert compute_tiles(300, 200, tile=640) == [Tile(0, 0, 300, 200)]


def test_neighbouring_tiles_actually_overlap():
    tiles = compute_tiles(1400, 640, tile=640, overlap=0.5)
    xs = sorted({t.x0 for t in tiles})
    # stride = 320, so consecutive starts must be no further apart than that.
    assert all(b - a <= 320 for a, b in zip(xs, xs[1:], strict=False))


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 1.5])
def test_invalid_overlap_rejected(overlap):
    with pytest.raises(ValueError, match="overlap"):
        compute_tiles(1000, 1000, tile=640, overlap=overlap)


def test_merge_suppresses_duplicates_of_the_same_class():
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 12.0, 52.0, 52.0]])
    scores = torch.tensor([0.9, 0.8])
    classes = torch.tensor([3.0, 3.0])

    kept_boxes, kept_scores, _ = merge_detections(boxes, scores, classes, iou_threshold=0.5)

    assert len(kept_boxes) == 1
    assert kept_scores[0] == pytest.approx(0.9)


def test_merge_keeps_overlapping_boxes_of_different_classes():
    """A pedestrian standing against a car must survive the same-region merge."""
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [11.0, 11.0, 51.0, 51.0]])
    scores = torch.tensor([0.9, 0.85])
    classes = torch.tensor([0.0, 3.0])

    kept_boxes, _, kept_classes = merge_detections(boxes, scores, classes, iou_threshold=0.5)

    assert len(kept_boxes) == 2
    assert set(kept_classes.tolist()) == {0.0, 3.0}


def test_merge_handles_no_detections():
    boxes, scores, classes = merge_detections(torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0))
    assert len(boxes) == 0

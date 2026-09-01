"""Ground-truth parsing for tiled evaluation.

YOLO labels are normalised centre-format; IoU matching needs absolute xyxy.
Getting that conversion subtly wrong does not crash -- it just quietly
produces a worse mAP, which would look like a modelling result rather than a
bug. Hence tests.
"""

from __future__ import annotations

import torch

from aerialdet.tiled_eval import _load_ground_truth


def test_converts_normalised_centre_format_to_pixel_corners(tmp_path):
    label = tmp_path / "frame.txt"
    # A box centred in a 1000x500 image, half its width and a fifth its height.
    label.write_text("3 0.5 0.5 0.5 0.2\n")

    boxes, classes = _load_ground_truth(label, width=1000, height=500)

    assert classes.tolist() == [3.0]
    assert boxes.tolist() == [[250.0, 200.0, 750.0, 300.0]]


def test_handles_a_box_touching_the_image_corner(tmp_path):
    label = tmp_path / "frame.txt"
    label.write_text("0 0.05 0.05 0.1 0.1\n")

    boxes, _ = _load_ground_truth(label, width=200, height=200)

    assert boxes.tolist() == [[0.0, 0.0, 20.0, 20.0]]


def test_reads_every_row(tmp_path):
    label = tmp_path / "frame.txt"
    label.write_text("0 0.1 0.1 0.1 0.1\n5 0.9 0.9 0.05 0.05\n2 0.5 0.5 0.2 0.2\n")

    boxes, classes = _load_ground_truth(label, width=640, height=640)

    assert len(boxes) == 3
    assert classes.tolist() == [0.0, 5.0, 2.0]


def test_missing_label_file_is_empty_not_an_error(tmp_path):
    """An image with no label file is a background image, not a failure."""
    boxes, classes = _load_ground_truth(tmp_path / "absent.txt", 100, 100)

    assert boxes.shape == (0, 4)
    assert classes.shape == (0,)


def test_empty_and_malformed_lines_are_skipped(tmp_path):
    label = tmp_path / "frame.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n\n1 0.5\n")

    boxes, _ = _load_ground_truth(label, width=100, height=100)

    assert len(boxes) == 1


def test_boxes_are_float32_for_iou(tmp_path):
    """box_iou needs float; an integer tensor would raise deep inside matching."""
    label = tmp_path / "frame.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n")

    boxes, _ = _load_ground_truth(label, width=100, height=100)

    assert boxes.dtype == torch.float32

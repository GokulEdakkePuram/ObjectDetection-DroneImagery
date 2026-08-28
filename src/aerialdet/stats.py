"""Dataset profiling.

Before touching a hyperparameter, you should be able to say *why* a dataset is
hard. For VisDrone the answer is object scale: the overwhelming majority of
boxes are "small" under the COCO definition (area < 32x32 px), and a detector
trained at the default 640 px shrinks them further. This module measures that
instead of assuming it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

# COCO's object-size convention, in pixels of box area.
SMALL_MAX_AREA = 32 * 32
MEDIUM_MAX_AREA = 96 * 96

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class SplitStats:
    """Aggregate label statistics for one dataset split."""

    split: str
    n_images: int = 0
    n_empty_images: int = 0
    n_boxes: int = 0
    class_counts: Counter[int] = field(default_factory=Counter)
    size_buckets: Counter[str] = field(default_factory=Counter)
    boxes_per_image: list[int] = field(default_factory=list)
    box_areas: list[float] = field(default_factory=list)

    @property
    def small_fraction(self) -> float:
        """Share of boxes that are 'small' by the COCO area convention."""
        return self.size_buckets["small"] / self.n_boxes if self.n_boxes else 0.0

    def to_dict(self, names: dict[int, str] | None = None) -> dict[str, Any]:
        names = names or {}
        return {
            "split": self.split,
            "n_images": self.n_images,
            "n_empty_images": self.n_empty_images,
            "n_boxes": self.n_boxes,
            "boxes_per_image_mean": (
                round(self.n_boxes / self.n_images, 2) if self.n_images else 0.0
            ),
            "boxes_per_image_max": max(self.boxes_per_image, default=0),
            "small_fraction": round(self.small_fraction, 4),
            "size_buckets": dict(self.size_buckets),
            "class_counts": {
                names.get(cid, str(cid)): count for cid, count in sorted(self.class_counts.items())
            },
        }


def _bucket(area_px: float) -> str:
    if area_px < SMALL_MAX_AREA:
        return "small"
    if area_px < MEDIUM_MAX_AREA:
        return "medium"
    return "large"


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def profile_split(images_dir: Path, labels_dir: Path, split: str) -> SplitStats:
    """Walk a YOLO-format split and measure class balance and box scale.

    Box areas are converted from normalized YOLO units back to pixels, because
    "small" is only meaningful in pixels -- a normalized area hides the fact
    that VisDrone frames are ~1400x1080.
    """
    stats = SplitStats(split=split)

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = _find_image(images_dir, label_path.stem)
        if image_path is None:
            continue

        # PIL reads only the header here, so this stays cheap across 6k images.
        with Image.open(image_path) as img:
            width, height = img.size

        stats.n_images += 1
        n_boxes_here = 0

        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            _, _, w_norm, h_norm = (float(p) for p in parts[1:5])

            area_px = (w_norm * width) * (h_norm * height)
            stats.class_counts[class_id] += 1
            stats.size_buckets[_bucket(area_px)] += 1
            stats.box_areas.append(area_px)
            n_boxes_here += 1

        stats.n_boxes += n_boxes_here
        stats.boxes_per_image.append(n_boxes_here)
        if n_boxes_here == 0:
            stats.n_empty_images += 1

    return stats


def profile_dataset(data_yaml: str = "VisDrone.yaml") -> dict[str, SplitStats]:
    """Profile every split declared in an Ultralytics dataset YAML."""
    from ultralytics.data.utils import check_det_dataset

    from .paths import configure_ultralytics

    configure_ultralytics()
    spec = check_det_dataset(data_yaml)

    results: dict[str, SplitStats] = {}
    for split in ("train", "val", "test"):
        images_dir = spec.get(split)
        if not images_dir:
            continue
        images_dir = Path(images_dir)
        if not images_dir.is_dir():
            continue
        labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))
        if not labels_dir.is_dir():
            continue
        results[split] = profile_split(images_dir, labels_dir, split)

    return results


def write_report(
    stats: dict[str, SplitStats],
    names: dict[int, str],
    out_dir: Path,
) -> Path:
    """Write a JSON summary and a markdown table of the profile."""
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {split: s.to_dict(names) for split, s in stats.items()}
    (out_dir / "dataset_stats.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# VisDrone2019-DET dataset profile",
        "",
        "| split | images | boxes | boxes/img | small % | medium % | large % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, s in stats.items():
        total = s.n_boxes or 1
        lines.append(
            f"| {split} | {s.n_images:,} | {s.n_boxes:,} | {s.n_boxes / max(s.n_images, 1):.1f} "
            f"| {100 * s.size_buckets['small'] / total:.1f} "
            f"| {100 * s.size_buckets['medium'] / total:.1f} "
            f"| {100 * s.size_buckets['large'] / total:.1f} |"
        )

    if "train" in stats:
        lines += [
            "",
            "## Class balance (train)",
            "",
            "| class | boxes | share |",
            "| --- | ---: | ---: |",
        ]
        train = stats["train"]
        total = train.n_boxes or 1
        for cid, count in train.class_counts.most_common():
            lines.append(f"| {names.get(cid, cid)} | {count:,} | {100 * count / total:.1f}% |")

    report = out_dir / "dataset_profile.md"
    report.write_text("\n".join(lines) + "\n")
    return report

"""Validation and cross-run comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import resolve_device
from .paths import REPORTS_DIR, RUNS_DIR, configure_ultralytics


def training_imgsz(weights: str) -> int | None:
    """Recover the resolution a checkpoint was trained at, from its run dir."""
    import yaml

    args = Path(weights).parent.parent / "args.yaml"
    if args.exists():
        return (yaml.safe_load(args.read_text()) or {}).get("imgsz")
    return None


def evaluate(
    weights: str,
    data: str = "VisDrone.yaml",
    imgsz: int | None = None,
    split: str = "val",
    device: str = "auto",
    batch: int = 8,
) -> dict[str, Any]:
    """Run validation for one checkpoint and return a flat metrics dict.

    ``imgsz`` defaults to the resolution the checkpoint was *trained* at, read
    back from its run directory. A fixed default silently handicaps a
    high-resolution model -- evaluating a 1280 run at 960 measures something
    other than the ablation it belongs to.
    """
    from ultralytics import YOLO

    configure_ultralytics()
    if imgsz is None:
        imgsz = training_imgsz(weights) or 960
    model = YOLO(weights)
    metrics = model.val(
        data=data,
        imgsz=imgsz,
        split=split,
        device=resolve_device(device),
        batch=batch,
        project=str(RUNS_DIR / "val"),
    )

    results = {k: float(v) for k, v in metrics.results_dict.items() if isinstance(v, (int, float))}
    per_class = {
        model.names[int(cid)]: float(ap)
        for cid, ap in zip(metrics.ap_class_index, metrics.box.ap50, strict=False)
    }
    return {
        "weights": str(weights),
        "data": data,
        "imgsz": imgsz,
        "split": split,
        "metrics": results,
        "ap50_per_class": per_class,
    }


def write_comparison(results: list[dict[str, Any]], out_dir: Path | None = None) -> Path:
    """Render a markdown comparison table across evaluated checkpoints."""
    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Results",
        "",
        "| run | imgsz | mAP50-95 | mAP50 | precision | recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        m = r["metrics"]
        label = Path(r["weights"]).parent.parent.name
        lines.append(
            f"| {label} | {r['imgsz']} "
            f"| {m.get('metrics/mAP50-95(B)', 0):.4f} "
            f"| {m.get('metrics/mAP50(B)', 0):.4f} "
            f"| {m.get('metrics/precision(B)', 0):.4f} "
            f"| {m.get('metrics/recall(B)', 0):.4f} |"
        )

    report = out_dir / "results.md"
    report.write_text("\n".join(lines) + "\n")
    return report

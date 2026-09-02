"""Training entry point."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, load_config
from .hardware import detect
from .paths import RUNS_DIR, configure_ultralytics
from .tracking import configure as configure_tracking


def train(
    config: str | ExperimentConfig,
    profile: str | None = None,
    tracker: str | None = None,
) -> dict[str, Any]:
    """Fine-tune a YOLO model according to an experiment config.

    ``profile`` overlays a hardware profile (or ``"auto"`` to detect one), so
    the same experiment runs unchanged on a laptop or a rented GPU. The
    resolved config is written next to the run's weights so that a result in
    ``runs/`` can always be traced back to the exact settings that made it --
    including which machine profile produced it.
    """
    from ultralytics import YOLO

    configure_ultralytics()
    cfg = load_config(config, profile=profile) if isinstance(config, str) else config
    if tracker is not None:
        cfg.tracker = tracker

    hw = detect()
    active = configure_tracking(cfg.tracker, run_name=cfg.name)
    print(f"[aerialdet] hardware: {hw.describe()}")
    print(f"[aerialdet] profile : {cfg.profile or '(none, using config defaults)'}")
    print(f"[aerialdet] tracking: {active}")

    model = YOLO(cfg.model)
    if cfg.pretrained:
        # Matching layers transfer by shape and name; the rest stay random.
        print(f"[aerialdet] transferring weights from {cfg.pretrained}")
        model = model.load(cfg.pretrained)
    kwargs = cfg.to_train_kwargs()
    kwargs.setdefault("project", str(RUNS_DIR / "train"))

    print(f"[aerialdet] training '{cfg.name}' on {kwargs['device']} ({cfg.model})")

    # Write the resolved config *before* training, not after. Written after, a
    # failure in any post-training step -- a W&B artifact upload, say -- loses
    # the traceability record for a run whose weights and metrics are all
    # perfectly fine. That happened: two 50-epoch runs finished, then raised
    # during upload, and were left unidentifiable.
    save_dir = Path(kwargs["project"]) / cfg.name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "aerialdet_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    results = model.train(**kwargs)

    # Ultralytics increments the directory if it already existed, so re-record
    # under wherever the run actually landed.
    actual = Path(results.save_dir)
    if actual != save_dir:
        (actual / "aerialdet_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    save_dir = actual

    return {
        "name": cfg.name,
        "save_dir": str(save_dir),
        "best_weights": str(save_dir / "weights" / "best.pt"),
    }

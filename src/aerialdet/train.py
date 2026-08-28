"""Training entry point."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, load_config
from .paths import RUNS_DIR, configure_ultralytics


def train(config: str | ExperimentConfig) -> dict[str, Any]:
    """Fine-tune a YOLO model according to an experiment config.

    The resolved config is written next to the run's weights so that a result
    in ``runs/`` can always be traced back to the exact settings that made it.
    """
    from ultralytics import YOLO

    configure_ultralytics()
    cfg = load_config(config) if isinstance(config, str) else config

    model = YOLO(cfg.model)
    kwargs = cfg.to_train_kwargs()
    kwargs.setdefault("project", str(RUNS_DIR / "train"))

    print(f"[aerialdet] training '{cfg.name}' on {kwargs['device']} ({cfg.model})")
    results = model.train(**kwargs)

    save_dir = Path(results.save_dir)
    (save_dir / "aerialdet_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    return {
        "name": cfg.name,
        "save_dir": str(save_dir),
        "best_weights": str(save_dir / "weights" / "best.pt"),
    }

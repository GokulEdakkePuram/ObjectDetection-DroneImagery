"""Experiment configuration.

A run is fully described by one YAML file in ``configs/``. Configs compose via
``extends:`` so that a family of experiments differs only by the lines that
actually matter -- which is what makes an ablation table honest.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR

_MAX_EXTENDS_DEPTH = 10


@dataclass
class ExperimentConfig:
    """One training/evaluation experiment."""

    name: str
    model: str = "yolo11s.pt"
    data: str = "VisDrone.yaml"
    epochs: int = 50
    imgsz: int = 960
    batch: int = 8
    device: str = "auto"
    seed: int = 0
    patience: int = 20
    workers: int = 8
    notes: str = ""
    # Anything Ultralytics accepts (lr0, mosaic, scale, freeze, ...) passes
    # through untouched. Keeping it in one place means our dataclass never has
    # to chase the upstream hyperparameter list.
    train_args: dict[str, Any] = field(default_factory=dict)

    def to_train_kwargs(self) -> dict[str, Any]:
        """Flatten into the keyword arguments ``YOLO.train`` expects."""
        kwargs: dict[str, Any] = {
            "data": self.data,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": resolve_device(self.device),
            "seed": self.seed,
            "patience": self.patience,
            "workers": self.workers,
            "name": self.name,
        }
        kwargs.update(self.train_args)
        return kwargs


def resolve_device(device: str = "auto") -> str:
    """Pick the best available torch device.

    ``auto`` prefers CUDA, then Apple Silicon MPS, then CPU. An explicit value
    is returned unchanged so a config can force CPU for a deterministic test.
    """
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_path(name_or_path: str | Path) -> Path:
    """Accept a config name (``baseline``), a filename, or a full path."""
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return path
    candidate = CONFIG_DIR / f"{path.stem}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No config named {name_or_path!r}. Looked for {path} and {candidate}.")


def _load_raw(name_or_path: str | Path, _depth: int = 0) -> dict[str, Any]:
    """Load a config dict, applying ``extends:`` inheritance depth-first."""
    if _depth > _MAX_EXTENDS_DEPTH:
        raise ValueError(f"'extends' chain deeper than {_MAX_EXTENDS_DEPTH}; likely a cycle.")

    path = _resolve_path(name_or_path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(raw).__name__}.")

    parent_name = raw.pop("extends", None)
    if parent_name is None:
        return raw
    return _deep_merge(_load_raw(parent_name, _depth + 1), raw)


def load_config(name_or_path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config by name or path."""
    raw = _load_raw(name_or_path)

    known = {f.name for f in fields(ExperimentConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown config keys {sorted(unknown)}. "
            f"Ultralytics hyperparameters belong under 'train_args:'."
        )
    if "name" not in raw:
        raw["name"] = Path(name_or_path).stem

    return ExperimentConfig(**raw)

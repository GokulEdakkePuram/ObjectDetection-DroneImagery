"""Short calibration run before a long one.

Renting a GPU makes a wrong batch size expensive: an OOM or a 30-hour estimate
is much cheaper to discover in four minutes than three hours in. A probe runs
a handful of epochs on a fraction of the data, measures the steady-state epoch
time, and extrapolates.

The first epoch is always slower -- label caching, warmup, autobatch probing --
so the estimate uses the *fastest* epoch observed, which is the closest thing
to steady state a short run can give.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config

DEFAULT_EPOCHS = 3
DEFAULT_FRACTION = 0.05


@dataclass
class ProbeResult:
    """What a calibration run measured, and what it implies for the real one."""

    name: str
    profile: str
    imgsz: int
    batch: int
    fraction: float
    epoch_seconds: float
    full_epoch_seconds: float
    target_epochs: int
    peak_memory_gb: float | None

    @property
    def estimated_hours(self) -> float:
        return self.full_epoch_seconds * self.target_epochs / 3600

    def summary(self) -> str:
        mem = f"{self.peak_memory_gb:.1f} GB peak" if self.peak_memory_gb else "peak memory n/a"
        return (
            f"{self.name} [{self.profile or 'no profile'}] imgsz={self.imgsz} batch={self.batch}\n"
            f"  measured : {self.epoch_seconds:.1f}s/epoch on {self.fraction:.0%} of train\n"
            f"  projected: {self.full_epoch_seconds / 60:.1f} min/epoch on the full split\n"
            f"  {self.target_epochs} epochs -> {self.estimated_hours:.1f} h   ({mem})"
        )


def _steady_epoch_seconds(results_csv: Path) -> float:
    """Fastest per-epoch delta, as the best available proxy for steady state."""
    rows = list(csv.DictReader(results_csv.open()))
    if not rows:
        raise RuntimeError(f"{results_csv} has no rows; the probe run produced no epochs.")

    times = [float(r["time"]) for r in rows]
    deltas = [b - a for a, b in zip(times, times[1:], strict=False)]
    return min(deltas) if deltas else times[0]


def probe(
    config: str,
    profile: str | None = "auto",
    epochs: int = DEFAULT_EPOCHS,
    fraction: float = DEFAULT_FRACTION,
) -> ProbeResult:
    """Time a short run and project it onto the full training schedule."""
    import torch

    from .train import train

    cfg = load_config(config, profile=profile)
    target_epochs = cfg.epochs

    cfg.name = f"probe_{cfg.name}"
    cfg.epochs = epochs
    cfg.patience = 0
    cfg.train_args = {
        **cfg.train_args,
        "fraction": fraction,
        "plots": False,
        "val": False,  # validation cost is fixed per epoch; exclude it from the rate
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = train(cfg)

    epoch_seconds = _steady_epoch_seconds(Path(result["save_dir"]) / "results.csv")
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None

    return ProbeResult(
        name=config,
        profile=cfg.profile,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        fraction=fraction,
        epoch_seconds=epoch_seconds,
        full_epoch_seconds=epoch_seconds / fraction,
        target_epochs=target_epochs,
        peak_memory_gb=peak,
    )


def probe_all(configs: list[str], **kwargs: Any) -> list[ProbeResult]:
    """Probe several configs and report what the whole sweep would cost."""
    results = [probe(c, **kwargs) for c in configs]

    total = sum(r.estimated_hours for r in results)
    print("\n" + "=" * 64)
    for r in results:
        print(r.summary())
    print("-" * 64)
    print(f"  full sweep: {total:.1f} h")
    print("=" * 64)
    return results

"""Canonical project paths.

Everything the pipeline writes lands under the repo root so that a run is
reproducible from a fresh clone without touching global Ultralytics state.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_dirs() -> None:
    """Create the output directories that runs write into."""
    for d in (DATA_DIR, RUNS_DIR, WEIGHTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def configure_ultralytics() -> None:
    """Redirect Ultralytics' global dirs into the repo.

    Ultralytics persists ``datasets_dir``/``runs_dir`` in a user-level settings
    file. Left alone it scatters artifacts across the home directory, which
    makes results impossible to reproduce or clean up. We pin them per-process.
    """
    from ultralytics.utils import SETTINGS

    ensure_dirs()
    SETTINGS.update(
        {
            "datasets_dir": str(DATA_DIR),
            "runs_dir": str(RUNS_DIR),
            "weights_dir": str(WEIGHTS_DIR),
        }
    )

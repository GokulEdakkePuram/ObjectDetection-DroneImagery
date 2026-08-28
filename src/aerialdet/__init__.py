"""Fine-tuning YOLO for small-object detection in aerial imagery."""

from .config import ExperimentConfig, load_config, resolve_device
from .tiling import Tile, compute_tiles, merge_detections, predict_tiled

__version__ = "0.1.0"

__all__ = [
    "ExperimentConfig",
    "Tile",
    "compute_tiles",
    "load_config",
    "merge_detections",
    "predict_tiled",
    "resolve_device",
]

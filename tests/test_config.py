"""Config loading, inheritance and validation."""

from __future__ import annotations

import pytest

from aerialdet.config import load_config, resolve_device


def test_baseline_and_finetune_differ_only_in_resolution():
    """The ablation claim in the README depends on this staying true."""
    baseline = load_config("baseline_640")
    finetune = load_config("finetune_960")

    assert baseline.model == finetune.model
    assert baseline.epochs == finetune.epochs
    assert baseline.train_args == finetune.train_args
    assert (baseline.imgsz, finetune.imgsz) == (640, 960)


def test_extends_deep_merges_train_args():
    """A child overriding one hyperparameter must not drop its siblings."""
    smoke = load_config("smoke")

    assert smoke.train_args["fraction"] == 0.02  # set by the child
    assert smoke.train_args["plots"] is False  # overridden
    assert smoke.train_args["mosaic"] == 1.0  # inherited from base


def test_unknown_top_level_key_is_rejected(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("name: bad\nlr0: 0.01\n")  # lr0 belongs under train_args

    with pytest.raises(ValueError, match="train_args"):
        load_config(cfg)


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")


def test_to_train_kwargs_resolves_device_and_flattens():
    kwargs = load_config("finetune_960").to_train_kwargs()

    assert kwargs["imgsz"] == 960
    assert kwargs["device"] in {"cpu", "mps", "0"}
    assert kwargs["mosaic"] == 1.0  # train_args flattened in


def test_explicit_device_passes_through():
    assert resolve_device("cpu") == "cpu"

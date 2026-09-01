"""Hardware profiles and their interaction with experiment configs."""

from __future__ import annotations

import pytest

from aerialdet.config import load_config, load_profile
from aerialdet.hardware import CPU, CUDA_LARGE, CUDA_MEDIUM, CUDA_SMALL, Hardware, auto_profile


def test_profile_overrides_experiment_batch():
    """The machine's limits win over whatever the experiment asked for."""
    laptop = load_config("finetune_1280", profile="mps")
    rented = load_config("finetune_1280", profile="cuda24")

    assert laptop.batch == 2
    assert rented.batch == 8
    assert laptop.imgsz == rented.imgsz == 1280  # the experiment is unchanged


def test_profile_gives_one_batch_across_the_whole_sweep():
    """The point of profiles: a constant batch removes the ablation's confound."""
    batches = {
        load_config(name, profile="cuda24").batch
        for name in ("baseline_640", "finetune_960", "finetune_1280")
    }
    assert batches == {8}, "resolution sweep must not vary batch size on one machine"


def test_profile_is_recorded_on_the_config():
    assert load_config("finetune_960", profile="cuda48").profile == "cuda48"


def test_config_without_profile_keeps_its_own_batch():
    assert load_config("finetune_960").batch == 4


@pytest.mark.parametrize("name", ["mps", "cpu", "cuda12", "cuda24", "cuda48"])
def test_every_profile_sets_only_hardware_keys(name):
    """A profile that could set epochs or imgsz would make two runs with the
    same label mean different things."""
    assert set(load_profile(name)) <= {"batch", "workers", "device", "train_args"}


def test_unknown_profile_lists_the_alternatives():
    with pytest.raises(FileNotFoundError, match="cuda24"):
        load_profile("cuda9000")


@pytest.mark.parametrize(
    ("hw", "expected"),
    [
        (Hardware("cpu", "CPU", 32), CPU),
        (Hardware("0", "RTX 4070", 12), CUDA_SMALL),
        (Hardware("0", "RTX 4090", 24), CUDA_MEDIUM),
        (Hardware("0", "A6000", 48), CUDA_LARGE),
        (Hardware("0", "A100", 80), CUDA_LARGE),
    ],
)
def test_auto_profile_matches_hardware(hw, expected):
    assert auto_profile(hw) == expected


def test_auto_profile_is_conservative_at_boundaries():
    """A card just under a threshold must not get the bigger profile: an OOM
    three hours into a rented run costs more than unused memory."""
    assert auto_profile(Hardware("0", "card", 19.9)) == CUDA_SMALL
    assert auto_profile(Hardware("0", "card", 39.9)) == CUDA_MEDIUM

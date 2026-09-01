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


class TestProbeFit:
    """The projection maths, which is where the first probe went wrong.

    Timing a run on 5% of the data and dividing by 0.05 assumes epoch time is
    proportional to dataset size. It is not -- there is a fixed per-epoch
    overhead, and that naive scaling multiplies it by twenty.
    """

    def test_recovers_known_overhead_and_rate(self):
        from aerialdet.probe import _fit

        # Ground truth: 4s overhead, 10ms per image.
        overhead, rate = _fit([(323, 4.0 + 0.010 * 323), (970, 4.0 + 0.010 * 970)])

        assert overhead == pytest.approx(4.0, abs=1e-6)
        assert rate == pytest.approx(0.010, abs=1e-9)

    def test_naive_scaling_would_have_overestimated(self):
        """Reproduces the real 4090 measurement: 6.9s/epoch at 5%."""
        from aerialdet.probe import _fit

        overhead, rate = _fit([(323, 6.9), (970, 6.9 + 0.006 * 647)])
        fitted_full = overhead + rate * 6471
        naive_full = 6.9 / 0.05

        assert naive_full > fitted_full
        # The error is (1/fraction - 1) x overhead -- 19x here, not a rounding issue.
        assert naive_full - fitted_full == pytest.approx(19 * overhead, rel=0.01)

    def test_indistinguishable_runs_raise_instead_of_extrapolating(self):
        """Two runs lost in the noise must not yield a confident wrong number."""
        from aerialdet.probe import _fit

        with pytest.raises(RuntimeError, match="too short to measure"):
            _fit([(323, 7.0), (970, 6.8)])

    def test_identical_fractions_rejected(self):
        from aerialdet.probe import _fit

        with pytest.raises(ValueError, match="must differ"):
            _fit([(323, 7.0), (323, 7.2)])

from __future__ import annotations

from pathlib import Path

import pytest

from utils.experiment_layout import (
    make_debug_output_dir,
    make_eval_output_dir,
    make_run_name,
    make_training_output_dir,
    trial_dir_name,
)


def test_debug_output_path_contains_phase_runtime_task_trial_and_seed():
    path = make_debug_output_dir(
        "outputs/debug/mappo_direct_specialists",
        "phase01_action_scale",
        "20260607_1430",
        "area_coverage",
        1,
        0,
    )
    assert path == Path(
        "outputs/debug/mappo_direct_specialists/"
        "phase01_action_scale/20260607_1430/area_coverage/trial01/s0"
    )


def test_training_output_path_without_and_with_trial():
    root = "outputs/training/mappo_direct_specialists"
    assert make_training_output_dir(root, "20260607_2200", "belief_search", 1) == Path(
        "outputs/training/mappo_direct_specialists/20260607_2200/belief_search/s1"
    )
    assert make_training_output_dir(root, "20260607_2200", "belief_search", 1, trial=2) == Path(
        "outputs/training/mappo_direct_specialists/20260607_2200/belief_search/trial02/s1"
    )


def test_eval_output_path_matches_training_layout():
    root = "outputs/eval/mappo_direct_specialists"
    assert make_eval_output_dir(root, "20260611_1430", "area_coverage", 0) == Path(
        "outputs/eval/mappo_direct_specialists/20260611_1430/area_coverage/s0"
    )
    assert make_eval_output_dir(root, "20260611_1430", "area_coverage", 0, trial=2) == Path(
        "outputs/eval/mappo_direct_specialists/20260611_1430/area_coverage/trial02/s0"
    )


@pytest.mark.parametrize("phase_name", ["phase1_action", "phase001_action", "action_scale", "phase01-foo"])
def test_invalid_phase_name_raises(phase_name: str):
    with pytest.raises(ValueError, match="phaseXX"):
        make_debug_output_dir("outputs/debug", phase_name, "20260607_1430", "area_coverage", 1, 0)


def test_run_name_is_short_and_seed_aligned():
    assert make_run_name(seed=0) == "s0"
    assert make_run_name(seed=2, trial=1) == "trial01_s2"
    assert "updates" not in make_run_name(seed=2, trial=1)


def test_trial_format_is_two_digits():
    assert trial_dir_name(1) == "trial01"
    assert trial_dir_name(12) == "trial12"
    with pytest.raises(ValueError):
        trial_dir_name(0)

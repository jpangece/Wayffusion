from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import torch
import yaml
import pytest

from algorithms.mappo_waypoint import MAPPOWaypointTrainer
from utils.tensorboard_metrics import should_log_tensorboard_metric
from utils.waypoint_evaluation import (
    env_config_for_randomization_mode,
    resolve_eval_randomization_mode,
)


def test_eval_yaml_randomization_mode_is_used():
    with open("configs/eval/waypoint_eval.yaml", "r", encoding="utf-8") as handle:
        eval_config = yaml.safe_load(handle)
    assert resolve_eval_randomization_mode(eval_config) == "both"
    assert eval_config["eval_episodes"] == 10
    assert eval_config["record_eval_episodes"] == 2


def test_cli_randomization_mode_overrides_eval_yaml():
    eval_config = {"randomization_mode": "fixed"}
    assert resolve_eval_randomization_mode(eval_config, "random") == "random"
    assert resolve_eval_randomization_mode({}, None) == "both"


def test_fixed_and_random_eval_configs_do_not_mutate_training_config():
    train_env_config = {
        "sensor_radius": 0.12,
        "domain_randomization": {"enabled": True, "spawn": {"max_radius": 0.25}},
    }
    original = deepcopy(train_env_config)
    fixed = env_config_for_randomization_mode(train_env_config, "fixed")
    random = env_config_for_randomization_mode(train_env_config, "random")

    assert fixed["domain_randomization"]["enabled"] is False
    assert random["domain_randomization"]["enabled"] is True
    assert train_env_config == original
    assert fixed["domain_randomization"]["spawn"] == original["domain_randomization"]["spawn"]


def test_both_eval_produces_prefixed_metrics_and_separate_media(tmp_path: Path, monkeypatch):
    calls = []

    def fake_evaluate(
        env_config,
        policy,
        task_names,
        episodes,
        device,
        headless=True,
        record_dir=None,
        record_episodes=0,
        record_format="gif",
        record_fps=8,
        record_prefix="eval",
    ):
        del policy, episodes, device, headless, record_episodes, record_format, record_fps, record_prefix
        enabled = bool(env_config["domain_randomization"]["enabled"])
        success = 0.25 if enabled else 0.75
        reward = 2.0 if enabled else 5.0
        calls.append((enabled, Path(record_dir)))
        Path(record_dir).mkdir(parents=True, exist_ok=True)
        records = [{"return": reward, "success_rate": success, "task_name": task_names[0]}]
        task_summary = {"return_mean": reward, "success_rate_mean": success, "coverage_ratio_mean": success}
        overall = {
            "return_mean": reward,
            "success_rate_mean": success,
            "path_length_mean": 1.0,
            "collision_rate_mean": 0.0,
            "action_validity_rate_mean": 1.0,
        }
        return records, {task_names[0]: task_summary}, overall

    monkeypatch.setattr("utils.waypoint_evaluation.evaluate_waypoint_policy_per_task", fake_evaluate)
    trainer = object.__new__(MAPPOWaypointTrainer)
    trainer.policy = object()
    trainer.device = torch.device("cpu")
    train_env_config = {"domain_randomization": {"enabled": True}}
    original = deepcopy(train_env_config)

    metrics = trainer.evaluate(
        train_env_config,
        ["area_coverage"],
        episodes=1,
        output_dir=tmp_path,
        update_idx=3,
        record_episodes=1,
        randomization_mode="both",
    )

    assert metrics["eval_fixed_success_rate"] == 0.75
    assert metrics["eval_random_success_rate"] == 0.25
    assert metrics["eval_success_rate"] == 0.25
    assert metrics["eval_generalization_gap"] == 0.5
    assert metrics["eval_reward_generalization_gap"] == 3.0
    assert calls[0][1] == tmp_path / "media" / "eval_fixed" / "eval_0003"
    assert calls[1][1] == tmp_path / "media" / "eval_random" / "eval_0003"
    assert train_env_config == original

    with open(tmp_path / "eval_metrics.csv", "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["eval_randomization_mode"] for row in rows} == {"fixed", "random"}

    assert should_log_tensorboard_metric("eval_fixed_success_rate", 0.75, mode="core")
    assert should_log_tensorboard_metric("eval_random_reward", 2.0, mode="core")


def test_goal_split_eval_produces_seen_interpolation_and_formal_metrics(tmp_path: Path, monkeypatch):
    calls = []

    def fake_evaluate(
        env_config,
        policy,
        task_names,
        episodes,
        device,
        headless=True,
        record_dir=None,
        record_episodes=0,
        record_format="gif",
        record_fps=8,
        record_prefix="eval",
        goal_split=None,
    ):
        del env_config, policy, episodes, device, headless, record_episodes, record_format, record_fps, record_prefix
        values = {"seen": 0.8, "interpolation": 0.6, "formal": 0.4}
        success = values[goal_split]
        calls.append((goal_split, Path(record_dir)))
        Path(record_dir).mkdir(parents=True, exist_ok=True)
        records = [{"return": success, "success_rate": success, "task_name": task_names[0]}]
        summary = {"return_mean": success, "success_rate_mean": success, "goal_progress_mean": success}
        return records, {task_names[0]: summary}, summary

    monkeypatch.setattr("utils.waypoint_evaluation.evaluate_waypoint_policy_per_task", fake_evaluate)
    trainer = object.__new__(MAPPOWaypointTrainer)
    trainer.policy = object()
    trainer.device = torch.device("cpu")
    metrics = trainer.evaluate(
        {"domain_randomization": {"enabled": True}},
        ["area_coverage"],
        episodes=1,
        output_dir=tmp_path,
        update_idx=5,
        record_episodes=1,
        randomization_mode="random",
        goal_splits=["seen", "interpolation", "formal"],
    )

    assert metrics["eval_random_goal_seen_success_rate"] == 0.8
    assert metrics["eval_random_goal_interpolation_success_rate"] == 0.6
    assert metrics["eval_random_goal_formal_success_rate"] == 0.4
    assert metrics["eval_goal_interpolation_gap"] == pytest.approx(0.2)
    assert metrics["eval_goal_formal_gap"] == pytest.approx(0.4)
    assert metrics["eval_success_rate"] == 0.4
    assert calls[0][1] == tmp_path / "media" / "eval_random" / "eval_0005" / "goal_seen"

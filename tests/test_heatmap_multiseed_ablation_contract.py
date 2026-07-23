from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "configs/experiments"
CONDITIONS = ("off", "real", "zero")
TRAINING_SEEDS = range(5)
FUTURE_TRAINING_SEEDS = range(1, 5)
EVALUATION_BLOCKS = {
    seed: list(range(10000 + seed * 1000, 10008 + seed * 1000))
    for seed in TRAINING_SEEDS
}


def _config_path(seed: int, condition: str) -> Path:
    return CONFIG_DIR / (
        f"priority_inspection_heatmap_{condition}_ablation_seed{seed}.yaml"
    )


def _load_config(seed: int, condition: str) -> dict:
    path = _config_path(seed, condition)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert isinstance(config, dict), f"config must be a mapping: {path}"
    return config


def _normalize_condition(config: dict, *, include_activation: bool) -> dict:
    normalized = deepcopy(config)
    normalized["experiment_name"] = "<experiment>"
    provider = normalized["environment"]["heatmap_observation"]["provider"]
    provider["source"] = "<provider>"
    if not include_activation:
        normalized["environment"]["heatmap_observation"]["enabled"] = False
        provider["enabled"] = False
        provider["refresh_on_step"] = False
        normalized["policy"]["use_task_element_heatmap"] = False
    return normalized


def _normalize_seed(config: dict) -> dict:
    normalized = deepcopy(config)
    normalized["experiment_name"] = "<experiment>"
    normalized["seed"] = "<seed>"
    normalized["environment"]["seed"] = "<seed>"
    normalized["training"]["policy_init_seed"] = "<seed>"
    normalized["training"]["evaluation_episode_seeds"] = "<evaluation-block>"
    return normalized


@pytest.mark.parametrize("seed", FUTURE_TRAINING_SEEDS)
def test_expected_three_condition_configs_exist_for_each_future_seed(seed):
    missing = [
        path.relative_to(ROOT)
        for condition in CONDITIONS
        if not (path := _config_path(seed, condition)).is_file()
    ]
    assert not missing, f"seed {seed} missing config(s): " + ", ".join(map(str, missing))


def test_completed_seed_zero_conforms_to_multiseed_protocol():
    configs = {condition: _load_config(0, condition) for condition in CONDITIONS}
    _assert_seed_contract(0, configs)
    _assert_condition_equality(configs)


def test_evaluation_blocks_are_ordered_unique_and_pairwise_disjoint():
    observed: set[int] = set()
    for seed, block in EVALUATION_BLOCKS.items():
        assert len(block) == 8
        assert all(type(value) is int for value in block)
        assert block == sorted(block)
        assert len(set(block)) == len(block)
        assert not observed.intersection(block), f"seed {seed} evaluation block overlaps"
        observed.update(block)


@pytest.mark.parametrize("seed", FUTURE_TRAINING_SEEDS)
def test_future_seed_configs_match_seed_and_evaluation_protocol(seed):
    paths = [_config_path(seed, condition) for condition in CONDITIONS]
    if not all(path.is_file() for path in paths):
        pytest.skip(f"seed {seed} configs are not implemented yet")
    configs = {condition: _load_config(seed, condition) for condition in CONDITIONS}
    _assert_seed_contract(seed, configs)
    _assert_condition_equality(configs)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_corresponding_conditions_differ_only_by_seed_domains(condition):
    paths = [_config_path(seed, condition) for seed in TRAINING_SEEDS]
    if not all(path.is_file() for path in paths):
        pytest.skip(f"future {condition.upper()} configs are not implemented yet")
    baseline = _normalize_seed(_load_config(0, condition))
    for seed in FUTURE_TRAINING_SEEDS:
        assert _normalize_seed(_load_config(seed, condition)) == baseline


def _assert_seed_contract(seed: int, configs: dict[str, dict]) -> None:
    expected_block = EVALUATION_BLOCKS[seed]
    expected_activation = {
        "off": (False, False, "none", False, False),
        "real": (True, True, "priority_inspection", True, True),
        "zero": (True, True, "zero", True, True),
    }
    expected_policy = {
        "policy_class": "direct_waypoint",
        "cnn_channels": [8, 16, 32],
        "agent_hidden_dim": 32,
        "joint_hidden_dim": 64,
        "use_attention": False,
        "use_spatial_field_context": False,
        "use_field_moment_context": False,
        "use_connectivity_auxiliary_loss": False,
        "use_comm_graph_encoder": False,
        "use_uvfa_goal": False,
        "use_task_element_heatmap": None,
        "log_std_init": -1.0,
        "log_std_min": -2.0,
        "log_std_max": 0.2,
    }
    expected_training = {
        "env_backend": "sync",
        "num_envs": 2,
        "rollout_steps": 32,
        "total_updates": 200,
        "epochs": 2,
        "minibatch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "learning_rate": 0.0003,
        "max_grad_norm": 0.5,
        "reward_norm": False,
        "advantage_norm": True,
        "lr_schedule": "constant",
        "target_kl": 0.05,
        "init_checkpoint": None,
        "evaluation_enabled": True,
        "eval_interval": 20,
        "eval_episodes": 8,
        "record_eval_episodes": 0,
        "headless": True,
        "policy_init_seed": seed,
        "evaluation_episode_seeds": expected_block,
    }
    for condition, config in configs.items():
        expected_name = f"priority_inspection_heatmap_{condition}_ablation_seed{seed}"
        assert config["experiment_name"] == expected_name
        assert config["seed"] == seed
        environment = config["environment"]
        assert environment["task_name"] == "priority_inspection"
        assert environment["task_names"] == ["priority_inspection"]
        assert environment["seed"] == seed
        assert environment["num_agents"] == 4
        assert environment["grid_size"] == 16
        assert environment["map_size"] == 1.0
        assert environment["max_steps"] == 64
        assert environment["max_waypoint_distance"] == 0.20
        assert environment["domain_randomization"]["enabled"] is False
        assert environment["tasks"]["priority_inspection"] == {
            "num_pois": 8,
            "num_pois_range": None,
        }
        heatmap = environment["heatmap_observation"]
        provider = heatmap["provider"]
        actual_activation = (
            heatmap["enabled"],
            provider["enabled"],
            provider["source"],
            provider["refresh_on_step"],
            config["policy"]["use_task_element_heatmap"],
        )
        assert actual_activation == expected_activation[condition]
        assert heatmap["channels"] == 1
        policy = dict(expected_policy)
        policy["use_task_element_heatmap"] = condition != "off"
        assert config["policy"] == policy
        assert config["training"] == expected_training
        assert config["training"]["evaluation_episode_seeds"] is not expected_block


def _assert_condition_equality(configs: dict[str, dict]) -> None:
    assert _normalize_condition(
        configs["real"], include_activation=True
    ) == _normalize_condition(configs["zero"], include_activation=True)
    normalized_off = _normalize_condition(configs["off"], include_activation=False)
    for condition in ("real", "zero"):
        assert normalized_off == _normalize_condition(
            configs[condition], include_activation=False
        )


def _contrasts_by_training_seed(
    summaries: dict[int, dict[str, float]],
) -> dict[str, list[float]]:
    """Return one paired condition contrast per independent training seed."""
    contrasts = {"real_zero": [], "zero_off": [], "real_off": []}
    for seed in sorted(summaries):
        values = summaries[seed]
        contrasts["real_zero"].append(values["real"] - values["zero"])
        contrasts["zero_off"].append(values["zero"] - values["off"])
        contrasts["real_off"].append(values["real"] - values["off"])
    return contrasts


def test_aggregate_analysis_uses_one_contrast_per_training_seed():
    per_seed_summaries = {
        seed: {
            "off": float(seed),
            "real": float(seed + 2),
            "zero": float(seed + 1),
        }
        for seed in TRAINING_SEEDS
    }
    contrasts = _contrasts_by_training_seed(per_seed_summaries)
    assert contrasts["real_zero"] == [1.0] * 5
    assert contrasts["zero_off"] == [1.0] * 5
    assert contrasts["real_off"] == [2.0] * 5
    assert all(len(values) == len(TRAINING_SEEDS) for values in contrasts.values())
    # Checkpoint evaluations are repeated measurements within a training run;
    # means and dispersion belong across these per-seed contrasts, not across
    # the ten checkpoint evaluations as if they were independent training runs.

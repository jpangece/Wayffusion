from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[1]
ON_PATH = ROOT / "configs/experiments/priority_inspection_heatmap_on_debug.yaml"
OFF_PATH = ROOT / "configs/experiments/priority_inspection_heatmap_off_debug.yaml"
EXPERIMENT_PATHS = {"on": ON_PATH, "off": OFF_PATH}


def _missing_experiment_paths() -> list[Path]:
    return [path for path in EXPERIMENT_PATHS.values() if not path.is_file()]


def _load_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    assert isinstance(parsed, dict), f"experiment config must be a mapping: {path}"
    return parsed


@pytest.fixture
def experiment_configs() -> dict[str, dict]:
    missing = _missing_experiment_paths()
    if missing:
        pytest.skip("experiment configs are not implemented yet")
    return {name: _load_mapping(path) for name, path in EXPERIMENT_PATHS.items()}


def _value(config: dict, *path: str):
    current = config
    for key in path:
        assert isinstance(current, dict) and key in current, (
            f"missing configuration path: {'.'.join(path)}"
        )
        current = current[key]
    return current


def _normalized_activation(config: dict) -> dict:
    normalized = deepcopy(config)
    normalized["experiment_name"] = "<heatmap-variant>"
    heatmap = normalized["environment"]["heatmap_observation"]
    heatmap["enabled"] = False
    heatmap["provider"]["enabled"] = False
    heatmap["provider"]["refresh_on_step"] = False
    normalized["policy"]["use_task_element_heatmap"] = False
    return normalized


def _differing_paths(left, right, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(_differing_paths(left[key], right[key], path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                _differing_paths(left_item, right_item, f"{prefix}[{index}]")
            )
        return differences
    return [] if left == right else [prefix]


def _intended_launcher_configs(experiment: dict) -> tuple[dict, dict, dict]:
    env_config = deepcopy(experiment["environment"])
    train_config = deepcopy(experiment["policy"])
    train_config.update(deepcopy(experiment["training"]))
    env_config["seed"] = int(experiment["seed"])
    train_config["seed"] = int(experiment["seed"])
    launch = {
        "env_backend": train_config.pop("env_backend"),
        "init_checkpoint": train_config.pop("init_checkpoint"),
        "evaluation_enabled": train_config.pop("evaluation_enabled"),
    }
    return env_config, train_config, launch


def test_expected_experiment_configs_exist_and_are_parseable():
    missing = _missing_experiment_paths()
    assert not missing, "missing experiment config(s): " + ", ".join(
        str(path.relative_to(ROOT)) for path in missing
    )
    for path in EXPERIMENT_PATHS.values():
        _load_mapping(path)


def test_experiment_identity_and_shared_short_debug_contract(experiment_configs):
    on = experiment_configs["on"]
    off = experiment_configs["off"]
    assert on["experiment_name"] == "priority_inspection_heatmap_on_debug"
    assert off["experiment_name"] == "priority_inspection_heatmap_off_debug"
    assert on["seed"] == off["seed"]
    for config in (on, off):
        assert set(config) >= {"experiment_name", "seed", "environment", "policy", "training"}
        assert _value(config, "environment", "task_name") == "priority_inspection"
        assert _value(config, "environment", "task_names") == ["priority_inspection"]
        assert _value(config, "environment", "num_agents") == 2
        assert _value(config, "environment", "grid_size") == 8
        assert _value(config, "environment", "max_steps") == 16
        assert _value(config, "environment", "domain_randomization", "enabled") is False
        assert _value(config, "environment", "tasks", "priority_inspection", "num_pois") == 4
        assert _value(config, "environment", "tasks", "priority_inspection", "num_pois_range") is None
        assert _value(config, "policy", "policy_class") == "direct_waypoint"
        assert _value(config, "training", "env_backend") == "sync"
        assert _value(config, "training", "num_envs") == 1
        assert _value(config, "training", "rollout_steps") == 4
        assert _value(config, "training", "total_updates") == 2
        assert _value(config, "training", "epochs") == 1
        minibatch_size = _value(config, "training", "minibatch_size")
        assert 1 <= minibatch_size <= 4
        assert 4 % minibatch_size == 0
        assert _value(config, "training", "evaluation_enabled") is False
        assert _value(config, "training", "init_checkpoint") is None

    shared_environment_paths = [
        ("map_size",),
        ("max_waypoint_distance",),
        ("seed",),
        ("tasks", "priority_inspection", "num_pois"),
        ("tasks", "priority_inspection", "num_pois_range"),
    ]
    for path in shared_environment_paths:
        assert _value(on, "environment", *path) == _value(off, "environment", *path)

    shared_policy_keys = [
        "policy_class",
        "cnn_channels",
        "agent_hidden_dim",
        "joint_hidden_dim",
        "use_attention",
        "use_spatial_field_context",
        "use_field_moment_context",
        "use_connectivity_auxiliary_loss",
        "use_comm_graph_encoder",
        "use_uvfa_goal",
        "log_std_init",
        "log_std_min",
        "log_std_max",
    ]
    for key in shared_policy_keys:
        assert _value(on, "policy", key) == _value(off, "policy", key)

    shared_training_keys = [
        "rollout_steps",
        "total_updates",
        "epochs",
        "minibatch_size",
        "learning_rate",
        "reward_norm",
        "advantage_norm",
        "lr_schedule",
        "evaluation_enabled",
        "init_checkpoint",
        "env_backend",
        "num_envs",
    ]
    for key in shared_training_keys:
        assert _value(on, "training", key) == _value(off, "training", key)


def test_only_semantic_heatmap_activation_differs(experiment_configs):
    on = experiment_configs["on"]
    off = experiment_configs["off"]
    assert _value(on, "environment", "heatmap_observation", "enabled") is True
    assert _value(on, "environment", "heatmap_observation", "channels") == 1
    assert _value(on, "environment", "heatmap_observation", "provider", "enabled") is True
    assert _value(on, "environment", "heatmap_observation", "provider", "refresh_on_step") is True
    assert _value(on, "policy", "use_task_element_heatmap") is True
    assert _value(off, "environment", "heatmap_observation", "enabled") is False
    assert _value(off, "environment", "heatmap_observation", "provider", "enabled") is False
    assert _value(off, "environment", "heatmap_observation", "provider", "refresh_on_step") is False
    assert _value(off, "policy", "use_task_element_heatmap") is False

    normalized_on = _normalized_activation(on)
    normalized_off = _normalized_activation(off)
    differences = _differing_paths(normalized_on, normalized_off)
    assert not differences, "unexpected ON/OFF differences: " + ", ".join(differences)


def test_combined_config_maps_to_short_launcher_inputs(experiment_configs):
    for variant, experiment in experiment_configs.items():
        env_config, train_config, launch = _intended_launcher_configs(experiment)
        enabled = variant == "on"
        assert env_config["task_name"] == "priority_inspection"
        assert env_config["task_names"] == ["priority_inspection"]
        assert env_config["heatmap_observation"]["enabled"] is enabled
        assert env_config["heatmap_observation"]["provider"]["enabled"] is enabled
        assert env_config["heatmap_observation"]["provider"]["refresh_on_step"] is enabled
        assert train_config["use_task_element_heatmap"] is enabled
        assert train_config["rollout_steps"] == 4
        assert train_config["total_updates"] == 2
        assert launch == {
            "env_backend": "sync",
            "init_checkpoint": None,
            "evaluation_enabled": False,
        }


def test_launcher_exposes_combined_experiment_config_hook(experiment_configs):
    del experiment_configs
    launcher_path = ROOT / "scripts/train_mappo_waypoint.py"
    source = launcher_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "load_experiment_config" in function_names, (
        "scripts/train_mappo_waypoint.py must expose load_experiment_config"
    )
    assert "--experiment-config" in string_literals, (
        "train_mappo_waypoint must accept --experiment-config"
    )


def test_default_configs_remain_heatmap_disabled():
    env_default = _load_mapping(ROOT / "configs/env/waypoint_missions.yaml")
    policy_default = _load_mapping(ROOT / "configs/policy/direct_waypoint_debug.yaml")
    assert not bool(env_default.get("heatmap_observation", {}).get("enabled", False))
    assert not bool(policy_default.get("use_task_element_heatmap", False))

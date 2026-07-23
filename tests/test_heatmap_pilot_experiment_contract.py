from __future__ import annotations

import ast
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[1]
LAUNCHER_PATH = ROOT / "scripts/train_mappo_waypoint.py"
TRAINER_PATH = ROOT / "algorithms/mappo_waypoint.py"
OFF_PATH = ROOT / "configs/experiments/priority_inspection_heatmap_off_pilot_seed0.yaml"
ON_PATH = ROOT / "configs/experiments/priority_inspection_heatmap_on_pilot_seed0.yaml"
PILOT_PATHS = {"off": OFF_PATH, "on": ON_PATH}

TRAINING_METRICS = {
    "update",
    "mean_rollout_reward",
    "weighted_poi_completion",
    "goal_achieved",
    "arrival_rate",
    "policy_loss",
    "value_loss",
    "entropy",
    "grad_norm",
}
EVALUATION_METRICS = {"eval_success_rate", "eval_reward"}


def _load_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    assert isinstance(parsed, dict), f"pilot config must be a mapping: {path}"
    return parsed


@pytest.fixture
def pilot_configs() -> dict[str, dict]:
    missing = [path for path in PILOT_PATHS.values() if not path.is_file()]
    if missing:
        pytest.skip("pilot experiment configs are not implemented yet")
    return {name: _load_mapping(path) for name, path in PILOT_PATHS.items()}


def _value(config: dict, *path: str):
    current = config
    for key in path:
        assert isinstance(current, dict) and key in current, (
            f"missing pilot configuration path: {'.'.join(path)}"
        )
        current = current[key]
    return current


def _normalized_activation(config: dict) -> dict:
    normalized = deepcopy(config)
    normalized["experiment_name"] = "<paired-pilot>"
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


def _load_source_functions(path: Path, names: list[str]) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = sorted(set(names).difference(functions))
    if missing:
        pytest.fail(
            f"{path.relative_to(ROOT)} missing public helper(s): "
            + ", ".join(missing)
        )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *(functions[name] for name in names),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "ROOT": ROOT,
        "Path": Path,
        "deepcopy": deepcopy,
        "yaml": yaml,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_expected_pilot_configs_exist_and_are_parseable():
    missing = [path for path in PILOT_PATHS.values() if not path.is_file()]
    assert not missing, "missing pilot experiment config(s): " + ", ".join(
        str(path.relative_to(ROOT)) for path in missing
    )
    for path in PILOT_PATHS.values():
        _load_mapping(path)


def test_pilot_identity_environment_and_policy_contract(pilot_configs):
    assert pilot_configs["off"]["experiment_name"] == (
        "priority_inspection_heatmap_off_pilot_seed0"
    )
    assert pilot_configs["on"]["experiment_name"] == (
        "priority_inspection_heatmap_on_pilot_seed0"
    )
    for config in pilot_configs.values():
        assert set(config) >= {
            "experiment_name",
            "seed",
            "environment",
            "policy",
            "training",
        }
        assert config["seed"] == 0
        environment = config["environment"]
        assert environment["task_name"] == "priority_inspection"
        assert environment["task_names"] == ["priority_inspection"]
        assert environment["seed"] == 0
        assert environment["num_agents"] == 4
        assert environment["grid_size"] == 16
        assert environment["map_size"] == 1.0
        assert environment["max_steps"] == 64
        assert environment["max_waypoint_distance"] == 0.20
        assert environment["domain_randomization"]["enabled"] is False
        assert environment["tasks"]["priority_inspection"]["num_pois"] == 8
        assert environment["tasks"]["priority_inspection"]["num_pois_range"] is None
        assert config["policy"] == {
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
            "use_task_element_heatmap": config["policy"]["use_task_element_heatmap"],
            "log_std_init": -1.0,
            "log_std_min": -2.0,
            "log_std_max": 0.2,
        }


def test_pilot_budget_evaluation_and_runtime_limits(pilot_configs):
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
    }
    for config in pilot_configs.values():
        training = config["training"]
        assert training == expected_training
        assert training["rollout_steps"] * training["num_envs"] == 64
        assert training["minibatch_size"] == 64
        assert training["total_updates"] <= 200
        assert training["rollout_steps"] <= 32
        assert training["num_envs"] <= 2
        assert training["eval_episodes"] <= 8
        assert training["record_eval_episodes"] <= 0


def test_only_heatmap_activation_differs(pilot_configs):
    off = pilot_configs["off"]
    on = pilot_configs["on"]
    assert _value(off, "environment", "heatmap_observation", "enabled") is False
    assert _value(off, "environment", "heatmap_observation", "channels") == 1
    assert _value(off, "environment", "heatmap_observation", "provider", "enabled") is False
    assert _value(off, "environment", "heatmap_observation", "provider", "refresh_on_step") is False
    assert _value(off, "policy", "use_task_element_heatmap") is False
    assert _value(on, "environment", "heatmap_observation", "enabled") is True
    assert _value(on, "environment", "heatmap_observation", "channels") == 1
    assert _value(on, "environment", "heatmap_observation", "provider", "enabled") is True
    assert _value(on, "environment", "heatmap_observation", "provider", "refresh_on_step") is True
    assert _value(on, "policy", "use_task_element_heatmap") is True
    differences = _differing_paths(
        _normalized_activation(off), _normalized_activation(on)
    )
    assert not differences, "unexpected pilot ON/OFF differences: " + ", ".join(
        differences
    )


def test_existing_combined_loader_resolves_pilot_configs(pilot_configs):
    del pilot_configs
    helpers = _load_source_functions(
        LAUNCHER_PATH, ["load_yaml", "load_experiment_config"]
    )
    load_experiment_config = helpers["load_experiment_config"]
    for variant, path in PILOT_PATHS.items():
        env_config, train_config, launch_config = load_experiment_config(
            path.relative_to(ROOT)
        )
        enabled = variant == "on"
        assert env_config["task_names"] == ["priority_inspection"]
        assert env_config["num_agents"] == 4
        assert env_config["grid_size"] == 16
        assert env_config["max_steps"] == 64
        assert env_config["tasks"]["priority_inspection"]["num_pois"] == 8
        assert env_config["seed"] == 0
        assert env_config["heatmap_observation"]["enabled"] is enabled
        assert train_config["num_envs"] == 2
        assert train_config["rollout_steps"] == 32
        assert train_config["total_updates"] == 200
        assert train_config["epochs"] == 2
        assert train_config["minibatch_size"] == 64
        assert train_config["use_task_element_heatmap"] is enabled
        assert train_config["eval_interval"] == 20
        assert train_config["eval_episodes"] == 8
        assert train_config["record_eval_episodes"] == 0
        assert launch_config == {
            "env_backend": "sync",
            "init_checkpoint": None,
            "evaluation_enabled": True,
        }


def test_combined_pilot_evaluation_settings_are_resolved(pilot_configs):
    train_config = {
        **pilot_configs["on"]["policy"],
        **pilot_configs["on"]["training"],
    }
    helpers = _load_source_functions(
        LAUNCHER_PATH, ["resolve_experiment_evaluation_settings"]
    )
    resolved = helpers["resolve_experiment_evaluation_settings"](
        args=Namespace(
            eval_interval=None,
            eval_episodes=None,
            record_eval_episodes=None,
            headless=True,
        ),
        explicit_options=set(),
        train_config=train_config,
        eval_config={"eval_episodes": 10, "record_eval_episodes": 2},
    )
    assert resolved == {
        "eval_interval": 20,
        "eval_episodes": 8,
        "record_eval_episodes": 0,
        "headless": True,
    }


def test_resolved_launch_includes_pilot_evaluation_provenance(pilot_configs):
    helpers = _load_source_functions(
        LAUNCHER_PATH, ["build_resolved_launch_snapshot"]
    )
    config = pilot_configs["off"]
    train_config = {**config["policy"], **config["training"]}
    for key in ("env_backend", "init_checkpoint", "evaluation_enabled"):
        train_config.pop(key)
    resolved = helpers["build_resolved_launch_snapshot"](
        args=Namespace(
            config="configs/policy/mappo_waypoint.yaml",
            env_config="configs/env/waypoint_missions.yaml",
            eval_config="configs/eval/waypoint_eval.yaml",
        ),
        task_names=["priority_inspection"],
        env_backend="sync",
        init_checkpoint=None,
        evaluation_enabled=True,
        experiment_config=str(OFF_PATH.relative_to(ROOT)),
        training_seed=0,
        output_dir="outputs/pilot",
        run_name="priority_inspection_heatmap_off_pilot_seed0",
        train_config=train_config,
    )
    assert resolved["eval_interval"] == 20
    assert resolved["eval_episodes"] == 8
    assert resolved["record_eval_episodes"] == 0


def test_pilot_evaluation_schedule_is_bounded_and_unique():
    helpers = _load_source_functions(TRAINER_PATH, ["should_run_evaluation"])
    should_run = helpers["should_run_evaluation"]
    scheduled = [
        update
        for update in range(1, 201)
        if should_run(
            evaluation_enabled=True,
            eval_episodes=8,
            update_idx=update,
            total_updates=200,
            eval_interval=20,
        )
    ]
    assert scheduled == list(range(20, 201, 20))
    assert len(scheduled) == len(set(scheduled)) == 10
    assert 1 not in scheduled and 19 not in scheduled and 21 not in scheduled
    assert not should_run(
        evaluation_enabled=False,
        eval_episodes=8,
        update_idx=20,
        total_updates=200,
        eval_interval=20,
    )
    assert not should_run(
        evaluation_enabled=True,
        eval_episodes=0,
        update_idx=20,
        total_updates=200,
        eval_interval=20,
    )


def test_primary_metric_names_exist_in_current_training_and_evaluation_records():
    source = TRAINER_PATH.read_text(encoding="utf-8")
    missing_training = sorted(metric for metric in TRAINING_METRICS if metric not in source)
    missing_evaluation = sorted(
        metric for metric in EVALUATION_METRICS if metric not in source
    )
    assert not missing_training, "missing training metric field(s): " + ", ".join(
        missing_training
    )
    assert not missing_evaluation, "missing evaluation metric field(s): " + ", ".join(
        missing_evaluation
    )


def test_pilot_artifact_and_interpretation_contract(pilot_configs, tmp_path):
    assert pilot_configs["off"]["seed"] == pilot_configs["on"]["seed"] == 0
    assert pilot_configs["off"]["training"] == pilot_configs["on"]["training"]
    scheduled_updates = list(range(20, 201, 20))
    required_files = {
        "training_metrics.csv",
        "tensorboard/<event-file>",
        "snapshot/cli_args.yaml",
        "snapshot/resolved_launch.yaml",
        "snapshot/env_config.yaml",
        "snapshot/train_config.yaml",
        "snapshot/eval_config.yaml",
        "snapshot/metadata.json",
        "checkpoints/checkpoint_best_eval.pt",
        "best_eval_summary.json",
        *(f"checkpoints/checkpoint_{update:04d}.pt" for update in scheduled_updates),
    }
    assert "checkpoints/checkpoint_0020.pt" in required_files
    assert "checkpoints/checkpoint_0200.pt" in required_files
    assert not any(path.endswith((".gif", ".mp4")) for path in required_files)
    assert tmp_path != ROOT / "outputs"
    # One seed validates runtime and signal only. It cannot establish a heatmap
    # benefit; checkpoint sizes are not performance metrics. Evaluation and task
    # completion metrics govern comparison, and multi-seed expansion follows only
    # after this bounded pilot is stable.

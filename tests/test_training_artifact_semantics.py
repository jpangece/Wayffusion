from __future__ import annotations

import ast
from argparse import Namespace
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[1]
LAUNCHER_PATH = ROOT / "scripts/train_mappo_waypoint.py"
TRAINER_PATH = ROOT / "algorithms/mappo_waypoint.py"
ON_CONFIG = "configs/experiments/priority_inspection_heatmap_on_debug.yaml"
OFF_CONFIG = "configs/experiments/priority_inspection_heatmap_off_debug.yaml"


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _load_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    if function is None:
        pytest.skip(f"{path.relative_to(ROOT)} does not expose {name} yet")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _experiment_values(path: str) -> tuple[dict, dict, dict]:
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    env_config = dict(experiment["environment"])
    train_config = {**experiment["policy"], **experiment["training"]}
    launch_config = {
        "env_backend": train_config.pop("env_backend"),
        "init_checkpoint": train_config.pop("init_checkpoint"),
        "evaluation_enabled": train_config.pop("evaluation_enabled"),
    }
    env_config["seed"] = experiment["seed"]
    train_config["seed"] = experiment["seed"]
    return env_config, train_config, launch_config


def _raw_args(experiment_config: str | None) -> Namespace:
    return Namespace(
        experiment_config=experiment_config,
        config="configs/policy/mappo_waypoint.yaml",
        env_config="configs/env/waypoint_missions.yaml",
        eval_config="configs/eval/waypoint_eval.yaml",
        tasks=[
            "area_coverage",
            "belief_search",
            "priority_inspection",
            "connectivity_expansion",
        ],
        env_backend="process",
        init_checkpoint=None,
    )


def test_required_artifact_semantics_hooks_are_public():
    launcher_functions = _function_names(LAUNCHER_PATH)
    trainer_functions = _function_names(TRAINER_PATH)
    assert "build_resolved_launch_snapshot" in launcher_functions, (
        "scripts.train_mappo_waypoint must expose "
        "build_resolved_launch_snapshot"
    )
    assert "should_run_evaluation" in trainer_functions, (
        "algorithms.mappo_waypoint must expose should_run_evaluation"
    )
    assert "should_write_best_eval_artifacts" in trainer_functions, (
        "algorithms.mappo_waypoint must expose "
        "should_write_best_eval_artifacts"
    )


@pytest.mark.parametrize("experiment_config", [ON_CONFIG, OFF_CONFIG])
def test_combined_config_resolved_snapshot_records_effective_values(
    experiment_config,
):
    build_snapshot = _load_function(
        LAUNCHER_PATH, "build_resolved_launch_snapshot"
    )
    env_config, train_config, launch_config = _experiment_values(experiment_config)
    raw_args = _raw_args(experiment_config)
    resolved = build_snapshot(
        args=raw_args,
        task_names=["priority_inspection"],
        env_backend=launch_config["env_backend"],
        init_checkpoint=launch_config["init_checkpoint"],
        evaluation_enabled=launch_config["evaluation_enabled"],
        experiment_config=experiment_config,
        training_seed=env_config["seed"],
        output_dir="outputs/test-artifacts",
        run_name="priority_inspection_debug",
        train_config=train_config,
    )

    assert raw_args.env_backend == "process"
    assert raw_args.tasks == [
        "area_coverage",
        "belief_search",
        "priority_inspection",
        "connectivity_expansion",
    ]
    assert resolved == {
        "experiment_config": experiment_config,
        "config": "configs/policy/mappo_waypoint.yaml",
        "env_config": "configs/env/waypoint_missions.yaml",
        "eval_config": "configs/eval/waypoint_eval.yaml",
        "task_names": ["priority_inspection"],
        "env_backend": "sync",
        "init_checkpoint": None,
        "evaluation_enabled": False,
        "training_seed": 0,
        "output_dir": "outputs/test-artifacts",
        "run_name": "priority_inspection_debug",
        "num_envs": 1,
        "rollout_steps": 4,
        "total_updates": 2,
    }


def test_separate_config_mode_also_supports_resolved_snapshot():
    build_snapshot = _load_function(
        LAUNCHER_PATH, "build_resolved_launch_snapshot"
    )
    raw_args = _raw_args(None)
    resolved = build_snapshot(
        args=raw_args,
        task_names=["area_coverage"],
        env_backend="thread",
        init_checkpoint="checkpoints/input.pt",
        evaluation_enabled=True,
        experiment_config=None,
        training_seed=17,
        output_dir="outputs/separate",
        run_name="separate_run",
        train_config={"num_envs": 2, "rollout_steps": 8, "total_updates": 3},
    )
    assert resolved["experiment_config"] is None
    assert resolved["task_names"] == ["area_coverage"]
    assert resolved["env_backend"] == "thread"
    assert resolved["init_checkpoint"] == "checkpoints/input.pt"
    assert resolved["evaluation_enabled"] is True
    assert resolved["training_seed"] == 17
    assert resolved["num_envs"] == 2
    assert resolved["rollout_steps"] == 8
    assert resolved["total_updates"] == 3


def test_launcher_preserves_raw_and_resolved_snapshot_files():
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "cli_args.yaml" in source
    assert "resolved_launch.yaml" in source, (
        "launcher must write snapshot/resolved_launch.yaml separately from "
        "raw snapshot/cli_args.yaml"
    )


def test_evaluation_disabled_never_runs_or_writes_best_artifacts():
    should_run = _load_function(TRAINER_PATH, "should_run_evaluation")
    should_write_best = _load_function(
        TRAINER_PATH, "should_write_best_eval_artifacts"
    )
    assert not should_run(
        evaluation_enabled=False,
        eval_episodes=0,
        update_idx=2,
        total_updates=2,
        eval_interval=1,
    )
    assert not should_write_best(
        evaluation_enabled=False,
        evaluation_result={"eval_success_rate": 0.0, "eval_reward": 0.0},
    )


def test_evaluation_enabled_schedule_and_best_artifact_policy():
    should_run = _load_function(TRAINER_PATH, "should_run_evaluation")
    should_write_best = _load_function(
        TRAINER_PATH, "should_write_best_eval_artifacts"
    )
    assert not should_run(
        evaluation_enabled=True,
        eval_episodes=2,
        update_idx=1,
        total_updates=4,
        eval_interval=2,
    )
    assert should_run(
        evaluation_enabled=True,
        eval_episodes=2,
        update_idx=2,
        total_updates=4,
        eval_interval=2,
    )
    assert not should_write_best(
        evaluation_enabled=True,
        evaluation_result=None,
    )
    assert should_write_best(
        evaluation_enabled=True,
        evaluation_result={"eval_success_rate": 0.5, "eval_reward": 1.25},
    )


def test_trainer_contract_keeps_final_checkpoint_separate_from_best_eval():
    source = TRAINER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    train_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    argument_names = {argument.arg for argument in train_method.args.args}
    assert "evaluation_enabled" in argument_names, (
        "trainer.train must receive evaluation_enabled independently of "
        "eval_episodes"
    )
    called_functions = {
        node.func.id
        for node in ast.walk(train_method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "should_run_evaluation" in called_functions
    assert "should_write_best_eval_artifacts" in called_functions
    train_source = ast.get_source_segment(source, train_method)
    assert 'f"checkpoint_{update_idx:04d}.pt"' in train_source
    assert '"checkpoint_best_eval.pt"' in train_source
    assert '"best_eval_summary.json"' in train_source

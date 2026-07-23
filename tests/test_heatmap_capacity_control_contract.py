from __future__ import annotations

import ast
import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts/train_mappo_waypoint.py"
TRAINER = ROOT / "algorithms/mappo_waypoint.py"
EVALUATION = ROOT / "utils/waypoint_evaluation.py"
ZERO_PROVIDER = ROOT / "envs/waypoint/zero_task_element_provider.py"
CONFIG_PATHS = {
    "off": ROOT / "configs/experiments/priority_inspection_heatmap_off_ablation_seed0.yaml",
    "real": ROOT / "configs/experiments/priority_inspection_heatmap_real_ablation_seed0.yaml",
    "zero": ROOT / "configs/experiments/priority_inspection_heatmap_zero_ablation_seed0.yaml",
}
EVALUATION_SEEDS = list(range(10000, 10008))


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict), f"ablation config must be a mapping: {path}"
    return value


@pytest.fixture
def ablation_configs() -> dict[str, dict]:
    missing = [path for path in CONFIG_PATHS.values() if not path.is_file()]
    if missing:
        pytest.skip("capacity-control configs are not implemented yet")
    return {name: _load_yaml(path) for name, path in CONFIG_PATHS.items()}


def _differences(left, right, prefix="") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(_differences(left[key], right[key], path))
        return result
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        result = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            result.extend(_differences(left_item, right_item, f"{prefix}[{index}]"))
        return result
    return [] if left == right else [prefix]


def _normalized(config: dict, *, normalize_activation: bool) -> dict:
    value = deepcopy(config)
    value["experiment_name"] = "<ablation>"
    provider = value["environment"]["heatmap_observation"]["provider"]
    provider["source"] = "<source>"
    if normalize_activation:
        value["environment"]["heatmap_observation"]["enabled"] = False
        provider["enabled"] = False
        provider["refresh_on_step"] = False
        value["policy"]["use_task_element_heatmap"] = False
    return value


def _top_level_function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def _extract_function(path: Path, name: str, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    assert function is not None, f"{path.relative_to(ROOT)} missing public {name}"
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
    values = dict(namespace or {})
    exec(compile(module, str(path), "exec"), values)
    return values[name]


def test_expected_capacity_control_configs_exist_and_parse():
    missing = [path for path in CONFIG_PATHS.values() if not path.is_file()]
    assert not missing, "missing capacity-control experiment config(s): " + ", ".join(
        str(path.relative_to(ROOT)) for path in missing
    )
    for path in CONFIG_PATHS.values():
        _load_yaml(path)


def test_evaluation_seed_resolution_hook_is_public():
    launcher_functions = _top_level_function_names(LAUNCHER)
    assert "resolve_evaluation_episode_seeds" in launcher_functions


def test_policy_initialization_seed_hook_is_public():
    launcher_functions = _top_level_function_names(LAUNCHER)
    assert "seed_policy_initialization" in launcher_functions
    assert "policy_init_seed" in LAUNCHER.read_text(encoding="utf-8")
    assert "evaluation_episode_seeds" in LAUNCHER.read_text(encoding="utf-8")


def test_explicit_zero_provider_selection_hook_is_public():
    assert ZERO_PROVIDER.is_file(), (
        "missing explicit zero provider: envs/waypoint/zero_task_element_provider.py"
    )
    source = LAUNCHER.read_text(encoding="utf-8")
    registration = ast.get_source_segment(
        source,
        next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "register_heatmap_task_element_providers"
        ),
    )
    assert "source" in registration
    assert "register_zero_task_element_provider" in registration


def test_shared_capacity_control_configuration_and_bounds(ablation_configs):
    expected_names = {
        "off": "priority_inspection_heatmap_off_ablation_seed0",
        "real": "priority_inspection_heatmap_real_ablation_seed0",
        "zero": "priority_inspection_heatmap_zero_ablation_seed0",
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
        "policy_init_seed": 0,
        "evaluation_episode_seeds": EVALUATION_SEEDS,
    }
    for name, config in ablation_configs.items():
        assert config["experiment_name"] == expected_names[name]
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
        assert environment["tasks"]["priority_inspection"] == {
            "num_pois": 8,
            "num_pois_range": None,
        }
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
        assert config["training"] == expected_training
        assert config["training"]["rollout_steps"] * config["training"]["num_envs"] == 64
        assert config["training"]["total_updates"] <= 200
        assert config["training"]["rollout_steps"] <= 32
        assert config["training"]["num_envs"] <= 2
        assert config["training"]["eval_episodes"] <= 8
        assert config["training"]["record_eval_episodes"] == 0


def test_three_condition_activation_and_equality_audit(ablation_configs):
    expected = {
        "off": (False, False, False, False, "none"),
        "real": (True, True, True, True, "priority_inspection"),
        "zero": (True, True, True, True, "zero"),
    }
    for name, config in ablation_configs.items():
        heatmap = config["environment"]["heatmap_observation"]
        actual = (
            heatmap["enabled"],
            heatmap["provider"]["enabled"],
            heatmap["provider"]["refresh_on_step"],
            config["policy"]["use_task_element_heatmap"],
            heatmap["provider"]["source"],
        )
        assert actual == expected[name]
        assert heatmap["channels"] == 1

    real_zero = _differences(
        _normalized(ablation_configs["real"], normalize_activation=False),
        _normalized(ablation_configs["zero"], normalize_activation=False),
    )
    assert not real_zero, "unexpected REAL/ZERO differences: " + ", ".join(real_zero)
    for controlled in ("real", "zero"):
        differences = _differences(
            _normalized(ablation_configs["off"], normalize_activation=True),
            _normalized(ablation_configs[controlled], normalize_activation=True),
        )
        assert not differences, "unexpected OFF/control differences: " + ", ".join(
            differences
        )


def test_zero_provider_uses_normal_generator_path(ablation_configs):
    del ablation_configs
    spec = importlib.util.spec_from_file_location("zero_task_element_provider", ZERO_PROVIDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider = module.ZeroTaskElementProvider()
    elements = provider.build_task_elements(scenario=object(), world=object(), task_state={})
    assert elements == []

    generator_path = ROOT / "envs/waypoint/task_element_heatmap.py"
    generator_spec = importlib.util.spec_from_file_location("task_element_heatmap", generator_path)
    generator_module = importlib.util.module_from_spec(generator_spec)
    generator_spec.loader.exec_module(generator_module)
    first = generator_module.generate_task_element_heatmap(16, elements)
    refreshed = generator_module.generate_task_element_heatmap(
        16, provider.build_task_elements(scenario=object(), world=object(), task_state={})
    )
    for heatmap in (first, refreshed):
        assert heatmap.shape == (1, 16, 16)
        assert heatmap.dtype == np.float32
        assert np.isfinite(heatmap).all()
        assert np.count_nonzero(heatmap) == 0


def test_fixed_evaluation_seed_resolution_and_provenance(ablation_configs):
    resolver = _extract_function(LAUNCHER, "resolve_evaluation_episode_seeds")
    original = list(EVALUATION_SEEDS)
    for config in ablation_configs.values():
        resolved = resolver(
            train_config=config["training"],
            eval_config={"evaluation_episode_seeds": list(range(20000, 20008))},
            eval_episodes=8,
        )
        assert resolved == EVALUATION_SEEDS
        assert resolved is not config["training"]["evaluation_episode_seeds"]
        assert config["training"]["evaluation_episode_seeds"] == original

    with pytest.raises(ValueError, match="evaluation_episode_seeds"):
        resolver(train_config={"evaluation_episode_seeds": [1, 1]}, eval_config={}, eval_episodes=2)
    with pytest.raises(ValueError, match="evaluation_episode_seeds"):
        resolver(train_config={"evaluation_episode_seeds": [1]}, eval_config={}, eval_episodes=2)
    with pytest.raises(ValueError, match="evaluation_episode_seeds"):
        resolver(train_config={"evaluation_episode_seeds": [1, 2.5]}, eval_config={}, eval_episodes=2)

    snapshot_source = LAUNCHER.read_text(encoding="utf-8")
    assert "policy_init_seed" in snapshot_source
    assert "evaluation_episode_seeds" in snapshot_source


def test_evaluation_seed_list_is_plumbed_without_update_regeneration(ablation_configs):
    del ablation_configs
    trainer_source = TRAINER.read_text(encoding="utf-8")
    evaluation_source = EVALUATION.read_text(encoding="utf-8")
    for name in ("train", "evaluate", "_evaluate_single_mode"):
        function = next(
            node
            for node in ast.walk(ast.parse(trainer_source))
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert "evaluation_episode_seeds" in {
            argument.arg for argument in function.args.args
        }, f"{name} must accept evaluation_episode_seeds"
    for name in ("evaluate_waypoint_policy_per_task", "evaluate_waypoint_policy_episodes"):
        function = next(
            node
            for node in ast.walk(ast.parse(evaluation_source))
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert "evaluation_episode_seeds" in {
            argument.arg for argument in function.args.args
        }, f"{name} must accept evaluation_episode_seeds"
    assert "evaluation_episode_seeds[episode_idx]" in evaluation_source


def test_policy_initialization_seed_boundary_and_matched_architecture(ablation_configs):
    torch = pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    seed_policy = _extract_function(
        LAUNCHER,
        "seed_policy_initialization",
        namespace={"random": __import__("random"), "np": np, "torch": torch},
    )
    from policies.direct_waypoint_policy import DirectWaypointPolicy

    class Space:
        def __init__(self, shape):
            self.shape = shape

    spaces = {
        "global_task_field": Space((5, 16, 16)),
        "task_element_heatmap": Space((1, 16, 16)),
        "all_uav_states": Space((4, 8)),
        "task_id": Space((6,)),
        "global_info": Space((8,)),
    }
    policy_kwargs = dict(
        cnn_channels=[8, 16, 32],
        agent_hidden_dim=32,
        joint_hidden_dim=64,
        use_attention=False,
        use_task_element_heatmap=True,
    )
    seed_policy(0)
    real = DirectWaypointPolicy(spaces, 2, **policy_kwargs)
    seed_policy(0)
    zero = DirectWaypointPolicy(spaces, 2, **policy_kwargs)
    off = DirectWaypointPolicy(
        spaces,
        2,
        **{**policy_kwargs, "use_task_element_heatmap": False},
    )
    assert hasattr(real, "heatmap_encoder") and hasattr(zero, "heatmap_encoder")
    assert not hasattr(off, "heatmap_encoder")
    assert real.state_dict().keys() == zero.state_dict().keys()
    for key in real.state_dict():
        assert real.state_dict()[key].shape == zero.state_dict()[key].shape
        assert torch.equal(real.state_dict()[key], zero.state_dict()[key])
    assert sum(value.numel() for value in real.state_dict().values()) == sum(
        value.numel() for value in zero.state_dict().values()
    )
    assert ablation_configs["off"]["policy"]["use_task_element_heatmap"] is False
    launcher_tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
    main = next(
        node
        for node in launcher_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"seed_policy_initialization", "build_policy"}
    }
    assert calls["seed_policy_initialization"] < calls["build_policy"]


def test_schedule_and_scientific_interpretation_contract():
    should_run = _extract_function(TRAINER, "should_run_evaluation")
    schedule = [
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
    assert schedule == list(range(20, 201, 20))
    assert len(schedule) == len(set(schedule))
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
    # REAL-ZERO estimates semantic information under matched architecture;
    # ZERO-OFF estimates architecture/capacity; REAL-OFF is the total system
    # difference. One seed is not a performance claim, repeated checkpoint
    # evaluations are not automatically independent samples, and model size is
    # not a performance metric. Multi-seed work follows validated controls.

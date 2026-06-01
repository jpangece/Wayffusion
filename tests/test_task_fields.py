from __future__ import annotations

import warnings

import numpy as np

from envs import CentralizedMultiUAVEnv
from fields.task_field import CHANNEL_INDEX, CHANNEL_NAMES, build_task_field
from scripts._common import load_env_config


def _env_config(**override) -> dict:
    base_override = {"task_name": "coverage", "num_agents": 4}
    base_override.update(override)
    return load_env_config("configs/env/multitask.yaml", override=base_override)


def test_channel_names_order_is_stable():
    assert CHANNEL_NAMES == [
        "obstacle",
        "goal_reward",
        "target_probability",
        "desired_occupancy",
        "risk",
        "visited",
        "agent_density",
        "communication_quality",
        "formation_template",
    ]


def test_build_task_field_fills_known_channels_and_ignores_unknown():
    grid_size = 8
    field = build_task_field(
        {
            "goal_reward": np.ones((grid_size, grid_size), dtype=np.float32),
            "unknown_channel": np.full((grid_size, grid_size), 7.0, dtype=np.float32),
        },
        grid_size=grid_size,
    )

    assert np.allclose(field[CHANNEL_INDEX["goal_reward"]], 1.0)
    untouched = [idx for name, idx in CHANNEL_INDEX.items() if name != "goal_reward"]
    assert np.allclose(field[untouched], 0.0)


def test_all_tasks_share_same_task_field_shape():
    config = load_env_config("configs/env/multitask.yaml")
    for task_name in ["goal_nav", "coverage", "formation", "risk_nav"]:
        env = CentralizedMultiUAVEnv({**config, "task_name": task_name})
        obs, info = env.reset(seed=3)
        assert obs["task_field"].shape == (len(CHANNEL_NAMES), config["grid_size"], config["grid_size"])
        if task_name == "goal_nav":
            assert np.allclose(info["full_task_field"][CHANNEL_INDEX["target_probability"]], 0.0)
        if task_name == "coverage":
            assert info["full_task_field"][CHANNEL_INDEX["target_probability"]].sum() > 0.0
        if task_name == "formation":
            assert info["full_task_field"][CHANNEL_INDEX["formation_template"]].sum() > 0.0


def test_coverage_route_hint_uses_formation_template_channel_when_enabled():
    env = CentralizedMultiUAVEnv(
        _env_config(
            coverage={
                "route_hint_enabled": True,
                "route_hint_stride": 3,
                "route_hint_sigma": 0.04,
            }
        )
    )
    _, info = env.reset(seed=14)
    route_hint = info["full_task_field"][CHANNEL_INDEX["formation_template"]]
    assert route_hint.shape == (env.grid_size, env.grid_size)
    assert float(route_hint.sum()) > 0.0
    assert env.current_task_state["route_hint_routes"] is not None
    assert len(env.current_task_state["route_hint_routes"]) == env.num_agents


def test_coverage_route_targets_can_be_appended_to_agent_observation():
    env = CentralizedMultiUAVEnv(
        _env_config(
            include_route_targets_in_agents=True,
            coverage={"route_hint_enabled": True},
        )
    )
    obs, _ = env.reset(seed=15)
    assert obs["agents"].shape == (env.num_agents, 10)
    assert env.observation_space["agents"].shape == (env.num_agents, 10)
    assert np.all(np.isfinite(obs["agents"]))
    assert np.all(obs["agents"][:, 6:8] >= -1.0)
    assert np.all(obs["agents"][:, 6:8] <= 1.0)
    assert np.all(obs["agents"][:, 8:10] >= 0.0)
    assert np.all(obs["agents"][:, 8:10] <= 1.0)


def test_goal_nav_task_targets_can_be_appended_to_agent_observation():
    config = load_env_config(
        "configs/env/multitask.yaml",
        override={
            "task_name": "goal_nav",
            "num_agents": 4,
            "include_task_targets_in_agents": True,
        },
    )
    env = CentralizedMultiUAVEnv(config)
    obs, _ = env.reset(seed=16)

    assert obs["agents"].shape == (env.num_agents, 10)
    assert env.observation_space["agents"].shape == (env.num_agents, 10)
    assert np.all(np.isfinite(obs["agents"]))
    assert np.all(obs["agents"][:, 6:8] >= -1.0)
    assert np.all(obs["agents"][:, 6:8] <= 1.0)
    assert np.all(obs["agents"][:, 8:10] >= 0.0)
    assert np.all(obs["agents"][:, 8:10] <= 1.0)

    map_size = float(obs["global_info"][-1])
    target_positions = obs["agents"][:, 8:10] * map_size
    remaining_goals = env.current_task_state["goals"][~env.current_task_state["goal_reached"]]
    current_positions = env.state["positions"]
    for target, position in zip(target_positions, current_positions):
        distance_to_goal = np.linalg.norm(remaining_goals - target, axis=-1).min()
        distance_to_self = np.linalg.norm(position - target)
        assert min(distance_to_goal, distance_to_self) < 1e-5


def test_drop_channels_and_agent_density_toggle_zero_out_channels():
    env = CentralizedMultiUAVEnv(
        _env_config(
            include_agent_density=False,
            drop_channels=["risk"],
        )
    )
    _, info = env.reset(seed=9)
    assert np.allclose(info["full_task_field"][CHANNEL_INDEX["risk"]], 0.0)
    assert np.allclose(info["full_task_field"][CHANNEL_INDEX["agent_density"]], 0.0)


def test_observation_variants_shapes_and_alias_equivalence():
    env_multi = CentralizedMultiUAVEnv(_env_config(observation_mode="multi_channel_field"))
    obs_multi, _ = env_multi.reset(seed=9)
    assert obs_multi["task_field"].shape == (len(CHANNEL_NAMES), env_multi.grid_size, env_multi.grid_size)

    env_single = CentralizedMultiUAVEnv(_env_config(observation_mode="single_channel_field"))
    obs_single, _ = env_single.reset(seed=9)
    assert obs_single["task_field"].shape == (1, env_single.grid_size, env_single.grid_size)

    env_zero = CentralizedMultiUAVEnv(_env_config(observation_mode="no_spatial_field"))
    obs_zero, _ = env_zero.reset(seed=9)
    assert obs_zero["task_field"].shape == (len(CHANNEL_NAMES), env_zero.grid_size, env_zero.grid_size)
    assert np.allclose(obs_zero["task_field"], 0.0)
    assert obs_zero["agents"].shape == (env_zero.num_agents, 6)
    assert obs_zero["task_id"].shape == (4,)
    assert obs_zero["global_info"].shape == (5,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        env_alias = CentralizedMultiUAVEnv(_env_config(observation_mode="task_id_only"))
        obs_alias, _ = env_alias.reset(seed=9)

    assert np.allclose(obs_alias["task_field"], obs_zero["task_field"])
    assert np.allclose(obs_alias["agents"], obs_zero["agents"])
    assert np.allclose(obs_alias["task_id"], obs_zero["task_id"])
    assert np.allclose(obs_alias["global_info"], obs_zero["global_info"])

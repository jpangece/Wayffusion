from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import yaml

from envs.waypoint_marl_env import WaypointMultiUAVEnv


GLOBAL_KEYS = {
    "task_field",
    "global_task_field",
    "all_uav_states",
    "task_id",
    "mission_goal",
    "goal_mask",
    "global_info",
    "comm_adjacency",
    "agent_mask",
}

AGENT_KEYS = {
    "self_state",
    "neighbor_relative_states",
    "task_field",
    "task_id",
    "mission_goal",
    "goal_mask",
    "global_summary",
    "all_uav_states",
    "comm_adjacency",
}

TASK_ELEMENTS = [
    {"x": 1, "y": 2, "priority": 2.0},
    {"x": 1, "y": 2, "priority": 1.0},
    {"x": 3, "y": 0, "priority": 6.0},
]


@pytest.fixture
def base_config() -> dict:
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.update(
        {
            "num_agents": 2,
            "grid_size": 8,
            "max_steps": 2,
            "task_name": "area_coverage",
            "task_names": ["area_coverage"],
        }
    )
    config["domain_randomization"]["enabled"] = False
    return config


def _enabled_config(base_config: dict, task_elements: list[dict]) -> dict:
    config = deepcopy(base_config)
    config["heatmap_observation"] = {
        "enabled": True,
        "channels": 1,
        "task_elements": deepcopy(task_elements),
    }
    return config


@pytest.mark.parametrize("explicitly_disabled", [False, True])
def test_disabled_mode_preserves_exact_legacy_observation_keys(base_config, explicitly_disabled):
    config = deepcopy(base_config)
    if explicitly_disabled:
        config["heatmap_observation"] = {"enabled": False}
    env = WaypointMultiUAVEnv(config)
    observations, _ = env.reset(seed=41)
    global_state = env.get_global_state()

    assert set(global_state) == GLOBAL_KEYS
    assert "task_element_heatmap" not in global_state
    for observation in observations.values():
        assert set(observation) == AGENT_KEYS
        assert "task_element_heatmap" not in observation
    env.close()


def test_enabled_mode_generates_shared_configured_heatmap_on_reset(base_config):
    env = WaypointMultiUAVEnv(_enabled_config(base_config, TASK_ELEMENTS))
    observations, _ = env.reset(seed=41)
    global_heatmap = env.get_global_state()["task_element_heatmap"]

    assert global_heatmap.shape == (1, 8, 8)
    assert global_heatmap.dtype == np.float32
    assert global_heatmap[0, 2, 1] == pytest.approx(0.5)
    assert global_heatmap[0, 0, 3] == pytest.approx(1.0)
    for observation in observations.values():
        assert "task_element_heatmap" in observation
        assert np.array_equal(observation["task_element_heatmap"], global_heatmap)
    env.close()


def test_configured_heatmap_remains_present_after_one_step(base_config):
    env = WaypointMultiUAVEnv(_enabled_config(base_config, TASK_ELEMENTS))
    observations, _ = env.reset(seed=41)
    expected = env.get_global_state()["task_element_heatmap"].copy()
    actions = {
        agent: observations[agent]["self_state"][:2].copy()
        for agent in env.possible_agents
    }

    next_observations, _, _, _, _ = env.step(actions)
    next_global = env.get_global_state()["task_element_heatmap"]

    assert np.array_equal(next_global, expected)
    for observation in next_observations.values():
        assert np.array_equal(observation["task_element_heatmap"], expected)
    env.close()


def test_enabled_mode_with_empty_source_returns_zero_heatmap(base_config):
    env = WaypointMultiUAVEnv(_enabled_config(base_config, []))
    observations, _ = env.reset(seed=41)
    expected = np.zeros((1, 8, 8), dtype=np.float32)

    assert np.array_equal(env.get_global_state()["task_element_heatmap"], expected)
    assert np.array_equal(observations["uav_0"]["task_element_heatmap"], expected)
    env.close()


def test_invalid_configured_coordinate_raises_value_error(base_config):
    config = _enabled_config(
        base_config,
        [{"x": 8, "y": 0, "priority": 1.0}],
    )
    env = WaypointMultiUAVEnv(config)

    with pytest.raises(ValueError):
        env.reset(seed=41)
    env.close()

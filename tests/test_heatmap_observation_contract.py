from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import yaml

from envs.waypoint_marl_env import WaypointMultiUAVEnv


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


def test_heatmap_observation_is_disabled_by_default(base_config):
    env = WaypointMultiUAVEnv(deepcopy(base_config))
    observations, _ = env.reset(seed=31)

    assert "task_element_heatmap" not in env.global_observation_space.spaces
    assert "task_element_heatmap" not in env.observation_space("uav_0").spaces
    assert "task_element_heatmap" not in env.get_global_state()
    assert "task_element_heatmap" not in observations["uav_0"]
    env.close()


def test_enabled_heatmap_observation_has_fixed_space_and_value_contract(base_config):
    config = deepcopy(base_config)
    config["heatmap_observation"] = {"enabled": True, "channels": 3}
    env = WaypointMultiUAVEnv(config)
    observations, _ = env.reset(seed=31)
    global_state = env.get_global_state()

    expected_shape = (3, 8, 8)
    global_heatmap = global_state["task_element_heatmap"]
    agent_heatmap = observations["uav_0"]["task_element_heatmap"]

    assert global_heatmap.shape == expected_shape
    assert agent_heatmap.shape == expected_shape
    assert global_heatmap.dtype == np.float32
    assert agent_heatmap.dtype == np.float32
    assert np.array_equal(global_heatmap, np.zeros(expected_shape, dtype=np.float32))
    assert np.array_equal(agent_heatmap, global_heatmap)
    assert env.global_observation_space.contains(global_state)
    assert env.observation_space("uav_0").contains(observations["uav_0"])
    env.close()


def test_heatmap_observation_contract_persists_across_step(base_config):
    config = deepcopy(base_config)
    config["heatmap_observation"] = {"enabled": True}
    env = WaypointMultiUAVEnv(config)
    observations, _ = env.reset(seed=31)
    actions = {
        agent: observations[agent]["self_state"][:2].copy()
        for agent in env.possible_agents
    }

    next_observations, _, _, _, _ = env.step(actions)

    assert next_observations["uav_0"]["task_element_heatmap"].shape == (1, 8, 8)
    assert next_observations["uav_0"]["task_element_heatmap"].dtype == np.float32
    assert env.observation_space("uav_0").contains(next_observations["uav_0"])
    env.close()


def test_heatmap_observation_rejects_non_positive_channel_count(base_config):
    config = deepcopy(base_config)
    config["heatmap_observation"] = {"enabled": True, "channels": 0}

    with pytest.raises(ValueError, match="channels must be at least 1"):
        WaypointMultiUAVEnv(config)

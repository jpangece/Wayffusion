from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import yaml

from envs.waypoint.task_element_provider import (
    clear_task_element_provider_registry,
    register_task_element_provider,
)
from envs.waypoint_marl_env import WaypointMultiUAVEnv


STATIC_ELEMENTS = [{"x": 0, "y": 0, "priority": 1.0}]
PROVIDER_ELEMENTS = [{"x": 3, "y": 2, "priority": 4.0}]


@pytest.fixture(autouse=True)
def isolated_provider_registry():
    clear_task_element_provider_registry()
    yield
    clear_task_element_provider_registry()


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


def _heatmap_config(base_config: dict, provider_enabled: bool | None) -> dict:
    config = deepcopy(base_config)
    heatmap_config = {
        "enabled": True,
        "channels": 1,
        "task_elements": deepcopy(STATIC_ELEMENTS),
    }
    if provider_enabled is not None:
        heatmap_config["provider"] = {"enabled": provider_enabled}
    config["heatmap_observation"] = heatmap_config
    return config


def _counting_provider(result):
    calls = {"constructed": 0, "built": 0, "inputs": []}

    class Provider:
        def build_task_elements(self, *, scenario, world, task_state):
            calls["built"] += 1
            calls["inputs"].append((scenario, world, task_state))
            return result

    def factory():
        calls["constructed"] += 1
        return Provider()

    return factory, calls


def test_absent_provider_configuration_preserves_static_source(base_config):
    env = WaypointMultiUAVEnv(_heatmap_config(base_config, provider_enabled=None))
    env.reset(seed=47)
    heatmap = env.get_global_state()["task_element_heatmap"]

    assert heatmap[0, 0, 0] == pytest.approx(1.0)
    assert heatmap[0, 2, 3] == pytest.approx(0.0)
    env.close()


def test_explicitly_disabled_provider_is_not_constructed_or_invoked(base_config):
    factory, calls = _counting_provider(PROVIDER_ELEMENTS)
    register_task_element_provider("area_coverage", factory)
    env = WaypointMultiUAVEnv(_heatmap_config(base_config, provider_enabled=False))

    env.reset(seed=47)
    heatmap = env.get_global_state()["task_element_heatmap"]

    assert calls["constructed"] == 0
    assert calls["built"] == 0
    assert heatmap[0, 0, 0] == pytest.approx(1.0)
    assert heatmap[0, 2, 3] == pytest.approx(0.0)
    env.close()


def test_enabled_registered_provider_wins_and_receives_reset_state_by_identity(base_config):
    factory, calls = _counting_provider(PROVIDER_ELEMENTS)
    register_task_element_provider("area_coverage", factory)
    env = WaypointMultiUAVEnv(_heatmap_config(base_config, provider_enabled=True))

    observations, _ = env.reset(seed=47)
    heatmap = env.get_global_state()["task_element_heatmap"]

    assert calls["constructed"] == 1
    assert calls["built"] == 1
    scenario, world, task_state = calls["inputs"][0]
    assert scenario is env.current_scenario
    assert world is env.world
    assert task_state is env.task_state
    assert heatmap[0, 2, 3] == pytest.approx(1.0)
    assert heatmap[0, 0, 0] == pytest.approx(0.0)
    for observation in observations.values():
        assert np.array_equal(observation["task_element_heatmap"], heatmap)
    env.close()


def test_empty_registered_provider_result_does_not_use_static_fallback(base_config):
    factory, calls = _counting_provider([])
    register_task_element_provider("area_coverage", factory)
    env = WaypointMultiUAVEnv(_heatmap_config(base_config, provider_enabled=True))

    env.reset(seed=47)
    heatmap = env.get_global_state()["task_element_heatmap"]

    assert calls["constructed"] == 1
    assert calls["built"] == 1
    assert np.array_equal(heatmap, np.zeros((1, 8, 8), dtype=np.float32))
    env.close()


def test_unregistered_active_task_uses_static_fallback(base_config):
    env = WaypointMultiUAVEnv(_heatmap_config(base_config, provider_enabled=True))

    env.reset(seed=47)
    heatmap = env.get_global_state()["task_element_heatmap"]

    assert heatmap[0, 0, 0] == pytest.approx(1.0)
    assert heatmap[0, 2, 3] == pytest.approx(0.0)
    env.close()


def test_disabled_heatmap_does_not_construct_provider_or_add_observation_key(base_config):
    factory, calls = _counting_provider(PROVIDER_ELEMENTS)
    register_task_element_provider("area_coverage", factory)
    config = _heatmap_config(base_config, provider_enabled=True)
    config["heatmap_observation"]["enabled"] = False
    env = WaypointMultiUAVEnv(config)

    observations, _ = env.reset(seed=47)

    assert calls["constructed"] == 0
    assert calls["built"] == 0
    assert "task_element_heatmap" not in env.get_global_state()
    assert all("task_element_heatmap" not in observation for observation in observations.values())
    env.close()


def test_provider_resolves_once_per_reset_and_not_during_step(base_config):
    factory, calls = _counting_provider(PROVIDER_ELEMENTS)
    register_task_element_provider("area_coverage", factory)
    env = WaypointMultiUAVEnv(_heatmap_config(base_config, provider_enabled=True))

    env.reset(seed=47)
    observations, _ = env.reset(seed=48)
    expected = env.get_global_state()["task_element_heatmap"].copy()
    actions = {
        agent: observations[agent]["self_state"][:2].copy()
        for agent in env.possible_agents
    }
    next_observations, _, _, _, _ = env.step(actions)

    assert calls["constructed"] == 2
    assert calls["built"] == 2
    assert np.array_equal(env.get_global_state()["task_element_heatmap"], expected)
    assert np.array_equal(next_observations["uav_0"]["task_element_heatmap"], expected)
    env.close()

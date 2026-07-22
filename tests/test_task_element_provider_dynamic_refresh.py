from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import yaml

from envs.waypoint.priority_inspection_task_element_provider import (
    register_priority_inspection_task_element_provider,
)
from envs.waypoint.task_element_provider import (
    clear_task_element_provider_registry,
    register_task_element_provider,
)
from envs.waypoint_marl_env import WaypointMultiUAVEnv


STATIC_ELEMENTS = [{"x": 0, "y": 0, "priority": 1.0}]


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
            "max_steps": 16,
            "task_name": "area_coverage",
            "task_names": ["area_coverage"],
        }
    )
    config["domain_randomization"]["enabled"] = False
    return config


def _config(
    base_config: dict,
    *,
    provider_enabled: bool = True,
    refresh_on_step: bool | None = None,
    heatmap_enabled: bool = True,
) -> dict:
    config = deepcopy(base_config)
    provider = {"enabled": provider_enabled}
    if refresh_on_step is not None:
        provider["refresh_on_step"] = refresh_on_step
    config["heatmap_observation"] = {
        "enabled": heatmap_enabled,
        "channels": 1,
        "task_elements": deepcopy(STATIC_ELEMENTS),
        "provider": provider,
    }
    return config


def _actions(env: WaypointMultiUAVEnv) -> dict[str, np.ndarray]:
    positions = env.world.get_uav_positions()
    return {
        agent: positions[index].copy()
        for index, agent in enumerate(env.possible_agents)
    }


def _counting_provider():
    calls = {"constructed": 0, "built": 0, "inputs": [], "markers": []}

    class Provider:
        def build_task_elements(self, *, scenario, world, task_state):
            calls["built"] += 1
            calls["inputs"].append((scenario, world, task_state))
            marker = int(task_state.get("refresh_marker", 0))
            calls["markers"].append(marker)
            return [{"x": marker, "y": 1, "priority": 1.0}]

    def factory():
        calls["constructed"] += 1
        return Provider()

    return factory, calls


def _install_step_marker(env: WaypointMultiUAVEnv) -> None:
    original_step_update = env.current_scenario.step_update

    def step_update(world, task_state):
        original_step_update(world, task_state)
        task_state["refresh_marker"] = int(task_state.get("refresh_marker", 0)) + 1

    env.current_scenario.step_update = step_update


@pytest.mark.parametrize("refresh_on_step", [None, False])
def test_refresh_defaults_off_and_preserves_reset_heatmap(base_config, refresh_on_step):
    factory, calls = _counting_provider()
    register_task_element_provider("area_coverage", factory)
    env = WaypointMultiUAVEnv(
        _config(base_config, refresh_on_step=refresh_on_step)
    )

    observations, _ = env.reset(seed=71)
    initial = observations["uav_0"]["task_element_heatmap"].copy()
    env.step(_actions(env))
    env.step_global(_actions(env))

    assert calls["constructed"] == 1
    assert calls["built"] == 1
    assert np.array_equal(env.get_global_state()["task_element_heatmap"], initial)
    env.close()


def test_step_refreshes_once_after_task_update_and_before_observations(base_config):
    factory, calls = _counting_provider()
    register_task_element_provider("area_coverage", factory)
    env = WaypointMultiUAVEnv(_config(base_config, refresh_on_step=True))

    env.reset(seed=71)
    _install_step_marker(env)
    observations, _, _, _, _ = env.step(_actions(env))
    global_heatmap = env.get_global_state()["task_element_heatmap"]

    assert calls["constructed"] == 2
    assert calls["built"] == 2
    assert calls["markers"] == [0, 1]
    scenario, world, task_state = calls["inputs"][-1]
    assert scenario is env.current_scenario
    assert world is env.world
    assert task_state is env.task_state
    assert global_heatmap[0, 1, 1] == pytest.approx(1.0)
    assert global_heatmap[0, 1, 0] == pytest.approx(0.0)
    for observation in observations.values():
        assert np.array_equal(observation["task_element_heatmap"], global_heatmap)

    env.get_global_state()
    assert calls["constructed"] == 2
    assert calls["built"] == 2
    env.close()


def test_step_global_refreshes_once_before_returned_global_state(base_config):
    factory, calls = _counting_provider()
    register_task_element_provider("area_coverage", factory)
    env = WaypointMultiUAVEnv(_config(base_config, refresh_on_step=True))

    env.reset(seed=71)
    _install_step_marker(env)
    current_global, next_global, *_ = env.step_global(_actions(env))

    assert calls["constructed"] == 2
    assert calls["built"] == 2
    assert calls["markers"] == [0, 1]
    assert current_global["task_element_heatmap"][0, 1, 1] == pytest.approx(1.0)
    assert np.array_equal(
        current_global["task_element_heatmap"],
        next_global["task_element_heatmap"],
    )
    env.get_global_state()
    assert calls["built"] == 2
    env.close()


def test_each_reset_and_step_performs_one_resolution(base_config):
    factory, calls = _counting_provider()
    register_task_element_provider("area_coverage", factory)
    env = WaypointMultiUAVEnv(_config(base_config, refresh_on_step=True))

    env.reset(seed=71)
    env.step(_actions(env))
    env.reset(seed=72)
    env.step(_actions(env))

    assert calls["constructed"] == 4
    assert calls["built"] == 4
    env.close()


def test_terminal_step_refreshes_final_observations(base_config):
    factory, calls = _counting_provider()
    register_task_element_provider("area_coverage", factory)
    env = WaypointMultiUAVEnv(_config(base_config, refresh_on_step=True))

    env.reset(seed=71)
    _install_step_marker(env)
    env.world.step_count = env.world.max_steps - 1
    observations, _, _, truncations, infos = env.step(_actions(env))

    assert all(truncations.values())
    assert calls["built"] == 2
    expected = env.get_global_state()["task_element_heatmap"]
    for agent in env.possible_agents:
        assert np.array_equal(observations[agent]["task_element_heatmap"], expected)
        assert np.array_equal(
            infos[agent]["terminal_observation"]["task_element_heatmap"],
            expected,
        )
    env.close()


def test_heatmap_disabled_ignores_refresh(base_config):
    factory, calls = _counting_provider()
    register_task_element_provider("area_coverage", factory)
    env = WaypointMultiUAVEnv(
        _config(base_config, refresh_on_step=True, heatmap_enabled=False)
    )

    observations, _ = env.reset(seed=71)
    next_observations, *_ = env.step(_actions(env))

    assert calls["constructed"] == 0
    assert calls["built"] == 0
    assert "task_element_heatmap" not in env.get_global_state()
    assert all("task_element_heatmap" not in value for value in observations.values())
    assert all("task_element_heatmap" not in value for value in next_observations.values())
    env.close()


def test_provider_disabled_ignores_refresh_and_preserves_static_source(base_config):
    factory, calls = _counting_provider()
    register_task_element_provider("area_coverage", factory)
    env = WaypointMultiUAVEnv(
        _config(base_config, provider_enabled=False, refresh_on_step=True)
    )

    env.reset(seed=71)
    initial = env.get_global_state()["task_element_heatmap"].copy()
    observations, *_ = env.step(_actions(env))

    assert calls["constructed"] == 0
    assert calls["built"] == 0
    assert initial[0, 0, 0] == pytest.approx(1.0)
    assert np.array_equal(observations["uav_0"]["task_element_heatmap"], initial)
    env.close()


def test_unregistered_task_with_refresh_uses_static_source(base_config):
    env = WaypointMultiUAVEnv(_config(base_config, refresh_on_step=True))

    env.reset(seed=71)
    initial = env.get_global_state()["task_element_heatmap"].copy()
    observations, *_ = env.step(_actions(env))

    assert initial[0, 0, 0] == pytest.approx(1.0)
    assert np.array_equal(observations["uav_0"]["task_element_heatmap"], initial)
    env.close()


def _expected_priority_heatmap(env: WaypointMultiUAVEnv) -> np.ndarray:
    pois = np.asarray(env.task_state["pois"], dtype=np.float32)
    weights = np.asarray(env.task_state["weights"], dtype=np.float32)
    visited = np.asarray(env.task_state["visited"], dtype=bool)
    grid_xy = env.world.world_to_grid(pois)
    expected = np.zeros((1, env.world.grid_size, env.world.grid_size), dtype=np.float32)
    for (grid_x, grid_y), weight, is_visited in zip(grid_xy, weights, visited):
        if not is_visited:
            expected[0, int(grid_y), int(grid_x)] += float(weight)
    maximum = float(expected.max())
    if maximum > 0.0:
        expected /= maximum
    return expected


def test_real_priority_provider_consumes_current_visited_state_after_step(base_config):
    config = _config(base_config, refresh_on_step=True)
    config["task_name"] = "priority_inspection"
    config["task_names"] = ["priority_inspection"]
    config["tasks"]["priority_inspection"]["num_pois"] = 5
    config["tasks"]["priority_inspection"]["num_pois_range"] = None
    register_priority_inspection_task_element_provider()
    env = WaypointMultiUAVEnv(config)

    env.reset(seed=53)
    initial = env.get_global_state()["task_element_heatmap"].copy()
    env.task_state["visited"][0] = True
    observations, *_ = env.step(_actions(env))
    expected = _expected_priority_heatmap(env)

    assert not np.array_equal(expected, initial)
    np.testing.assert_allclose(
        observations["uav_0"]["task_element_heatmap"],
        expected,
    )
    np.testing.assert_allclose(
        env.get_global_state()["task_element_heatmap"],
        expected,
    )
    env.close()

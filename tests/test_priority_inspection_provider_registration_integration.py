from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml


PROVIDER_MODULE_PATH = (
    Path(__file__).parents[1]
    / "envs"
    / "waypoint"
    / "priority_inspection_task_element_provider.py"
)
PROVIDER_MODULE_SPEC = importlib.util.spec_from_file_location(
    "envs.waypoint.priority_inspection_task_element_provider",
    PROVIDER_MODULE_PATH,
)
if PROVIDER_MODULE_SPEC is None or PROVIDER_MODULE_SPEC.loader is None:
    raise ImportError(f"Could not load module specification from {PROVIDER_MODULE_PATH}")
PROVIDER_MODULE = importlib.util.module_from_spec(PROVIDER_MODULE_SPEC)
PROVIDER_MODULE_SPEC.loader.exec_module(PROVIDER_MODULE)
register_priority_inspection_task_element_provider = (
    PROVIDER_MODULE.register_priority_inspection_task_element_provider
)

from envs.waypoint.task_element_provider import clear_task_element_provider_registry
from envs.waypoint_marl_env import WaypointMultiUAVEnv


STATIC_ELEMENTS = [{"x": 0, "y": 7, "priority": 1.0}]


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
            "task_name": "priority_inspection",
            "task_names": ["priority_inspection"],
        }
    )
    config["domain_randomization"]["enabled"] = False
    config["tasks"]["priority_inspection"]["num_pois"] = 5
    config["tasks"]["priority_inspection"]["num_pois_range"] = None
    return config


def _config(base_config: dict, provider_enabled: bool | None) -> dict:
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


def _static_heatmap(grid_size: int) -> np.ndarray:
    expected = np.zeros((1, grid_size, grid_size), dtype=np.float32)
    expected[0, 7, 0] = 1.0
    return expected


def _expected_provider_heatmap(env: WaypointMultiUAVEnv) -> np.ndarray:
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


@pytest.mark.parametrize("provider_enabled", [None, False])
def test_real_registration_does_not_change_default_static_behavior(base_config, provider_enabled):
    register_priority_inspection_task_element_provider()
    env = WaypointMultiUAVEnv(_config(base_config, provider_enabled))

    observations, _ = env.reset(seed=53)
    expected = _static_heatmap(env.world.grid_size)

    assert np.array_equal(env.get_global_state()["task_element_heatmap"], expected)
    assert np.array_equal(observations["uav_0"]["task_element_heatmap"], expected)
    env.close()


def test_enabled_real_provider_uses_live_priority_inspection_state(base_config):
    register_priority_inspection_task_element_provider()
    register_priority_inspection_task_element_provider()
    env = WaypointMultiUAVEnv(_config(base_config, provider_enabled=True))

    observations, _ = env.reset(seed=53)
    actual = env.get_global_state()["task_element_heatmap"]
    expected = _expected_provider_heatmap(env)

    assert actual.shape == (1, 8, 8)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected)
    assert not np.array_equal(actual, _static_heatmap(env.world.grid_size))
    for observation in observations.values():
        assert np.array_equal(observation["task_element_heatmap"], actual)
    env.close()


def test_real_provider_preserves_grid_xy_to_heatmap_yx_orientation(base_config):
    register_priority_inspection_task_element_provider()
    env = WaypointMultiUAVEnv(_config(base_config, provider_enabled=True))

    env.reset(seed=53)
    actual = env.get_global_state()["task_element_heatmap"]
    expected = _expected_provider_heatmap(env)
    asymmetric_cells = np.argwhere(expected[0] != expected[0].T)

    assert len(asymmetric_cells) > 0
    grid_y, grid_x = map(int, asymmetric_cells[0])
    assert actual[0, grid_y, grid_x] == pytest.approx(expected[0, grid_y, grid_x])
    assert actual[0, grid_y, grid_x] != pytest.approx(expected[0, grid_x, grid_y])
    env.close()


def test_registration_can_be_reapplied_after_registry_clear(base_config):
    register_priority_inspection_task_element_provider()
    clear_task_element_provider_registry()
    register_priority_inspection_task_element_provider()
    env = WaypointMultiUAVEnv(_config(base_config, provider_enabled=True))

    env.reset(seed=53)

    np.testing.assert_allclose(
        env.get_global_state()["task_element_heatmap"],
        _expected_provider_heatmap(env),
    )
    env.close()

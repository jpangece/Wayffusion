from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "envs"
    / "waypoint"
    / "priority_inspection_task_element_provider.py"
)
if not MODULE_PATH.exists():
    raise ModuleNotFoundError(
        "No module named 'envs.waypoint.priority_inspection_task_element_provider'"
    )
MODULE_SPEC = importlib.util.spec_from_file_location(
    "envs.waypoint.priority_inspection_task_element_provider",
    MODULE_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"Could not load module specification from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)
PriorityInspectionTaskElementProvider = MODULE.PriorityInspectionTaskElementProvider


class GridWorld:
    """Dependency-light stand-in matching MissionWorld's grid conversion fields."""

    def __init__(self, map_size: float = 2.0, grid_size: int = 4):
        self.map_size = map_size
        self.grid_size = grid_size
        self.geofence = (0.0, map_size, 0.0, map_size)

    def world_to_grid(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        scaled = np.clip(points[..., :2] / max(self.map_size, 1e-6), 0.0, 0.999999)
        return np.floor(scaled * self.grid_size).astype(np.int64)


@pytest.fixture
def provider():
    return PriorityInspectionTaskElementProvider()


@pytest.fixture
def world():
    return GridWorld()


def _state(pois, weights, visited):
    return {
        "pois": np.asarray(pois, dtype=np.float32),
        "weights": np.asarray(weights, dtype=np.float32),
        "visited": np.asarray(visited, dtype=bool),
        "deadlines": np.arange(len(weights), dtype=np.int64) + 10,
        "repeat_visit_count": 7,
    }


def _build(provider, world, task_state):
    return provider.build_task_elements(
        scenario=object(),
        world=world,
        task_state=task_state,
    )


def test_unvisited_pois_convert_to_grid_xy_with_weights_as_priority(provider, world):
    task_state = _state(
        pois=[[0.25, 0.25], [1.25, 1.75]],
        weights=[2.0, 5.0],
        visited=[False, False],
    )

    elements = _build(provider, world, task_state)

    assert elements == [
        {"x": 0, "y": 0, "priority": 2.0},
        {"x": 2, "y": 3, "priority": 5.0},
    ]


def test_visited_pois_are_filtered_and_source_order_is_preserved(provider, world):
    task_state = _state(
        pois=[[1.75, 0.25], [0.75, 1.25], [0.25, 1.75]],
        weights=[3.0, 4.0, 5.0],
        visited=[False, True, False],
    )

    elements = _build(provider, world, task_state)

    assert elements == [
        {"x": 3, "y": 0, "priority": 3.0},
        {"x": 0, "y": 3, "priority": 5.0},
    ]


def test_all_visited_returns_empty_sequence(provider, world):
    task_state = _state(
        pois=[[0.25, 0.25], [1.25, 1.25]],
        weights=[1.0, 2.0],
        visited=[True, True],
    )

    assert _build(provider, world, task_state) == []


def test_same_cell_pois_remain_separate_and_in_source_order(provider, world):
    task_state = _state(
        pois=[[0.10, 0.10], [0.20, 0.20]],
        weights=[1.5, 2.5],
        visited=[False, False],
    )

    elements = _build(provider, world, task_state)

    assert elements == [
        {"x": 0, "y": 0, "priority": 1.5},
        {"x": 0, "y": 0, "priority": 2.5},
    ]


@pytest.mark.parametrize(
    ("pois", "weights", "visited"),
    [
        ([[0.25, 0.25]], [1.0, 2.0], [False]),
        ([[0.25, 0.25]], [1.0], [False, True]),
        ([[0.25, 0.25], [0.75, 0.75]], [1.0], [False, False]),
    ],
)
def test_mismatched_state_lengths_raise_value_error(provider, world, pois, weights, visited):
    with pytest.raises(ValueError):
        _build(provider, world, _state(pois, weights, visited))


@pytest.mark.parametrize("missing_key", ["pois", "weights", "visited"])
def test_missing_required_state_raises_value_error(provider, world, missing_key):
    task_state = _state([[0.25, 0.25]], [1.0], [False])
    del task_state[missing_key]

    with pytest.raises(ValueError):
        _build(provider, world, task_state)


@pytest.mark.parametrize("weight", [-0.01, np.nan, np.inf, -np.inf])
def test_negative_or_non_finite_weight_raises_value_error(provider, world, weight):
    task_state = _state([[0.25, 0.25]], [weight], [False])

    with pytest.raises(ValueError):
        _build(provider, world, task_state)


@pytest.mark.parametrize(
    "poi",
    [[np.nan, 0.25], [0.25, np.inf], [-np.inf, 0.25]],
)
def test_non_finite_poi_coordinates_raise_value_error(provider, world, poi):
    task_state = _state([poi], [1.0], [False])

    with pytest.raises(ValueError):
        _build(provider, world, task_state)


@pytest.mark.parametrize(
    "poi",
    [[-0.01, 0.25], [2.01, 0.25], [0.25, -0.01], [0.25, 2.01]],
)
def test_world_space_out_of_bounds_poi_raises_before_clipping(provider, world, poi):
    task_state = _state([poi], [1.0], [False])

    with pytest.raises(ValueError):
        _build(provider, world, task_state)


def test_build_does_not_mutate_task_state_or_world(provider, world):
    task_state = _state(
        pois=[[0.25, 0.25], [1.25, 1.25]],
        weights=[1.0, 2.0],
        visited=[False, True],
    )
    original_state = deepcopy(task_state)
    original_world = deepcopy(world.__dict__)

    _build(provider, world, task_state)

    for key, original in original_state.items():
        if isinstance(original, np.ndarray):
            assert np.array_equal(task_state[key], original)
        else:
            assert task_state[key] == original
    assert world.__dict__ == original_world


def test_repeated_calls_are_deterministic(provider, world):
    task_state = _state(
        pois=[[1.25, 0.25], [0.25, 1.25]],
        weights=[3.0, 4.0],
        visited=[False, False],
    )

    first = _build(provider, world, task_state)
    second = _build(provider, world, task_state)

    assert first == second

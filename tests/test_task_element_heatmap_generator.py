from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "envs" / "waypoint" / "task_element_heatmap.py"
if not MODULE_PATH.exists():
    raise ModuleNotFoundError("No module named 'envs.waypoint.task_element_heatmap'")
MODULE_SPEC = importlib.util.spec_from_file_location("envs.waypoint.task_element_heatmap", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"Could not load module specification from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)
generate_task_element_heatmap = MODULE.generate_task_element_heatmap


def test_priority_channel_accumulates_cells_and_normalizes_by_maximum():
    heatmap = generate_task_element_heatmap(
        grid_size=4,
        task_elements=[
            {"x": 1, "y": 2, "priority": 2.0},
            {"x": 1, "y": 2, "priority": 1.0},
            {"x": 3, "y": 0, "priority": 6.0},
            {"x": 0, "y": 3, "priority": 1.5},
        ],
    )

    expected = np.zeros((1, 4, 4), dtype=np.float32)
    expected[0, 2, 1] = 0.5
    expected[0, 0, 3] = 1.0
    expected[0, 3, 0] = 0.25

    assert heatmap.shape == (1, 4, 4)
    assert heatmap.dtype == np.float32
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
    assert heatmap[0, 0, 3] == pytest.approx(1.0)
    np.testing.assert_allclose(heatmap, expected)


def test_empty_elements_return_an_all_zero_float32_heatmap():
    heatmap = generate_task_element_heatmap(grid_size=5, task_elements=[])

    assert heatmap.shape == (1, 5, 5)
    assert heatmap.dtype == np.float32
    assert np.array_equal(heatmap, np.zeros((1, 5, 5), dtype=np.float32))


@pytest.mark.parametrize(
    "element",
    [
        {"x": -1, "y": 0, "priority": 1.0},
        {"x": 4, "y": 0, "priority": 1.0},
        {"x": 0, "y": -1, "priority": 1.0},
        {"x": 0, "y": 4, "priority": 1.0},
    ],
)
def test_coordinates_outside_the_grid_raise_value_error(element):
    with pytest.raises(ValueError):
        generate_task_element_heatmap(grid_size=4, task_elements=[element])


def test_negative_priority_raises_value_error():
    with pytest.raises(ValueError):
        generate_task_element_heatmap(
            grid_size=4,
            task_elements=[{"x": 1, "y": 2, "priority": -0.01}],
        )


def test_zero_priority_is_allowed_and_contributes_zero():
    heatmap = generate_task_element_heatmap(
        grid_size=3,
        task_elements=[{"x": 1, "y": 1, "priority": 0.0}],
    )

    assert heatmap.dtype == np.float32
    assert np.array_equal(heatmap, np.zeros((1, 3, 3), dtype=np.float32))


def test_input_order_does_not_change_the_deterministic_result():
    elements = [
        {"x": 0, "y": 2, "priority": 4.0},
        {"x": 1, "y": 1, "priority": 2.0},
        {"x": 0, "y": 2, "priority": 1.0},
    ]

    first = generate_task_element_heatmap(grid_size=3, task_elements=elements)
    repeated = generate_task_element_heatmap(grid_size=3, task_elements=elements)
    reversed_order = generate_task_element_heatmap(
        grid_size=3,
        task_elements=list(reversed(elements)),
    )

    assert np.array_equal(first, repeated)
    assert np.array_equal(first, reversed_order)


@pytest.mark.parametrize("channels", [0, 2])
def test_only_one_semantic_channel_is_supported(channels):
    with pytest.raises(ValueError):
        generate_task_element_heatmap(
            grid_size=3,
            task_elements=[],
            channels=channels,
        )

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "envs" / "waypoint" / "task_element_provider.py"
if not MODULE_PATH.exists():
    raise ModuleNotFoundError("No module named 'envs.waypoint.task_element_provider'")
MODULE_SPEC = importlib.util.spec_from_file_location("envs.waypoint.task_element_provider", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"Could not load module specification from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

clear_task_element_provider_registry = MODULE.clear_task_element_provider_registry
register_task_element_provider = MODULE.register_task_element_provider
resolve_task_elements = MODULE.resolve_task_elements


@pytest.fixture(autouse=True)
def isolated_provider_registry():
    clear_task_element_provider_registry()
    yield
    clear_task_element_provider_registry()


def _inputs(static_task_elements):
    return {
        "task_name": "sample_task",
        "static_task_elements": static_task_elements,
        "scenario": object(),
        "world": object(),
        "task_state": {"marker": object()},
    }


@pytest.mark.parametrize(
    ("heatmap_enabled", "provider_enabled"),
    [(False, False), (False, True), (True, False)],
)
def test_disabled_selection_returns_static_elements_without_constructing_provider(
    heatmap_enabled,
    provider_enabled,
):
    calls = {"constructed": 0, "built": 0}

    class Provider:
        def build_task_elements(self, **kwargs):
            del kwargs
            calls["built"] += 1
            return []

    def factory():
        calls["constructed"] += 1
        return Provider()

    register_task_element_provider("sample_task", factory)
    static = [{"x": 2, "y": 1, "priority": 3.0}]

    resolved = resolve_task_elements(
        heatmap_enabled=heatmap_enabled,
        provider_enabled=provider_enabled,
        **_inputs(static),
    )

    assert resolved is static
    assert calls == {"constructed": 0, "built": 0}


def test_registered_provider_is_constructed_and_called_once_with_inputs_by_identity():
    calls = {"constructed": 0, "built": 0}
    received = {}
    provider_elements = [{"x": 3, "y": 2, "priority": 7.0}]

    class Provider:
        def build_task_elements(self, *, scenario, world, task_state):
            calls["built"] += 1
            received.update(scenario=scenario, world=world, task_state=task_state)
            return provider_elements

    def factory():
        calls["constructed"] += 1
        return Provider()

    register_task_element_provider("sample_task", factory)
    static = [{"x": 0, "y": 0, "priority": 1.0}]
    inputs = _inputs(static)

    resolved = resolve_task_elements(
        heatmap_enabled=True,
        provider_enabled=True,
        **inputs,
    )

    assert resolved is provider_elements
    assert resolved is not static
    assert calls == {"constructed": 1, "built": 1}
    assert received["scenario"] is inputs["scenario"]
    assert received["world"] is inputs["world"]
    assert received["task_state"] is inputs["task_state"]


def test_empty_registered_provider_result_does_not_fall_back_to_static_elements():
    provider_elements = []

    class Provider:
        def build_task_elements(self, **kwargs):
            del kwargs
            return provider_elements

    register_task_element_provider("sample_task", Provider)
    static = [{"x": 0, "y": 0, "priority": 1.0}]

    resolved = resolve_task_elements(
        heatmap_enabled=True,
        provider_enabled=True,
        **_inputs(static),
    )

    assert resolved is provider_elements
    assert resolved == []


def test_unregistered_task_returns_static_elements_without_error():
    static = [{"x": 1, "y": 1, "priority": 2.0}]

    resolved = resolve_task_elements(
        heatmap_enabled=True,
        provider_enabled=True,
        **_inputs(static),
    )

    assert resolved is static


def test_provider_output_is_returned_without_semantic_modification():
    raw_elements = [
        {"x": 9, "y": -1, "priority": -2.0},
        {"x": 1, "y": 1, "priority": 4.0},
        {"x": 1, "y": 1, "priority": 1.0},
    ]

    class Provider:
        def build_task_elements(self, **kwargs):
            del kwargs
            return raw_elements

    register_task_element_provider("sample_task", Provider)

    resolved = resolve_task_elements(
        heatmap_enabled=True,
        provider_enabled=True,
        **_inputs([{"x": 0, "y": 0, "priority": 1.0}]),
    )

    assert resolved is raw_elements
    assert resolved == raw_elements


def test_repeated_resolution_with_deterministic_provider_is_equivalent():
    class Provider:
        def build_task_elements(self, **kwargs):
            del kwargs
            return [{"x": 2, "y": 0, "priority": 5.0}]

    register_task_element_provider("sample_task", Provider)
    inputs = _inputs([])

    first = resolve_task_elements(heatmap_enabled=True, provider_enabled=True, **inputs)
    second = resolve_task_elements(heatmap_enabled=True, provider_enabled=True, **inputs)

    assert first == second


def test_registry_rejects_empty_and_duplicate_task_names():
    with pytest.raises(ValueError):
        register_task_element_provider("", lambda: object())

    register_task_element_provider("sample_task", lambda: object())
    with pytest.raises(ValueError):
        register_task_element_provider("sample_task", lambda: object())


def test_clear_registry_removes_registered_provider():
    calls = {"constructed": 0}

    def factory():
        calls["constructed"] += 1
        return object()

    register_task_element_provider("sample_task", factory)
    clear_task_element_provider_registry()
    static = [{"x": 0, "y": 0, "priority": 1.0}]

    resolved = resolve_task_elements(
        heatmap_enabled=True,
        provider_enabled=True,
        **_inputs(static),
    )

    assert resolved is static
    assert calls["constructed"] == 0

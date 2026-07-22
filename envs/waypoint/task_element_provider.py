from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol


TaskElement = Mapping[str, float]


class TaskElementProvider(Protocol):
    """Build raw task elements from the active scenario state."""

    def build_task_elements(
        self,
        *,
        scenario: Any,
        world: Any,
        task_state: Mapping[str, Any],
    ) -> Sequence[TaskElement]: ...


_PROVIDER_FACTORIES: dict[str, Callable[[], TaskElementProvider]] = {}


def register_task_element_provider(
    task_name: str,
    provider_factory: Callable[[], TaskElementProvider],
) -> None:
    """Register one provider factory for a task name."""
    if not task_name:
        raise ValueError("task_name must not be empty")
    if task_name in _PROVIDER_FACTORIES:
        raise ValueError(f"task-element provider already registered for {task_name!r}")
    _PROVIDER_FACTORIES[task_name] = provider_factory


def clear_task_element_provider_registry() -> None:
    """Remove all task-element provider registrations."""
    _PROVIDER_FACTORIES.clear()


def resolve_task_elements(
    *,
    heatmap_enabled: bool,
    provider_enabled: bool,
    task_name: str,
    static_task_elements: Sequence[TaskElement],
    scenario: Any,
    world: Any,
    task_state: Mapping[str, Any],
) -> Sequence[TaskElement]:
    """Select static elements or the registered provider result unchanged."""
    if not heatmap_enabled or not provider_enabled:
        return static_task_elements
    provider_factory = _PROVIDER_FACTORIES.get(task_name)
    if provider_factory is None:
        return static_task_elements
    provider = provider_factory()
    return provider.build_task_elements(
        scenario=scenario,
        world=world,
        task_state=task_state,
    )


__all__ = [
    "TaskElement",
    "TaskElementProvider",
    "clear_task_element_provider_registry",
    "register_task_element_provider",
    "resolve_task_elements",
]

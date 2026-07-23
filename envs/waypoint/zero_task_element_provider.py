from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ZeroTaskElementProvider:
    """Provide no semantic elements for a zero-input capacity control."""

    def build_task_elements(
        self,
        *,
        scenario: Any,
        world: Any,
        task_state: Mapping[str, Any],
    ) -> Sequence[Mapping[str, float]]:
        del scenario, world, task_state
        return []


def register_zero_task_element_provider() -> None:
    """Register the zero provider for priority-inspection experiments."""
    from envs.waypoint.task_element_provider import register_task_element_provider

    task_name = "priority_inspection"
    try:
        register_task_element_provider(task_name, ZeroTaskElementProvider)
    except ValueError as exc:
        duplicate_error = f"task-element provider already registered for {task_name!r}"
        if str(exc) != duplicate_error:
            raise


__all__ = ["ZeroTaskElementProvider", "register_zero_task_element_provider"]

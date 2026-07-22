from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class PriorityInspectionTaskElementProvider:
    """Convert unvisited inspection POIs into raw grid-space task elements."""

    def build_task_elements(
        self,
        *,
        scenario: Any,
        world: Any,
        task_state: Mapping[str, Any],
    ) -> Sequence[Mapping[str, float]]:
        del scenario
        required = ("pois", "weights", "visited")
        if any(key not in task_state for key in required):
            raise ValueError("task_state must contain pois, weights, and visited")

        try:
            pois = np.asarray(task_state["pois"], dtype=np.float64)
            weights = np.asarray(task_state["weights"], dtype=np.float64)
            visited = np.asarray(task_state["visited"])
        except (TypeError, ValueError) as exc:
            raise ValueError("priority-inspection task state must be numeric") from exc

        if pois.size == 0:
            pois = np.empty((0, 2), dtype=np.float64)
        if pois.ndim != 2 or pois.shape[1] != 2:
            raise ValueError("pois must have shape [N, 2]")
        if weights.ndim != 1 or visited.ndim != 1:
            raise ValueError("weights and visited must have shape [N]")
        if len(pois) != len(weights) or len(pois) != len(visited):
            raise ValueError("pois, weights, and visited must have equal lengths")
        if not np.isfinite(pois).all():
            raise ValueError("POI coordinates must be finite")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("POI weights must be finite and non-negative")

        bounds = getattr(world, "geofence", None)
        if bounds is None:
            map_size = float(world.map_size)
            bounds = (0.0, map_size, 0.0, map_size)
        x_min, x_max, y_min, y_max = map(float, bounds)
        in_bounds = (
            (pois[:, 0] >= x_min)
            & (pois[:, 0] <= x_max)
            & (pois[:, 1] >= y_min)
            & (pois[:, 1] <= y_max)
        )
        if not bool(in_bounds.all()):
            raise ValueError("POI coordinates are outside the world bounds")

        elements: list[Mapping[str, float]] = []
        for poi, weight, is_visited in zip(pois, weights, visited.astype(bool)):
            if is_visited:
                continue
            grid_x, grid_y = world.world_to_grid(poi[None, :])[0]
            elements.append(
                {
                    "x": int(grid_x),
                    "y": int(grid_y),
                    "priority": float(weight),
                }
            )
        return elements


def register_priority_inspection_task_element_provider() -> None:
    """Register the priority-inspection provider if it is not registered."""
    from envs.waypoint.task_element_provider import register_task_element_provider

    task_name = "priority_inspection"
    try:
        register_task_element_provider(task_name, PriorityInspectionTaskElementProvider)
    except ValueError as exc:
        duplicate_error = f"task-element provider already registered for {task_name!r}"
        if str(exc) != duplicate_error:
            raise


__all__ = [
    "PriorityInspectionTaskElementProvider",
    "register_priority_inspection_task_element_provider",
]

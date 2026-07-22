from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def generate_task_element_heatmap(
    grid_size: int,
    task_elements: Sequence[Mapping[str, float]],
    channels: int = 1,
) -> np.ndarray:
    """Build one normalized spatial-priority channel from grid-space elements."""
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    if channels != 1:
        raise ValueError("exactly one task-element heatmap channel is supported")

    validated: list[tuple[int, int, float]] = []
    for element in task_elements:
        try:
            raw_x = element["x"]
            raw_y = element["y"]
            priority = float(element["priority"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each task element must provide numeric x, y, and priority") from exc

        if isinstance(raw_x, bool) or isinstance(raw_y, bool):
            raise ValueError("task-element coordinates must be integers")
        try:
            x = int(raw_x)
            y = int(raw_y)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("task-element coordinates must be integers") from exc
        if x != raw_x or y != raw_y:
            raise ValueError("task-element coordinates must be integers")
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            raise ValueError("task-element coordinates are outside the grid")
        if not np.isfinite(priority) or priority < 0.0:
            raise ValueError("task-element priority must be finite and non-negative")
        validated.append((y, x, priority))

    heatmap = np.zeros((1, grid_size, grid_size), dtype=np.float64)
    for y, x, priority in sorted(validated):
        heatmap[0, y, x] += priority

    maximum = float(heatmap.max())
    if maximum > 0.0:
        heatmap /= maximum
    return np.clip(heatmap, 0.0, 1.0).astype(np.float32)


__all__ = ["generate_task_element_heatmap"]

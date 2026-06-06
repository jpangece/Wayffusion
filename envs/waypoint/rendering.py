from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from envs.waypoint.world import MissionWorld


def render_world(
    world: MissionWorld,
    task_name: str,
    task_state: dict | None = None,
    candidate_waypoints: np.ndarray | None = None,
    selected_waypoints: np.ndarray | None = None,
    metrics: dict | None = None,
) -> np.ndarray:
    task_state = task_state or {}
    metrics = metrics or {}
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=120)
    extent = [0.0, world.map_size, 0.0, world.map_size]
    if "belief" in task_name:
        ax.imshow(world.belief_grid, origin="lower", extent=extent, cmap="YlOrRd", alpha=0.5)
    elif task_name in {"area_coverage", "connectivity_expansion"}:
        ax.imshow(world.coverage_grid, origin="lower", extent=extent, cmap="Greens", alpha=0.45)
    elif "target" in task_name:
        ax.imshow(world.coverage_grid, origin="lower", extent=extent, cmap="Blues", alpha=0.25)
    else:
        ax.imshow(world.visit_count_grid, origin="lower", extent=extent, cmap="viridis", alpha=0.25)

    for zone in world.no_fly_zones:
        if "center" in zone and "radius" in zone:
            circle = plt.Circle(zone["center"], zone["radius"], color="black", alpha=0.25)
            ax.add_patch(circle)
        elif {"x_min", "x_max", "y_min", "y_max"}.issubset(zone):
            rect = plt.Rectangle(
                (zone["x_min"], zone["y_min"]),
                zone["x_max"] - zone["x_min"],
                zone["y_max"] - zone["y_min"],
                color="black",
                alpha=0.25,
            )
            ax.add_patch(rect)

    ax.scatter([world.base_position[0]], [world.base_position[1]], marker="s", c="tab:blue", s=90, label="base")
    positions = world.get_uav_positions()
    if len(positions):
        connected = world.connected_to_base_mask()
        colors = ["tab:green" if flag else "tab:red" for flag in connected]
        ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=60, label="uav")
        for idx, uav in enumerate(world.uavs):
            traj = np.asarray(uav.trajectory, dtype=np.float32)
            if len(traj) > 1:
                ax.plot(traj[:, 0], traj[:, 1], color=colors[idx], linewidth=1.0, alpha=0.75)
            if np.linalg.norm(uav.velocity) > 1e-8:
                ax.arrow(
                    uav.position[0],
                    uav.position[1],
                    uav.velocity[0],
                    uav.velocity[1],
                    color=colors[idx],
                    width=0.001,
                    head_width=0.012,
                    alpha=0.65,
                    length_includes_head=True,
                )
            ax.text(uav.position[0], uav.position[1], str(idx), fontsize=8, ha="center", va="center", color="white")

    if candidate_waypoints is not None:
        cand = np.asarray(candidate_waypoints)
        if cand.ndim == 3:
            ax.scatter(cand[..., 0].reshape(-1), cand[..., 1].reshape(-1), c="gray", s=8, alpha=0.25)
    if selected_waypoints is not None:
        sel = np.asarray(selected_waypoints)
        ax.scatter(sel[:, 0], sel[:, 1], marker="x", c="tab:orange", s=50, label="selected")
        for pos, target in zip(positions, sel):
            ax.plot([pos[0], target[0]], [pos[1], target[1]], color="tab:orange", linewidth=0.8, alpha=0.6)

    if "pois" in task_state:
        pois = np.asarray(task_state["pois"], dtype=np.float32)
        visited = np.asarray(task_state.get("visited", np.zeros(len(pois), dtype=bool)), dtype=bool)
        ax.scatter(pois[~visited, 0], pois[~visited, 1], marker="*", c="gold", s=120, edgecolors="black", label="poi")
        if visited.any():
            ax.scatter(pois[visited, 0], pois[visited, 1], marker="*", c="lightgray", s=90, edgecolors="black")
    if "target_position" in task_state:
        target = np.asarray(task_state["target_position"], dtype=np.float32)
        ax.scatter([target[0]], [target[1]], marker="D", c="crimson", s=80, label="target")
        route = np.asarray(task_state.get("target_route", []), dtype=np.float32)
        if route.ndim == 2 and len(route) > 1:
            ax.plot(route[:, 0], route[:, 1], color="crimson", linestyle="--", linewidth=1.0, alpha=0.6)

    graph = world.communication_graph
    if graph.shape[0] == len(world.uavs) + 1:
        nodes = np.vstack([world.base_position[None, :], positions]) if len(positions) else world.base_position[None, :]
        for i in range(graph.shape[0]):
            for j in range(i + 1, graph.shape[1]):
                if graph[i, j]:
                    ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], color="tab:blue", alpha=0.25)

    score = metrics.get("coverage_ratio", metrics.get("success", ""))
    ax.set_title(f"{task_name} | step {world.step_count}/{world.max_steps} | {score}")
    ax.set_xlim(0.0, world.map_size)
    ax.set_ylim(0.0, world.map_size)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    if hasattr(fig.canvas, "tostring_rgb"):
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    else:
        image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return image

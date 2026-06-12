from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np

from envs.waypoint.world import MissionWorld


WAYF_COLORS = {
    "paper": "#F5F0E3",
    "ink": "#263238",
    "grid": "#D8D0BD",
    "blue": "#276FBF",
    "teal": "#158A8A",
    "green": "#2F9E44",
    "orange": "#E1772D",
    "red": "#C2410C",
    "gold": "#F2B705",
    "gray": "#6B7280",
    "white": "#FFFFFF",
}
UAV_PALETTE = ["#008C8C", "#E1772D", "#2F6FED", "#B13B5E", "#6C8E2E", "#7C5CFF", "#BF7A00", "#0083A3"]
COVERAGE_CMAP = LinearSegmentedColormap.from_list("wayf_coverage", ["#F5F0E3", "#B7E4C7", "#2D6A4F"])
BELIEF_CMAP = LinearSegmentedColormap.from_list("wayf_belief", ["#F5F0E3", "#FFD166", "#D62828"])
VISIT_CMAP = LinearSegmentedColormap.from_list("wayf_visit", ["#F5F0E3", "#9AD7D1", "#205375"])


def _with_outline(text, width: float = 2.0):
    text.set_path_effects([patheffects.withStroke(linewidth=width, foreground=WAYF_COLORS["white"])])
    return text


def _draw_task_layer(ax, world: MissionWorld, task_name: str, extent: list[float]) -> None:
    ax.set_facecolor(WAYF_COLORS["paper"])
    if "belief" in task_name:
        layer = np.ma.masked_where(world.belief_grid <= 1e-7, world.belief_grid)
        ax.imshow(layer, origin="lower", extent=extent, cmap=BELIEF_CMAP, alpha=0.74, interpolation="bilinear")
    elif task_name in {"area_coverage", "connectivity_expansion"}:
        layer = np.ma.masked_where(world.coverage_grid <= 1e-7, world.coverage_grid)
        ax.imshow(layer, origin="lower", extent=extent, cmap=COVERAGE_CMAP, alpha=0.70, interpolation="nearest")
    elif "target" in task_name:
        ax.imshow(world.coverage_grid, origin="lower", extent=extent, cmap=VISIT_CMAP, alpha=0.30, interpolation="nearest")
    else:
        layer = np.ma.masked_where(world.visit_count_grid <= 0, world.visit_count_grid)
        ax.imshow(layer, origin="lower", extent=extent, cmap=VISIT_CMAP, alpha=0.42, interpolation="nearest")
    ax.grid(color=WAYF_COLORS["grid"], linewidth=0.45, alpha=0.55)


def _draw_no_fly_zones(ax, world: MissionWorld) -> None:
    for zone in world.no_fly_zones:
        if "center" in zone and "radius" in zone:
            circle = Circle(
                zone["center"],
                zone["radius"],
                facecolor=WAYF_COLORS["red"],
                edgecolor="#7F1D1D",
                linewidth=1.35,
                alpha=0.18,
                hatch="///",
                zorder=3,
            )
            ax.add_patch(circle)
            _with_outline(
                ax.text(zone["center"][0], zone["center"][1], "NO-FLY", fontsize=6.5, ha="center", va="center", color="#7F1D1D", zorder=4),
                1.4,
            )
        elif {"x_min", "x_max", "y_min", "y_max"}.issubset(zone):
            rect = Rectangle(
                (zone["x_min"], zone["y_min"]),
                zone["x_max"] - zone["x_min"],
                zone["y_max"] - zone["y_min"],
                facecolor=WAYF_COLORS["red"],
                edgecolor="#7F1D1D",
                linewidth=1.35,
                alpha=0.18,
                hatch="///",
                zorder=3,
            )
            ax.add_patch(rect)


def _draw_drone_icon(ax, position: np.ndarray, heading: float, color: str, agent_id: int, connected: bool, scale: float) -> None:
    center = np.asarray(position, dtype=np.float32)
    c, s = float(np.cos(heading)), float(np.sin(heading))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    arm = float(scale)
    motor_r = arm * 0.22
    body_r = arm * 0.18
    arm_points = [
        np.asarray([[-arm, 0.0], [arm, 0.0]], dtype=np.float32),
        np.asarray([[0.0, -arm], [0.0, arm]], dtype=np.float32),
    ]
    halo = Circle(center, arm * 1.34, facecolor=color, edgecolor="none", alpha=0.14 if connected else 0.22, zorder=7)
    ax.add_patch(halo)
    for segment in arm_points:
        points = segment @ rot.T + center
        ax.plot(points[:, 0], points[:, 1], color="#23313D", linewidth=2.4, alpha=0.92, solid_capstyle="round", zorder=8)
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.15, alpha=0.95, solid_capstyle="round", zorder=9)
    motor_offsets = np.asarray([[-arm, 0.0], [arm, 0.0], [0.0, -arm], [0.0, arm]], dtype=np.float32) @ rot.T
    for offset in motor_offsets:
        motor = Circle(center + offset, motor_r, facecolor=WAYF_COLORS["white"], edgecolor="#23313D", linewidth=1.0, zorder=10)
        ax.add_patch(motor)
        ax.add_patch(Circle(center + offset, motor_r * 0.48, facecolor=color, edgecolor="none", alpha=0.88, zorder=11))
    ax.add_patch(Circle(center, body_r * 1.25, facecolor="#263238", edgecolor=WAYF_COLORS["white"], linewidth=1.0, zorder=12))
    nose = (np.asarray([[body_r * 2.1, 0.0], [body_r * 0.35, body_r * 0.95], [body_r * 0.35, -body_r * 0.95]], dtype=np.float32) @ rot.T) + center
    ax.add_patch(Polygon(nose, closed=True, facecolor=color, edgecolor=WAYF_COLORS["white"], linewidth=0.75, zorder=13))
    label_color = WAYF_COLORS["white"] if connected else "#7F1D1D"
    label = ax.text(center[0], center[1] - arm * 1.62, f"U{agent_id}", fontsize=6.5, ha="center", va="center", color=label_color, weight="bold", zorder=14)
    _with_outline(label, 1.6)


def render_world(
    world: MissionWorld,
    task_name: str,
    task_state: dict | None = None,
    candidate_waypoints: np.ndarray | None = None,
    selected_waypoints: np.ndarray | None = None,
    metrics: dict | None = None,
    render_detail: str = "full",
) -> np.ndarray:
    task_state = task_state or {}
    metrics = metrics or {}
    detail = str(render_detail or "full").lower()
    low_detail = detail == "low"
    fig, ax = plt.subplots(figsize=(4.8, 4.8) if low_detail else (6.0, 6.0), dpi=90 if low_detail else 130)
    fig.patch.set_facecolor("#EFE7D2")
    extent = [0.0, world.map_size, 0.0, world.map_size]
    _draw_task_layer(ax, world, task_name, extent)
    _draw_no_fly_zones(ax, world)

    base = world.base_position
    ax.scatter([base[0]], [base[1]], marker="h", c=WAYF_COLORS["blue"], s=70 if low_detail else 190, edgecolors=WAYF_COLORS["white"], linewidths=1.0 if low_detail else 1.6, label="base", zorder=9)
    if not low_detail:
        _with_outline(ax.text(base[0], base[1] - 0.035 * world.map_size, "BASE", fontsize=7, ha="center", va="top", color=WAYF_COLORS["blue"], weight="bold", zorder=10), 1.5)
    positions = world.get_uav_positions()
    if len(positions):
        connected = world.connected_to_base_mask()
        colors = [UAV_PALETTE[idx % len(UAV_PALETTE)] if flag else WAYF_COLORS["red"] for idx, flag in enumerate(connected)]
        if low_detail:
            if any(len(uav.trajectory) > 1 for uav in world.uavs):
                for idx, uav in enumerate(world.uavs):
                    traj = np.asarray(uav.trajectory[-16:], dtype=np.float32)
                    if len(traj) > 1:
                        ax.plot(traj[:, 0], traj[:, 1], color=colors[idx], linewidth=0.8, alpha=0.45, zorder=5)
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=colors,
                s=20,
                edgecolors=WAYF_COLORS["white"],
                linewidths=0.45,
                zorder=10,
            )
        else:
            for idx, uav in enumerate(world.uavs):
                traj = np.asarray(uav.trajectory, dtype=np.float32)
                if len(traj) > 1:
                    ax.plot(traj[:, 0], traj[:, 1], color=colors[idx], linewidth=1.65, alpha=0.74, zorder=5)
                if np.linalg.norm(uav.velocity) > 1e-8:
                    heading = float(np.arctan2(uav.velocity[1], uav.velocity[0]))
                elif selected_waypoints is not None:
                    target_delta = np.asarray(selected_waypoints, dtype=np.float32)[idx] - uav.position
                    heading = float(np.arctan2(target_delta[1], target_delta[0])) if np.linalg.norm(target_delta) > 1e-8 else float(idx) * 0.7
                else:
                    heading = float(idx) * 0.7
                _draw_drone_icon(ax, uav.position, heading, colors[idx], idx, bool(connected[idx]), 0.026 * world.map_size)
                if np.linalg.norm(uav.velocity) > 1e-8:
                    ax.arrow(uav.position[0], uav.position[1], uav.velocity[0], uav.velocity[1], color=colors[idx], width=0.0006, head_width=0.012, alpha=0.52, length_includes_head=True, zorder=6)

    if candidate_waypoints is not None and not low_detail:
        cand = np.asarray(candidate_waypoints)
        if cand.ndim == 3:
            ax.scatter(cand[..., 0].reshape(-1), cand[..., 1].reshape(-1), c=WAYF_COLORS["gray"], s=7, alpha=0.18, linewidths=0, label="candidates", zorder=4)
    if selected_waypoints is not None:
        sel = np.asarray(selected_waypoints)
        ax.scatter(sel[:, 0], sel[:, 1], marker="D", c=WAYF_COLORS["orange"], s=14 if low_detail else 44, edgecolors=WAYF_COLORS["white"], linewidths=0.45 if low_detail else 0.9, label="waypoint", zorder=8)
        if not low_detail:
            for pos, target in zip(positions, sel):
                ax.plot([pos[0], target[0]], [pos[1], target[1]], color=WAYF_COLORS["orange"], linewidth=1.1, alpha=0.62, linestyle=(0, (3, 2)), zorder=6)

    if "pois" in task_state:
        pois = np.asarray(task_state["pois"], dtype=np.float32)
        visited = np.asarray(task_state.get("visited", np.zeros(len(pois), dtype=bool)), dtype=bool)
        ax.scatter(pois[~visited, 0], pois[~visited, 1], marker="*", c=WAYF_COLORS["gold"], s=140, edgecolors="#3A2D00", linewidths=0.9, label="poi", zorder=8)
        if visited.any():
            ax.scatter(pois[visited, 0], pois[visited, 1], marker="*", c="#B9B7AD", s=95, edgecolors="#525252", linewidths=0.6, zorder=7)
    target_positions = task_state.get("target_positions")
    if target_positions is None and "target_position" in task_state:
        target_positions = np.asarray(task_state["target_position"], dtype=np.float32)[None, :]
    if target_positions is not None:
        targets = np.asarray(target_positions, dtype=np.float32).reshape(-1, 2)
        ax.scatter(
            targets[:, 0],
            targets[:, 1],
            marker="D",
            c=WAYF_COLORS["red"],
            s=95,
            edgecolors=WAYF_COLORS["white"],
            linewidths=1.0,
            label="target",
            zorder=8,
        )
        for idx, target in enumerate(targets):
            if not low_detail:
                _with_outline(
                    ax.text(target[0], target[1] + 0.025 * world.map_size, f"T{idx}", fontsize=6.5, ha="center", color=WAYF_COLORS["red"], zorder=9),
                    1.3,
                )
        routes = task_state.get("target_routes")
        if routes is None:
            routes = [task_state.get("target_route", [])]
        for route in routes:
            route_array = np.asarray(route, dtype=np.float32)
            if route_array.ndim == 2 and len(route_array) > 1:
                ax.plot(route_array[:, 0], route_array[:, 1], color=WAYF_COLORS["red"], linestyle="--", linewidth=1.2, alpha=0.58, zorder=5)

    graph = world.communication_graph
    if graph.shape[0] == len(world.uavs) + 1 and not low_detail:
        nodes = np.vstack([world.base_position[None, :], positions]) if len(positions) else world.base_position[None, :]
        for i in range(graph.shape[0]):
            for j in range(i + 1, graph.shape[1]):
                if graph[i, j]:
                    ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], color=WAYF_COLORS["blue"], alpha=0.28, linewidth=1.05, zorder=4)

    score = metrics.get("coverage_ratio", metrics.get("searched_probability_mass", metrics.get("success", "")))
    hud = f"{task_name.replace('_', ' ').title()}   step {world.step_count}/{world.max_steps}"
    if score != "":
        hud += f"   score {score:.3f}" if isinstance(score, (int, float, np.integer, np.floating, bool)) else f"   score {score}"
    ax.set_title(hud, fontsize=10, color=WAYF_COLORS["ink"], weight="bold", pad=8)
    ax.set_xlim(0.0, world.map_size)
    ax.set_ylim(0.0, world.map_size)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color("#A99F88")
        spine.set_linewidth(1.2)
    ax.tick_params(labelsize=0 if low_detail else 7, colors="#746B5C")
    if low_detail:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.legend(loc="upper right", fontsize=6, frameon=True, framealpha=0.86, facecolor=WAYF_COLORS["paper"], edgecolor="#C7BCA3")
    fig.tight_layout(pad=0.65)
    fig.canvas.draw()
    if hasattr(fig.canvas, "tostring_rgb"):
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    else:
        image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return image

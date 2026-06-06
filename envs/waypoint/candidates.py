from __future__ import annotations

import numpy as np

from envs.waypoint.world import MissionWorld


def radial_candidates_around_agent(world: MissionWorld, num_candidates: int, radius: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    positions = world.get_uav_positions()
    n = len(positions)
    k = int(num_candidates)
    radius = float(radius if radius is not None else world.max_waypoint_distance)
    angles = np.linspace(0.0, 2.0 * np.pi, max(k - 1, 1), endpoint=False)
    candidates = np.zeros((n, k, 2), dtype=np.float32)
    candidates[:, 0, :] = positions
    for idx, angle in enumerate(angles, start=1):
        if idx >= k:
            break
        offset = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32) * radius
        candidates[:, idx, :] = positions + offset[None, :]
    return world.project_to_geofence(candidates), np.ones((n, k), dtype=bool)


def _top_grid_points(world: MissionWorld, score_grid: np.ndarray, limit: int) -> np.ndarray:
    flat = np.asarray(score_grid, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    limit = min(int(limit), flat.size)
    top = np.argpartition(-flat, np.arange(limit))[:limit]
    rows = top // world.grid_size
    cols = top % world.grid_size
    coords = np.stack([cols, rows], axis=-1)
    return (coords.astype(np.float32) + 0.5) / float(world.grid_size) * world.map_size


def grid_frontier_candidates(world: MissionWorld, num_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    unvisited = 1.0 - world.coverage_grid
    return merge_and_pad_candidates(world, [radial_candidates_around_agent(world, 4)[0], _repeat_points(world, _top_grid_points(world, unvisited, num_candidates))], num_candidates)


def coverage_candidates(world: MissionWorld, num_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    score = (1.0 - world.coverage_grid) / (1.0 + world.visit_count_grid)
    return merge_and_pad_candidates(world, [radial_candidates_around_agent(world, 5)[0], _repeat_points(world, _top_grid_points(world, score, num_candidates))], num_candidates)


def belief_candidates(world: MissionWorld, num_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    score = world.belief_grid * (1.0 - np.clip(world.visit_count_grid, 0.0, 1.0) * 0.35)
    return merge_and_pad_candidates(world, [radial_candidates_around_agent(world, 5)[0], _repeat_points(world, _top_grid_points(world, score, num_candidates))], num_candidates)


def poi_candidates(world: MissionWorld, poi_positions: np.ndarray, num_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    return merge_and_pad_candidates(world, [radial_candidates_around_agent(world, 4)[0], _repeat_points(world, poi_positions)], num_candidates)


def connectivity_boundary_candidates(world: MissionWorld, num_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    positions = world.get_uav_positions()
    n = len(positions)
    k = int(num_candidates)
    candidates = np.zeros((n, k, 2), dtype=np.float32)
    candidates[:, 0, :] = positions
    base = world.base_position
    for i, pos in enumerate(positions):
        direction = pos - base
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm
        lateral = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        offsets = [direction, 0.7 * direction + 0.5 * lateral, 0.7 * direction - 0.5 * lateral, -direction, lateral, -lateral]
        for j in range(1, k):
            candidates[i, j] = pos + offsets[(j - 1) % len(offsets)] * world.max_waypoint_distance
    return world.project_to_geofence(candidates), np.ones((n, k), dtype=bool)


def target_prediction_candidates(
    world: MissionWorld,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    num_candidates: int,
    ring_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(target_position, dtype=np.float32) + np.asarray(target_velocity, dtype=np.float32) * world.dt_decision * 2.0
    angles = np.linspace(0.0, 2.0 * np.pi, max(num_candidates - 1, 1), endpoint=False)
    points = [pred]
    for angle in angles:
        points.append(pred + ring_radius * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32))
    return merge_and_pad_candidates(world, [radial_candidates_around_agent(world, 4)[0], _repeat_points(world, np.asarray(points, dtype=np.float32))], num_candidates)


def _repeat_points(world: MissionWorld, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    n = len(world.uavs)
    if points.ndim == 2:
        return np.repeat(points[None, :, :], n, axis=0)
    return points


def merge_and_pad_candidates(
    world: MissionWorld,
    candidate_sets: list[np.ndarray | tuple[np.ndarray, np.ndarray]],
    num_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(world.uavs)
    merged = []
    masks = []
    for item in candidate_sets:
        if isinstance(item, tuple):
            points, mask = item
        else:
            points = item
            mask = np.ones(points.shape[:2], dtype=bool)
        if points.shape[0] != n:
            points = np.repeat(points[:1], n, axis=0)
            mask = np.repeat(mask[:1], n, axis=0)
        merged.append(np.asarray(points, dtype=np.float32))
        masks.append(np.asarray(mask, dtype=bool))
    if merged:
        points_all = np.concatenate(merged, axis=1)
        mask_all = np.concatenate(masks, axis=1)
    else:
        points_all, mask_all = radial_candidates_around_agent(world, num_candidates)
    k = int(num_candidates)
    out = np.zeros((n, k, 2), dtype=np.float32)
    out_mask = np.zeros((n, k), dtype=bool)
    current = world.get_uav_positions()
    for agent_idx in range(n):
        unique = []
        unique_mask = []
        seen = set()
        for cand, valid in zip(points_all[agent_idx], mask_all[agent_idx]):
            key = tuple(np.round(cand, 4))
            if key in seen:
                continue
            seen.add(key)
            unique.append(cand)
            unique_mask.append(bool(valid))
            if len(unique) >= k:
                break
        while len(unique) < k:
            unique.append(current[agent_idx])
            unique_mask.append(True if len(unique) == 1 else False)
        out[agent_idx] = np.asarray(unique[:k], dtype=np.float32)
        out_mask[agent_idx] = np.asarray(unique_mask[:k], dtype=bool)
    return world.project_to_geofence(out), out_mask

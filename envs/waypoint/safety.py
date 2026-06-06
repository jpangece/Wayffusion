from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.waypoint.world import MissionWorld


@dataclass
class SafetyResult:
    safe_waypoints: np.ndarray
    valid_mask: np.ndarray
    clipped_mask: np.ndarray
    rejected_mask: np.ndarray
    geofence_violation_mask: np.ndarray
    no_fly_violation_mask: np.ndarray
    distance_violation_mask: np.ndarray
    connectivity_violation_mask: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def to_info(self) -> dict[str, Any]:
        return {
            "safe_waypoints": self.safe_waypoints.astype(np.float32),
            "valid_mask": self.valid_mask.astype(bool),
            "clipped_mask": self.clipped_mask.astype(bool),
            "rejected_mask": self.rejected_mask.astype(bool),
            "geofence_violation_mask": self.geofence_violation_mask.astype(bool),
            "no_fly_violation_mask": self.no_fly_violation_mask.astype(bool),
            "distance_violation_mask": self.distance_violation_mask.astype(bool),
            "connectivity_violation_mask": self.connectivity_violation_mask.astype(bool),
            "geofence_violation_count": int(self.geofence_violation_mask.sum()),
            "no_fly_violation_count": int(self.no_fly_violation_mask.sum()),
            "distance_violation_count": int(self.distance_violation_mask.sum()),
            "connectivity_action_rejected_count": int(self.connectivity_violation_mask.sum()),
            "clipped_waypoint_count": int(self.clipped_mask.sum()),
            "rejected_waypoint_count": int(self.rejected_mask.sum()),
            **dict(self.info),
        }


class SafetyLayer:
    """Safety checks for final waypoints plus optional candidate-filter helpers."""

    def validate_waypoints(
        self,
        world: MissionWorld,
        waypoints: np.ndarray,
        config: dict[str, Any] | None = None,
        require_connectivity: bool = False,
    ) -> SafetyResult:
        cfg = dict(config or {})
        clip_to_geofence = bool(cfg.get("clip_to_geofence", True))
        reject_no_fly = bool(cfg.get("reject_no_fly_waypoints", True))
        strict = bool(cfg.get("strict_action_validation", False))
        max_command_distance = float(cfg.get("max_command_distance", world.max_waypoint_distance))
        selected = np.asarray(waypoints, dtype=np.float32).copy()
        current = world.get_uav_positions()
        if selected.shape != current.shape:
            raise ValueError(f"waypoints must have shape {current.shape}, got {selected.shape}")

        original = selected.copy()
        geofence_ok_before = world.check_geofence(selected)
        geofence_violation = ~geofence_ok_before
        clipped_mask = np.zeros(len(selected), dtype=bool)
        rejected = np.zeros(len(selected), dtype=bool)

        if geofence_violation.any():
            if strict:
                raise ValueError(f"Waypoint geofence violation: {np.where(geofence_violation)[0].tolist()}")
            if clip_to_geofence:
                selected = world.project_to_geofence(selected)
                clipped_mask |= np.linalg.norm(selected - original, axis=-1) > 1e-8
            else:
                selected[geofence_violation] = self._fallback_waypoints(world)[geofence_violation]
                rejected |= geofence_violation

        no_fly_violation = ~world.check_no_fly(selected)
        if no_fly_violation.any():
            if strict:
                raise ValueError(f"Waypoint no-fly violation: {np.where(no_fly_violation)[0].tolist()}")
            if reject_no_fly:
                selected[no_fly_violation] = self._fallback_waypoints(world)[no_fly_violation]
                rejected |= no_fly_violation

        delta = selected - current
        distance = np.linalg.norm(delta, axis=-1)
        distance_violation = distance > max_command_distance + 1e-8
        if distance_violation.any():
            direction = np.divide(delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 1e-8)
            selected[distance_violation] = current[distance_violation] + direction[distance_violation] * max_command_distance
            clipped_mask |= distance_violation

        connectivity_violation = np.zeros(len(selected), dtype=bool)
        if require_connectivity:
            connected_mask = self._connected_to_base_mask_for_positions(world, selected)
            if not connected_mask.all():
                connectivity_violation = ~connected_mask
                selected[connectivity_violation] = self._fallback_waypoints(world)[connectivity_violation]
                rejected |= connectivity_violation

        min_distance_violation = np.zeros(len(selected), dtype=bool)
        if len(selected) > 1 and not self.check_min_distance(world, selected):
            pairwise = world.compute_pairwise_distances(selected)
            conflict = np.argwhere((pairwise < world.min_uav_distance) & (pairwise > 0.0))
            for i, j in conflict:
                min_distance_violation[int(i)] = True
                min_distance_violation[int(j)] = True
            selected[min_distance_violation] = self._fallback_waypoints(world)[min_distance_violation]
            rejected |= min_distance_violation

        selected = world.project_to_geofence(selected)
        final_no_fly = ~world.check_no_fly(selected)
        if final_no_fly.any():
            selected[final_no_fly] = world.base_position[None, :]
            rejected |= final_no_fly

        valid = ~rejected
        return SafetyResult(
            safe_waypoints=selected.astype(np.float32),
            valid_mask=valid.astype(bool),
            clipped_mask=clipped_mask.astype(bool),
            rejected_mask=rejected.astype(bool),
            geofence_violation_mask=geofence_violation.astype(bool),
            no_fly_violation_mask=no_fly_violation.astype(bool),
            distance_violation_mask=distance_violation.astype(bool),
            connectivity_violation_mask=connectivity_violation.astype(bool),
            info={
                "safety_finite": bool(np.isfinite(selected).all()),
                "max_command_distance": float(max_command_distance),
                "min_distance_violation_count": int(min_distance_violation.sum()),
            },
        )

    def filter_waypoints(
        self,
        world: MissionWorld,
        selected_waypoints: np.ndarray,
        require_connectivity: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | np.ndarray]]:
        result = self.validate_waypoints(world, selected_waypoints, require_connectivity=require_connectivity)
        return result.safe_waypoints, result.valid_mask, result.to_info()

    def filter_candidates(
        self,
        world: MissionWorld,
        candidate_waypoints: np.ndarray,
        raw_mask: np.ndarray,
        require_connectivity: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray(candidate_waypoints, dtype=np.float32).copy()
        mask = np.asarray(raw_mask, dtype=bool).copy()
        n, k, _ = candidates.shape
        candidates = world.project_to_geofence(candidates)

        geofence_ok = world.check_geofence(candidates)
        no_fly_ok = world.check_no_fly(candidates)
        current = world.get_uav_positions()[:, None, :]
        step_distance_ok = np.linalg.norm(candidates - current, axis=-1) <= float(world.max_waypoint_distance) + 1e-6
        mask &= geofence_ok & no_fly_ok & step_distance_ok

        if require_connectivity:
            mask &= self.check_connectivity_if_selected(world, candidates)

        return self.ensure_fallback_candidate(world, candidates, mask)

    def check_geofence(self, world: MissionWorld, candidates: np.ndarray) -> np.ndarray:
        return world.check_geofence(candidates)

    def check_no_fly(self, world: MissionWorld, candidates: np.ndarray) -> np.ndarray:
        return world.check_no_fly(candidates)

    def check_min_distance(self, world: MissionWorld, selected: np.ndarray) -> bool:
        dist = world.compute_pairwise_distances(selected)
        if len(selected) <= 1:
            return True
        upper = dist[np.triu_indices(len(selected), k=1)]
        return bool(np.all(upper >= world.min_uav_distance))

    def check_connectivity_if_selected(self, world: MissionWorld, candidates: np.ndarray) -> np.ndarray:
        n, k, _ = candidates.shape
        result = np.zeros((n, k), dtype=bool)
        original_positions = world.get_uav_positions()
        for agent_idx in range(n):
            for cand_idx in range(k):
                test_positions = original_positions.copy()
                test_positions[agent_idx] = candidates[agent_idx, cand_idx]
                result[agent_idx, cand_idx] = self._all_connected(world, test_positions)
        return result

    def _connectivity_for_selected_waypoints(self, world: MissionWorld, selected: np.ndarray) -> np.ndarray:
        original_positions = world.get_uav_positions()
        result = np.zeros(len(selected), dtype=bool)
        for agent_idx in range(len(selected)):
            test_positions = original_positions.copy()
            test_positions[agent_idx] = selected[agent_idx]
            result[agent_idx] = self._all_connected(world, test_positions)
        return result

    def _connected_to_base_mask_for_positions(self, world: MissionWorld, positions: np.ndarray) -> np.ndarray:
        n = len(positions)
        graph = np.zeros((n + 1, n + 1), dtype=bool)
        graph[0, 0] = True
        for idx, pos in enumerate(positions, start=1):
            if np.linalg.norm(pos - world.base_position) <= world.base_comm_radius:
                graph[0, idx] = graph[idx, 0] = True
        for i in range(n):
            for j in range(i + 1, n):
                radius = min(world.uavs[i].comm_radius, world.uavs[j].comm_radius)
                if np.linalg.norm(positions[i] - positions[j]) <= radius:
                    graph[i + 1, j + 1] = graph[j + 1, i + 1] = True
        visited = np.zeros(n + 1, dtype=bool)
        visited[0] = True
        stack = [0]
        while stack:
            node = stack.pop()
            for nxt in np.where(graph[node])[0]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(int(nxt))
        return visited[1:].astype(bool)

    def _fallback_waypoint(self, world: MissionWorld, agent_idx: int) -> np.ndarray:
        current = world.get_uav_positions()[agent_idx].copy()
        if world.check_geofence(current[None, :])[0] and world.check_no_fly(current[None, :])[0]:
            return current.astype(np.float32)
        fallback = world.base_position.copy()
        if not (world.check_geofence(fallback[None, :])[0] and world.check_no_fly(fallback[None, :])[0]):
            fallback = world.project_to_geofence(fallback[None, :])[0]
        return fallback.astype(np.float32)

    def _fallback_waypoints(self, world: MissionWorld) -> np.ndarray:
        current = world.get_uav_positions().copy()
        valid_current = world.check_geofence(current) & world.check_no_fly(current)
        fallback = current.copy()
        fallback[~valid_current] = world.base_position[None, :]
        return world.project_to_geofence(fallback).astype(np.float32)

    def _all_connected(self, world: MissionWorld, positions: np.ndarray) -> bool:
        n = len(positions)
        graph = np.zeros((n + 1, n + 1), dtype=bool)
        graph[0, 0] = True
        for idx, pos in enumerate(positions, start=1):
            if np.linalg.norm(pos - world.base_position) <= world.base_comm_radius:
                graph[0, idx] = graph[idx, 0] = True
        for i in range(n):
            for j in range(i + 1, n):
                radius = min(world.uavs[i].comm_radius, world.uavs[j].comm_radius)
                if np.linalg.norm(positions[i] - positions[j]) <= radius:
                    graph[i + 1, j + 1] = graph[j + 1, i + 1] = True
        visited = np.zeros(n + 1, dtype=bool)
        stack = [0]
        visited[0] = True
        while stack:
            node = stack.pop()
            for nxt in np.where(graph[node])[0]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(int(nxt))
        return bool(np.all(visited[1:]))

    def ensure_fallback_candidate(
        self,
        world: MissionWorld,
        candidates: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        current = world.get_uav_positions()
        for agent_idx in range(candidates.shape[0]):
            if mask[agent_idx].any():
                continue
            fallback = current[agent_idx].copy()
            if not (world.check_geofence(fallback[None, :])[0] and world.check_no_fly(fallback[None, :])[0]):
                fallback = world.base_position.copy()
            candidates[agent_idx, 0] = world.project_to_geofence(fallback[None, :])[0]
            mask[agent_idx, 0] = True
        return candidates.astype(np.float32), mask.astype(bool)

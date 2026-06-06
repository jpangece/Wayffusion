from __future__ import annotations

import numpy as np

from envs.waypoint.world import MissionWorld


class SafetyLayer:
    """Masks illegal waypoint candidates and guarantees a safe fallback."""

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

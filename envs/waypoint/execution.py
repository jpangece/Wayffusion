from __future__ import annotations

import numpy as np

from envs.waypoint.world import MissionWorld


class WaypointExecutionModel:
    """Simulates waypoint execution without instantaneous teleports."""

    def execute(self, world: MissionWorld, selected_waypoints: np.ndarray) -> dict[str, np.ndarray | float | int]:
        selected = np.asarray(selected_waypoints, dtype=np.float32)
        old_positions = world.get_uav_positions()
        selected = world.project_to_geofence(selected)
        n = len(world.uavs)
        if selected.shape != old_positions.shape:
            raise ValueError(f"selected_waypoints must have shape {old_positions.shape}, got {selected.shape}")

        rejected = np.zeros(n, dtype=bool)
        no_fly_rejected = ~world.check_no_fly(selected)
        rejected |= no_fly_rejected
        selected[no_fly_rejected] = old_positions[no_fly_rejected]

        max_step = float(world.max_speed * world.dt_decision)
        delta = selected - old_positions
        distance = np.linalg.norm(delta, axis=-1)
        clipped_distance = np.minimum(distance, max_step)
        direction = np.divide(delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 1e-8)
        proposed = old_positions + direction * clipped_distance[:, None]
        proposed = world.project_to_geofence(proposed)

        geofence_rejected = ~world.check_geofence(proposed)
        no_fly_step_rejected = ~world.check_no_fly(proposed)
        rejected |= geofence_rejected | no_fly_step_rejected
        proposed[geofence_rejected | no_fly_step_rejected] = old_positions[geofence_rejected | no_fly_step_rejected]

        pairwise = world.compute_pairwise_distances(proposed)
        min_dist = float(world.map_size)
        if n > 1:
            upper = pairwise[np.triu_indices(n, k=1)]
            min_dist = float(upper.min()) if upper.size else float(world.map_size)
            conflict_pairs = np.argwhere((pairwise < world.min_uav_distance) & (pairwise > 0.0))
            for i, j in conflict_pairs:
                rejected[int(i)] = True
                rejected[int(j)] = True
            proposed[rejected] = old_positions[rejected]
            pairwise = world.compute_pairwise_distances(proposed)
            upper = pairwise[np.triu_indices(n, k=1)]
            if upper.size:
                min_dist = float(upper.min())

        path_delta = np.linalg.norm(proposed - old_positions, axis=-1).astype(np.float32)
        arrived = np.linalg.norm(proposed - selected, axis=-1) <= max(1e-4, max_step * 0.05)
        for idx, uav in enumerate(world.uavs):
            uav.velocity = ((proposed[idx] - old_positions[idx]) / max(world.dt_decision, 1e-6)).astype(np.float32)
            uav.position = proposed[idx].astype(np.float32)
            uav.current_waypoint = selected[idx].astype(np.float32)
            uav.path_length += float(path_delta[idx])
            uav.last_action_valid = not bool(rejected[idx])
            uav.battery = max(0.0, float(uav.battery) - float(path_delta[idx]) * 0.01)
            uav.trajectory.append(uav.position.copy())

        world.step_count += 1
        world.update_communication_graph()
        footprint_info = world.accumulate_sensor_footprints()
        return {
            "path_length_delta": path_delta,
            "arrived_mask": arrived.astype(bool),
            "rejected_mask": rejected.astype(bool),
            "safety_violation_count": int(rejected.sum()),
            "min_pairwise_distance": float(min_dist),
            "no_fly_violation_count": int((no_fly_rejected | no_fly_step_rejected).sum()),
            "geofence_violation_count": int(geofence_rejected.sum()),
            **footprint_info,
        }

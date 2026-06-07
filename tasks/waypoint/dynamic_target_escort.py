from __future__ import annotations

import numpy as np

from envs.waypoint.world import MissionWorld
from planners.waypoint_candidates import target_prediction_candidates
from tasks.waypoint.base import WaypointScenario, path_length_delta, safety_penalty_count


class DynamicTargetEscortScenario(WaypointScenario):
    def __init__(self, config: dict):
        super().__init__("dynamic_target_escort", 4, config)

    def reset(self, rng: np.random.Generator, world: MissionWorld) -> dict:
        target_position = world.sample_safe_points(1, lower_fraction=0.20, upper_fraction=0.80)[0]
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        speed = float(self.cfg("target_speed", 0.035)) * world.map_size
        velocity = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32) * speed
        return {"target_position": target_position, "target_velocity": velocity, "observed_steps": 0, "lost_target_steps": 0}

    def step_update(self, world: MissionWorld, task_state: dict) -> None:
        pos = np.asarray(task_state["target_position"], dtype=np.float32)
        vel = np.asarray(task_state["target_velocity"], dtype=np.float32)
        new_pos = world.project_to_geofence((pos + vel * world.dt_decision)[None, :])[0]
        bounced = np.any(np.isclose(new_pos, 0.0)) or np.any(np.isclose(new_pos, world.map_size))
        if bounced:
            vel = -vel
        task_state["target_position"] = new_pos
        task_state["target_velocity"] = vel

    def generate_candidate_waypoints(self, world: MissionWorld, task_state: dict) -> tuple[np.ndarray, np.ndarray]:
        return target_prediction_candidates(
            world,
            np.asarray(task_state["target_position"], dtype=np.float32),
            np.asarray(task_state["target_velocity"], dtype=np.float32),
            int(self.cfg("candidate_count", 12)),
            float(self.cfg("escort_radius", 0.18)) * world.map_size,
        )

    def compute_rewards(self, prev_world: MissionWorld, world: MissionWorld, task_state: dict, transition_info: dict) -> dict:
        del prev_world
        target = np.asarray(task_state["target_position"], dtype=np.float32)
        positions = world.get_uav_positions()
        distances = np.linalg.norm(positions - target[None, :], axis=-1)
        observed = distances <= float(self.cfg("observation_radius", 0.2)) * world.map_size
        if observed.any():
            task_state["observed_steps"] += 1
        else:
            task_state["lost_target_steps"] += 1
        desired = float(self.cfg("escort_radius", 0.18)) * world.map_size
        distance_error = float(np.mean(np.abs(distances - desired)) / max(world.map_size, 1e-6))
        angles = np.arctan2(positions[:, 1] - target[1], positions[:, 0] - target[0])
        diversity = float(np.std(np.unwrap(np.sort(angles))) / np.pi) if len(angles) > 1 else 0.0
        too_close = float(np.mean(distances < world.min_uav_distance * 2.0))
        safety = safety_penalty_count(transition_info)
        path = path_length_delta(transition_info)
        reward = (
            float(self.cfg("w_obs", 2.0)) * float(observed.any())
            + float(self.cfg("w_view", 0.5)) * diversity
            - float(self.cfg("w_too_close", 2.0)) * too_close
            - float(self.cfg("w_too_far", 1.0)) * distance_error
            - float(self.cfg("w_distance", 0.1)) * path
            - float(self.cfg("w_safety", 2.0)) * safety
        )
        per_agent = observed.astype(np.float32) * float(self.cfg("w_obs", 2.0))
        metrics = self.get_metrics(world, task_state)
        metrics.update({"mean_observation_distance": float(np.mean(distances)), "multi_view_diversity": diversity, "lost_target_steps": float(task_state["lost_target_steps"])})
        return {
            "team_reward": float(reward),
            "per_agent_rewards": per_agent,
            "components": {"observed": float(observed.any()), "view": diversity, "too_close": -too_close, "distance_error": -distance_error, "path": -path, "safety": -safety},
            "metrics": metrics,
            "success": bool(metrics["success"]),
        }

    def get_metrics(self, world: MissionWorld, task_state: dict) -> dict:
        ratio = float(task_state.get("observed_steps", 0) / max(world.step_count, 1))
        return {"target_coverage_ratio": ratio, "lost_target_steps": float(task_state.get("lost_target_steps", 0)), "success": ratio >= float(self.cfg("success_target_coverage", 0.7))}

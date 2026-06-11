from __future__ import annotations

import numpy as np

from envs.waypoint.world import MissionWorld
from planners.waypoint_candidates import merge_and_pad_candidates, radial_candidates_around_agent
from tasks.waypoint.base import WaypointScenario, path_length_delta, safety_penalty_count


class DynamicTargetEscortScenario(WaypointScenario):
    def __init__(self, config: dict):
        super().__init__("dynamic_target_escort", 4, config)

    def reset(self, rng: np.random.Generator, world: MissionWorld) -> dict:
        count = max(int(self.cfg("num_targets", 1)), 1)
        positions = world.sample_safe_points(
            count,
            lower_fraction=0.20,
            upper_fraction=0.80,
            min_separation=self.distance_cfg("target_min_separation", 0.12, world),
        )
        angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
        speed = self.distance_cfg("target_speed", 0.035, world)
        velocities = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32) * speed
        state = {
            "target_positions": positions.astype(np.float32),
            "target_velocities": velocities.astype(np.float32),
            "observed_steps": np.zeros(count, dtype=np.int64),
            "lost_target_steps": np.zeros(count, dtype=np.int64),
        }
        self._sync_legacy_aliases(state)
        return state

    def step_update(self, world: MissionWorld, task_state: dict) -> None:
        positions = np.asarray(task_state["target_positions"], dtype=np.float32)
        velocities = np.asarray(task_state["target_velocities"], dtype=np.float32)
        x_min, x_max, y_min, y_max = world.geofence
        proposed = positions + velocities * world.dt_decision
        hit_x = (proposed[:, 0] < x_min) | (proposed[:, 0] > x_max)
        hit_y = (proposed[:, 1] < y_min) | (proposed[:, 1] > y_max)
        velocities[hit_x, 0] *= -1.0
        velocities[hit_y, 1] *= -1.0
        proposed = positions + velocities * world.dt_decision
        task_state["target_positions"] = world.project_to_geofence(proposed).astype(np.float32)
        task_state["target_velocities"] = velocities.astype(np.float32)
        self._sync_legacy_aliases(task_state)

    def generate_candidate_waypoints(self, world: MissionWorld, task_state: dict) -> tuple[np.ndarray, np.ndarray]:
        count = int(self.cfg("candidate_count", 12))
        radius = self.distance_cfg("escort_radius", 0.18, world)
        points = []
        for position, velocity in zip(task_state["target_positions"], task_state["target_velocities"]):
            predicted = np.asarray(position, dtype=np.float32) + np.asarray(velocity, dtype=np.float32) * world.dt_decision * 2.0
            points.append(predicted)
            for angle in np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False):
                points.append(predicted + radius * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32))
        target_points = np.repeat(np.asarray(points, dtype=np.float32)[None, :, :], len(world.uavs), axis=0)
        return merge_and_pad_candidates(
            world,
            [radial_candidates_around_agent(world, 3)[0], target_points],
            count,
        )

    def compute_rewards(
        self,
        prev_world: MissionWorld,
        world: MissionWorld,
        task_state: dict,
        transition_info: dict,
    ) -> dict:
        del prev_world
        targets = np.asarray(task_state["target_positions"], dtype=np.float32)
        positions = world.get_uav_positions()
        distances = np.linalg.norm(positions[:, None, :] - targets[None, :, :], axis=-1)
        observation_radius = self.distance_cfg("observation_radius", 0.20, world)
        observed_by_agent = distances <= observation_radius
        observed_targets = observed_by_agent.any(axis=0)
        task_state["observed_steps"] += observed_targets.astype(np.int64)
        task_state["lost_target_steps"] += (~observed_targets).astype(np.int64)

        assigned_target = np.argmin(distances, axis=1)
        assigned_distances = distances[np.arange(len(positions)), assigned_target]
        desired = self.distance_cfg("escort_radius", 0.18, world)
        distance_error = float(np.mean(np.abs(assigned_distances - desired)) / max(desired, 1e-6))
        diversities = [self._target_diversity(positions, target, distances[:, idx], observation_radius) for idx, target in enumerate(targets)]
        diversity = float(np.mean(diversities)) if diversities else 0.0
        too_close = float(np.mean(assigned_distances < world.min_uav_distance * 2.0))
        observed_fraction = float(np.mean(observed_targets))
        safety = safety_penalty_count(transition_info)
        path = path_length_delta(transition_info)
        reward = (
            float(self.cfg("w_obs", 2.0)) * observed_fraction
            + float(self.cfg("w_view", 0.5)) * diversity
            - float(self.cfg("w_too_close", 2.0)) * too_close
            - float(self.cfg("w_too_far", 1.0)) * distance_error
            - float(self.cfg("w_distance", 0.1)) * path
            - float(self.cfg("w_safety", 2.0)) * safety
        )
        per_agent = observed_by_agent.any(axis=1).astype(np.float32) * float(self.cfg("w_obs", 2.0))
        metrics = self.get_metrics(world, task_state)
        metrics.update(
            {
                "target_observed_fraction": observed_fraction,
                "mean_observation_distance": float(np.mean(assigned_distances)),
                "multi_view_diversity": diversity,
                "lost_target_steps": float(np.asarray(task_state["lost_target_steps"]).sum()),
            }
        )
        return {
            "team_reward": float(reward),
            "per_agent_rewards": per_agent,
            "components": {
                "observed": observed_fraction,
                "view": diversity,
                "too_close": -too_close,
                "distance_error": -distance_error,
                "path": -path,
                "safety": -safety,
            },
            "metrics": metrics,
            "success": bool(metrics["success"]),
        }

    def get_metrics(self, world: MissionWorld, task_state: dict) -> dict:
        observed_steps = np.asarray(task_state.get("observed_steps", []), dtype=np.float32)
        ratios = observed_steps / max(world.step_count, 1)
        mean_ratio = float(ratios.mean()) if len(ratios) else 0.0
        worst_ratio = float(ratios.min()) if len(ratios) else 0.0
        mean_threshold = float(self.cfg("success_target_coverage", 0.70))
        worst_threshold = float(self.cfg("success_worst_target_coverage", mean_threshold))
        return {
            "target_coverage_ratio": mean_ratio,
            "worst_target_coverage_ratio": worst_ratio,
            "lost_target_steps": float(np.asarray(task_state.get("lost_target_steps", 0)).sum()),
            "num_targets": float(len(observed_steps)),
            "success": mean_ratio >= mean_threshold and worst_ratio >= worst_threshold,
        }

    @staticmethod
    def _target_diversity(
        positions: np.ndarray,
        target: np.ndarray,
        distances: np.ndarray,
        observation_radius: float,
    ) -> float:
        relevant = positions[distances <= observation_radius]
        if len(relevant) < 2:
            return 0.0
        angles = np.arctan2(relevant[:, 1] - target[1], relevant[:, 0] - target[0])
        unit = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        return float(np.clip(1.0 - np.linalg.norm(unit.mean(axis=0)), 0.0, 1.0))

    @staticmethod
    def _sync_legacy_aliases(task_state: dict) -> None:
        positions = np.asarray(task_state["target_positions"], dtype=np.float32)
        velocities = np.asarray(task_state["target_velocities"], dtype=np.float32)
        task_state["target_position"] = positions[0].copy()
        task_state["target_velocity"] = velocities[0].copy()

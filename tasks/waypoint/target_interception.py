from __future__ import annotations

import numpy as np

from envs.waypoint.world import MissionWorld
from planners.waypoint_candidates import merge_and_pad_candidates, radial_candidates_around_agent
from tasks.waypoint.base import WaypointScenario, path_length_delta, safety_penalty_count


class TargetInterceptionScenario(WaypointScenario):
    def __init__(self, config: dict):
        super().__init__("target_interception", 5, config)

    def reset(self, rng: np.random.Generator, world: MissionWorld) -> dict:
        count = max(int(self.cfg("num_targets", 1)), 1)
        all_branches = []
        all_probs = []
        chosen = []
        routes = []
        positions = []
        for target_idx in range(count):
            y_fraction = 0.12 + 0.76 * (target_idx + 0.5) / count
            start = np.asarray([0.12, y_fraction], dtype=np.float32) * world.map_size
            branches = self._make_branches(start, y_fraction, world)
            probs = rng.dirichlet(np.asarray([5.0, 3.0, 2.0], dtype=np.float32)).astype(np.float32)
            branch_idx = int(rng.choice(np.arange(len(branches)), p=probs))
            all_branches.append(branches)
            all_probs.append(probs)
            chosen.append(branch_idx)
            routes.append(branches[branch_idx])
            positions.append(branches[branch_idx][0])
        state = {
            "target_branches": all_branches,
            "target_branch_probs": all_probs,
            "chosen_branches": np.asarray(chosen, dtype=np.int64),
            "target_indices": np.zeros(count, dtype=np.int64),
            "intercepted": np.zeros(count, dtype=bool),
            "escaped": np.zeros(count, dtype=bool),
            "reached_route_end": np.zeros(count, dtype=bool),
            "interception_steps": np.zeros(count, dtype=np.int64),
            "target_positions": np.asarray(positions, dtype=np.float32),
            "target_routes": routes,
        }
        self._sync_legacy_aliases(state)
        return state

    def step_update(self, world: MissionWorld, task_state: dict) -> None:
        del world
        indices = np.asarray(task_state["target_indices"], dtype=np.int64)
        positions = []
        routes = []
        for idx, (branches, branch_idx) in enumerate(
            zip(task_state["target_branches"], task_state["chosen_branches"])
        ):
            branch = branches[int(branch_idx)]
            indices[idx] = min(int(indices[idx]) + 1, len(branch) - 1)
            task_state["reached_route_end"][idx] = bool(indices[idx] >= len(branch) - 1)
            positions.append(branch[int(indices[idx])])
            routes.append(branch)
        task_state["target_indices"] = indices
        task_state["target_positions"] = np.asarray(positions, dtype=np.float32)
        task_state["target_routes"] = routes
        self._sync_legacy_aliases(task_state)

    def build_task_field(self, world: MissionWorld, task_state: dict) -> np.ndarray:
        field = super().build_task_field(world, task_state)
        future_belief = np.zeros((world.grid_size, world.grid_size), dtype=np.float32)
        for branches, probs, current_idx in zip(
            task_state["target_branches"],
            task_state["target_branch_probs"],
            task_state["target_indices"],
        ):
            points, weights = self._future_route_points_and_weights(
                branches,
                probs,
                int(current_idx),
            )
            if len(points):
                indices = world.world_to_grid(points)
                np.add.at(future_belief, (indices[:, 1], indices[:, 0]), weights)
        future_belief[~world.navigable_grid_mask()] = 0.0
        if future_belief.max() > 1e-8:
            future_belief /= float(future_belief.max())
        field[2] = future_belief
        return field

    def generate_candidate_waypoints(self, world: MissionWorld, task_state: dict) -> tuple[np.ndarray, np.ndarray]:
        per_target_points = []
        for branches, probs, current_idx in zip(
            task_state["target_branches"],
            task_state["target_branch_probs"],
            task_state["target_indices"],
        ):
            del probs
            target_points = []
            for branch in branches:
                start = min(int(current_idx) + 2, len(branch) - 1)
                target_points.extend(branch[start::2])
            per_target_points.append(target_points)
        future_points = []
        max_points = max((len(points) for points in per_target_points), default=0)
        for point_idx in range(max_points):
            for points in per_target_points:
                if point_idx < len(points):
                    future_points.append(points[point_idx])
        count = int(self.cfg("candidate_count", 12))
        return merge_and_pad_candidates(
            world,
            [
                radial_candidates_around_agent(world, 4)[0],
                np.repeat(np.asarray(future_points, dtype=np.float32)[None, :, :], len(world.uavs), axis=0),
            ],
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
        interception_radius = self.distance_cfg("interception_radius", 0.12, world)
        intercept_now = (distances <= interception_radius).any(axis=0)
        intercepted = np.asarray(task_state["intercepted"], dtype=bool)
        newly_intercepted = intercept_now & ~intercepted
        task_state["intercepted"] = intercepted | intercept_now
        task_state["interception_steps"][newly_intercepted] = world.step_count
        reached_end = np.asarray(task_state.get("reached_route_end", []), dtype=bool)
        task_state["escaped"] |= reached_end & ~np.asarray(task_state["intercepted"], dtype=bool)
        bonus = float(self.cfg("interception_bonus", 8.0)) * float(newly_intercepted.mean())
        escape_penalty = float(self.cfg("w_escape", 5.0)) * float(
            np.mean(reached_end & ~np.asarray(task_state["intercepted"], dtype=bool))
        )
        belief_mass = self._covered_future_belief(world, task_state)
        containment = float(np.mean([self._containment_score(world, target) for target in targets]))
        path = path_length_delta(transition_info)
        safety = safety_penalty_count(transition_info)
        reward = (
            float(self.cfg("w_belief", 2.0)) * belief_mass
            + bonus
            + float(self.cfg("w_contain", 0.5)) * containment
            - escape_penalty
            - float(self.cfg("w_distance", 0.1)) * path
            - float(self.cfg("w_safety", 2.0)) * safety
        )
        new_target_mask = newly_intercepted[None, :] & (distances <= interception_radius)
        per_agent = new_target_mask.any(axis=1).astype(np.float32) * float(self.cfg("interception_bonus", 8.0))
        metrics = self.get_metrics(world, task_state)
        metrics.update(
            {
                "covered_future_belief_mass": belief_mass,
                "containment_score": containment,
                "time_to_interception": float(self._mean_interception_time(task_state)),
            }
        )
        finished = np.asarray(task_state["intercepted"], dtype=bool) | np.asarray(task_state["escaped"], dtype=bool)
        failure = bool(finished.all() and not metrics["success"])
        return {
            "team_reward": float(reward),
            "per_agent_rewards": per_agent,
            "components": {
                "belief": belief_mass,
                "intercept": bonus,
                "contain": containment,
                "escape": -escape_penalty,
                "path": -path,
                "safety": -safety,
            },
            "metrics": metrics,
            "success": bool(metrics["success"]),
            "failure": failure,
        }

    def _covered_future_belief(self, world: MissionWorld, task_state: dict) -> float:
        masses = []
        positions = world.get_uav_positions()
        radii = np.asarray([uav.sensor_radius for uav in world.uavs], dtype=np.float32)
        for branches, probs, current_idx in zip(
            task_state["target_branches"],
            task_state["target_branch_probs"],
            task_state["target_indices"],
        ):
            points, weights = self._future_route_points_and_weights(
                branches,
                probs,
                int(current_idx),
            )
            total = float(sum(float(weight) for weight in weights))
            if len(points) and len(positions):
                distances = np.linalg.norm(
                    positions[:, None, :] - points[None, :, :],
                    axis=-1,
                )
                covered_mask = (distances <= radii[:, None]).any(axis=0)
                covered = float(
                    sum(
                        float(weight)
                        for weight, is_covered in zip(weights, covered_mask)
                        if bool(is_covered)
                    )
                )
            else:
                covered = 0.0
            masses.append(covered / max(total, 1e-8))
        return float(np.mean(masses)) if masses else 0.0

    @staticmethod
    def _future_route_points_and_weights(
        branches: list[np.ndarray],
        probs: np.ndarray,
        current_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = []
        weights = []
        for branch, probability in zip(branches, probs):
            future = np.asarray(branch[int(current_idx) :], dtype=np.float32)
            if not len(future):
                continue
            points.append(future)
            weights.append(
                np.full(
                    len(future),
                    float(probability) / len(future),
                    dtype=np.float64,
                )
            )
        if not points:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float64)
        return np.concatenate(points, axis=0), np.concatenate(weights, axis=0)

    def _containment_score(self, world: MissionWorld, target: np.ndarray) -> float:
        positions = world.get_uav_positions()
        if len(positions) < 2:
            return 0.0
        angles = np.arctan2(positions[:, 1] - target[1], positions[:, 0] - target[0])
        unit = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        spread = float(np.clip(1.0 - np.linalg.norm(unit.mean(axis=0)), 0.0, 1.0))
        near_radius = self.distance_cfg("containment_radius", 0.30, world)
        near = float(np.mean(np.linalg.norm(positions - target[None, :], axis=-1) <= near_radius))
        return spread * near

    def get_metrics(self, world: MissionWorld, task_state: dict) -> dict:
        del world
        intercepted = np.asarray(task_state.get("intercepted", []), dtype=bool)
        escaped = np.asarray(task_state.get("escaped", np.zeros(len(intercepted), dtype=bool)), dtype=bool)
        ratio = float(intercepted.mean()) if len(intercepted) else 0.0
        threshold = float(self.cfg("success_interception_ratio", 1.0))
        return {
            "interception_success": ratio,
            "intercepted_target_ratio": ratio,
            "escaped_target_ratio": float(escaped.mean()) if len(escaped) else 0.0,
            "num_targets": float(len(intercepted)),
            "mean_time_to_interception": float(self._mean_interception_time(task_state)),
            "success": ratio >= threshold,
        }

    def _make_branches(self, start: np.ndarray, y_fraction: float, world: MissionWorld) -> list[np.ndarray]:
        branches = []
        step_distance = max(
            self.distance_cfg("target_speed", 0.035, world) * world.dt_decision,
            1e-6,
        )
        for offset in (-0.18, 0.0, 0.18):
            end_y = float(np.clip(y_fraction + offset, 0.08, 0.92))
            end = np.asarray([0.88, end_y], dtype=np.float32) * world.map_size
            point_count = max(int(np.ceil(np.linalg.norm(end - start) / step_distance)) + 1, 2)
            branches.append(np.linspace(start, end, point_count, dtype=np.float32))
        return branches

    @staticmethod
    def _mean_interception_time(task_state: dict) -> float:
        intercepted = np.asarray(task_state.get("intercepted", []), dtype=bool)
        steps = np.asarray(task_state.get("interception_steps", []), dtype=np.float32)
        return float(steps[intercepted].mean()) if bool(intercepted.any()) else 0.0

    @staticmethod
    def _sync_legacy_aliases(task_state: dict) -> None:
        task_state["target_position"] = np.asarray(task_state["target_positions"], dtype=np.float32)[0].copy()
        task_state["target_route"] = np.asarray(task_state["target_routes"][0], dtype=np.float32)
        task_state["branches"] = task_state["target_branches"][0]
        task_state["branch_probs"] = task_state["target_branch_probs"][0]
        task_state["chosen_branch"] = int(task_state["chosen_branches"][0])
        task_state["target_index"] = int(task_state["target_indices"][0])

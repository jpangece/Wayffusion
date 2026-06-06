from __future__ import annotations

import numpy as np

from envs.waypoint.candidates import merge_and_pad_candidates, radial_candidates_around_agent
from envs.waypoint.world import MissionWorld
from tasks.waypoint.base import WaypointScenario, path_length_delta, safety_penalty_count


class TargetInterceptionScenario(WaypointScenario):
    def __init__(self, config: dict):
        super().__init__("target_interception", 5, config)

    def reset(self, rng: np.random.Generator, world: MissionWorld) -> dict:
        start = np.asarray([0.15, 0.15], dtype=np.float32) * world.map_size
        branches = []
        for angle in [0.0, np.pi / 6.0, np.pi / 3.0]:
            end = np.asarray([0.85, 0.2 + 0.55 * np.sin(angle)], dtype=np.float32) * world.map_size
            branch = np.linspace(start, end, 12, dtype=np.float32)
            branches.append(branch)
        probs = np.asarray([0.5, 0.3, 0.2], dtype=np.float32)
        branch_idx = int(rng.choice(np.arange(len(branches)), p=probs))
        return {"branches": branches, "branch_probs": probs, "chosen_branch": branch_idx, "target_index": 0, "intercepted": False, "target_position": branches[branch_idx][0], "target_route": branches[branch_idx]}

    def step_update(self, world: MissionWorld, task_state: dict) -> None:
        branch = task_state["branches"][int(task_state["chosen_branch"])]
        task_state["target_index"] = min(int(task_state["target_index"]) + 1, len(branch) - 1)
        task_state["target_position"] = branch[int(task_state["target_index"])]
        task_state["target_route"] = branch

    def generate_candidate_waypoints(self, world: MissionWorld, task_state: dict) -> tuple[np.ndarray, np.ndarray]:
        future_points = []
        for branch, prob in zip(task_state["branches"], task_state["branch_probs"]):
            start = min(int(task_state["target_index"]) + 2, len(branch) - 1)
            for point in branch[start::2]:
                future_points.append(point)
        return merge_and_pad_candidates(world, [radial_candidates_around_agent(world, 4)[0], np.repeat(np.asarray(future_points, dtype=np.float32)[None, :, :], len(world.uavs), axis=0)], int(self.cfg("candidate_count", 12)))

    def compute_rewards(self, prev_world: MissionWorld, world: MissionWorld, task_state: dict, transition_info: dict) -> dict:
        del prev_world
        target = np.asarray(task_state["target_position"], dtype=np.float32)
        positions = world.get_uav_positions()
        distances = np.linalg.norm(positions - target[None, :], axis=-1)
        intercept_now = bool(np.any(distances <= float(self.cfg("interception_radius", 0.12)) * world.map_size))
        bonus = 0.0
        if intercept_now and not task_state.get("intercepted", False):
            task_state["intercepted"] = True
            bonus = float(self.cfg("interception_bonus", 8.0))
        belief_mass = self._covered_future_belief(world, task_state)
        containment = self._containment_score(world, target)
        path = path_length_delta(transition_info)
        safety = safety_penalty_count(transition_info)
        reward = (
            float(self.cfg("w_belief", 2.0)) * belief_mass
            + bonus
            + float(self.cfg("w_contain", 0.5)) * containment
            - float(self.cfg("w_distance", 0.1)) * path
            - float(self.cfg("w_safety", 2.0)) * safety
        )
        per_agent = (distances <= float(self.cfg("interception_radius", 0.12)) * world.map_size).astype(np.float32) * bonus
        metrics = self.get_metrics(world, task_state)
        metrics.update({"covered_future_belief_mass": belief_mass, "containment_score": containment, "time_to_interception": float(world.step_count if task_state.get("intercepted", False) else 0)})
        return {
            "team_reward": float(reward),
            "per_agent_rewards": per_agent.astype(np.float32),
            "components": {"belief": belief_mass, "intercept": bonus, "contain": containment, "path": -path, "safety": -safety},
            "metrics": metrics,
            "success": bool(metrics["success"]),
        }

    def _covered_future_belief(self, world: MissionWorld, task_state: dict) -> float:
        total = 0.0
        for branch, prob in zip(task_state["branches"], task_state["branch_probs"]):
            for point in branch[int(task_state["target_index"]) :]:
                if any(np.linalg.norm(uav.position - point) <= uav.sensor_radius for uav in world.uavs):
                    total += float(prob) / max(len(branch), 1)
        return float(total)

    def _containment_score(self, world: MissionWorld, target: np.ndarray) -> float:
        positions = world.get_uav_positions()
        if len(positions) < 2:
            return 0.0
        angles = np.arctan2(positions[:, 1] - target[1], positions[:, 0] - target[0])
        spread = float(np.ptp(np.unwrap(np.sort(angles))) / (2.0 * np.pi))
        near = float(np.mean(np.linalg.norm(positions - target[None, :], axis=-1) <= 0.3 * world.map_size))
        return spread * near

    def get_metrics(self, world: MissionWorld, task_state: dict) -> dict:
        del world
        return {"interception_success": float(task_state.get("intercepted", False)), "success": bool(task_state.get("intercepted", False))}

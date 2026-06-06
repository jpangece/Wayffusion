from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.waypoint.execution import WaypointExecutionModel
from envs.waypoint.rendering import render_world
from envs.waypoint.safety import SafetyLayer
from envs.waypoint.world import MissionWorld
from tasks.waypoint import WAYPOINT_TASKS, build_waypoint_scenario
from tasks.waypoint.base import normalized_positions


class WaypointMultiUAVEnv:
    """PettingZoo-style Parallel API for waypoint-level multi-UAV missions.

    Each UAV is an independent agent and chooses one discrete candidate waypoint
    index per decision step. The environment then simulates bounded waypoint
    execution and exposes a centralized global state for CTDE/MAPPO critics.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict[str, Any]):
        self.config = deepcopy(config)
        self.num_agents = int(self.config.get("num_agents", 4))
        self.candidate_count = int(self.config.get("candidate_count", 12))
        self.seed_value = int(self.config.get("seed", 0))
        self.rng = np.random.default_rng(self.seed_value)
        self.task_names = list(self.config.get("task_names", ["area_coverage", "belief_search", "priority_inspection", "connectivity_expansion"]))
        if self.config.get("task_name") is not None:
            self.task_names = [str(self.config["task_name"])]
        self.task_sampling_probs = self._task_probs(self.task_names)
        self.possible_agents = [f"uav_{idx}" for idx in range(self.num_agents)]
        self.agents = self.possible_agents.copy()
        self.world = MissionWorld.from_config(self.config, rng=self.rng)
        self.world.sensor_radius = float(self.config.get("sensor_radius", 0.12))
        self.world.comm_radius = float(self.config.get("comm_radius", 0.35))
        self.safety = SafetyLayer()
        self.execution = WaypointExecutionModel()
        self.current_scenario = None
        self.task_state: dict[str, Any] = {}
        self.last_candidate_waypoints = np.zeros((self.num_agents, self.candidate_count, 2), dtype=np.float32)
        self.last_candidate_mask = np.zeros((self.num_agents, self.candidate_count), dtype=bool)
        self.last_selected_waypoints: np.ndarray | None = None
        self.last_reward_result: dict[str, Any] = {}
        self.last_transition_info: dict[str, Any] = {}
        self._build_spaces()

    def _task_probs(self, task_names: list[str]) -> np.ndarray:
        raw = self.config.get("task_sampling_probs", {})
        probs = np.asarray([float(raw.get(name, 1.0)) for name in task_names], dtype=np.float32)
        probs = probs / max(float(probs.sum()), 1e-8)
        return probs

    def _build_spaces(self) -> None:
        k = self.candidate_count
        n = self.num_agents
        g = int(self.config.get("grid_size", 32))
        map_size = float(self.config.get("map_size", 1.0))
        self._single_observation_space = spaces.Dict(
            {
                "self_state": spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32),
                "neighbor_relative_states": spaces.Box(low=-10.0, high=10.0, shape=(max(n - 1, 0), 4), dtype=np.float32),
                "task_field": spaces.Box(low=0.0, high=1.0, shape=(5, g, g), dtype=np.float32),
                "candidate_waypoints": spaces.Box(low=0.0, high=map_size, shape=(k, 2), dtype=np.float32),
                "candidate_features": spaces.Box(low=-10.0, high=10.0, shape=(k, 6), dtype=np.float32),
                "candidate_mask": spaces.MultiBinary(k),
                "task_id": spaces.Box(low=0.0, high=1.0, shape=(len(WAYPOINT_TASKS),), dtype=np.float32),
                "global_summary": spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32),
                "all_uav_states": spaces.Box(low=-10.0, high=10.0, shape=(n, 8), dtype=np.float32),
                "comm_adjacency": spaces.MultiBinary((n + 1, n + 1)),
            }
        )
        self._action_space = spaces.Discrete(k)
        self.global_observation_space = spaces.Dict(
            {
                "task_field": spaces.Box(low=0.0, high=1.0, shape=(5, g, g), dtype=np.float32),
                "global_task_field": spaces.Box(low=0.0, high=1.0, shape=(5, g, g), dtype=np.float32),
                "all_uav_states": spaces.Box(low=-10.0, high=10.0, shape=(n, 8), dtype=np.float32),
                "candidate_waypoints": spaces.Box(low=0.0, high=map_size, shape=(n, k, 2), dtype=np.float32),
                "candidate_features": spaces.Box(low=-10.0, high=10.0, shape=(n, k, 6), dtype=np.float32),
                "candidate_mask": spaces.MultiBinary((n, k)),
                "task_id": spaces.Box(low=0.0, high=1.0, shape=(len(WAYPOINT_TASKS),), dtype=np.float32),
                "global_info": spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32),
                "comm_adjacency": spaces.MultiBinary((n + 1, n + 1)),
                "agent_mask": spaces.MultiBinary(n),
            }
        )
        self.action_space_n = spaces.MultiDiscrete(np.full(n, k, dtype=np.int64))

    def observation_space(self, agent_id: str):
        self._assert_agent(agent_id)
        return self._single_observation_space

    def action_space(self, agent_id: str):
        self._assert_agent(agent_id)
        return self._action_space

    def _assert_agent(self, agent_id: str) -> None:
        if agent_id not in self.possible_agents:
            raise KeyError(f"Unknown agent_id: {agent_id}")

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)
        options = options or {}
        self.world = MissionWorld.from_config(self.config, rng=self.rng)
        self.world.sensor_radius = float(self.config.get("sensor_radius", 0.12))
        self.world.comm_radius = float(self.config.get("comm_radius", 0.35))
        self.world.reset_common(self.num_agents, rng=self.rng, initial_positions=options.get("initial_positions"))
        task_name = str(options.get("task_name") or self.rng.choice(self.task_names, p=self.task_sampling_probs))
        self.current_scenario = build_waypoint_scenario(task_name, self.config.get("tasks", self.config))
        self.task_state = self.current_scenario.reset(self.rng, self.world)
        self.agents = self.possible_agents.copy()
        self.last_selected_waypoints = None
        self.last_reward_result = {}
        self.last_transition_info = {}
        self._refresh_candidates()
        observations = self._build_agent_observations()
        infos = {agent: self._build_agent_info(agent_idx) for agent_idx, agent in enumerate(self.possible_agents)}
        return observations, infos

    def step(self, actions: dict[str, int]):
        if self.current_scenario is None:
            raise RuntimeError("reset() must be called before step().")
        action_array, validity = self._actions_to_array(actions)
        selected = self.last_candidate_waypoints[np.arange(self.num_agents), action_array].copy()
        if not np.all(validity):
            fallback = self._fallback_indices()
            invalid_indices = np.where(~validity)[0]
            selected[invalid_indices] = self.last_candidate_waypoints[invalid_indices, fallback[invalid_indices]]
        self.last_selected_waypoints = selected.copy()
        prev_world = self.world.snapshot()
        self.current_scenario.step_update(self.world, self.task_state)
        transition_info = self.execution.execute(self.world, selected)
        transition_info["invalid_action_count"] = int((~validity).sum())
        reward_result = self.current_scenario.compute_rewards(prev_world, self.world, self.task_state, transition_info)
        self.last_transition_info = transition_info
        self.last_reward_result = reward_result
        self._refresh_candidates()

        success = bool(reward_result.get("success", False))
        terminated = success
        truncated = bool(self.world.step_count >= self.world.max_steps)
        observations = self._build_agent_observations()
        rewards = {
            agent: float(reward_result["per_agent_rewards"][idx])
            for idx, agent in enumerate(self.possible_agents)
        }
        terminations = {agent: terminated for agent in self.possible_agents}
        truncations = {agent: truncated for agent in self.possible_agents}
        infos = {agent: self._build_agent_info(idx) for idx, agent in enumerate(self.possible_agents)}
        if terminated or truncated:
            terminal_observations = observations
            for idx, agent in enumerate(self.possible_agents):
                infos[agent]["terminal_observation"] = terminal_observations[agent]
                infos[agent]["terminal_info"] = deepcopy(infos[agent])
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _actions_to_array(self, actions: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
        strict = bool(self.config.get("strict_action_validation", False))
        action_array = np.zeros(self.num_agents, dtype=np.int64)
        valid = np.ones(self.num_agents, dtype=bool)
        fallback = self._fallback_indices()
        for idx, agent in enumerate(self.possible_agents):
            raw = actions.get(agent, int(fallback[idx]))
            action = int(raw)
            if action < 0 or action >= self.candidate_count or not bool(self.last_candidate_mask[idx, action]):
                if strict:
                    raise ValueError(f"Illegal action {action} for {agent}")
                action = int(fallback[idx])
                valid[idx] = False
            action_array[idx] = action
        return action_array, valid

    def _fallback_indices(self) -> np.ndarray:
        out = np.zeros(self.num_agents, dtype=np.int64)
        for idx in range(self.num_agents):
            valid = np.where(self.last_candidate_mask[idx])[0]
            out[idx] = int(valid[0]) if len(valid) else 0
        return out

    def _refresh_candidates(self) -> None:
        candidates, raw_mask = self.current_scenario.generate_candidate_waypoints(self.world, self.task_state)
        require_connectivity = self.current_scenario.name == "connectivity_expansion" and bool(self.config.get("connectivity_candidate_filter", True))
        candidates, mask = self.safety.filter_candidates(self.world, candidates, raw_mask, require_connectivity=require_connectivity)
        self.last_candidate_waypoints = candidates.astype(np.float32)
        self.last_candidate_mask = mask.astype(bool)

    def _candidate_features(self) -> np.ndarray:
        positions = self.world.get_uav_positions()[:, None, :]
        rel = (self.last_candidate_waypoints - positions) / max(self.world.map_size, 1e-6)
        dist = np.linalg.norm(rel, axis=-1, keepdims=True)
        grid = self.world.world_to_grid(self.last_candidate_waypoints.reshape(-1, 2)).reshape(self.num_agents, self.candidate_count, 2)
        x = grid[..., 0]
        y = grid[..., 1]
        unvisited = 1.0 - self.world.coverage_grid[y, x][..., None]
        belief = self.world.belief_grid[y, x][..., None]
        no_fly_ok = self.world.check_no_fly(self.last_candidate_waypoints)[..., None].astype(np.float32)
        return np.concatenate([rel, dist, unvisited, belief, no_fly_ok], axis=-1).astype(np.float32)

    def _neighbor_relative_states(self, agent_idx: int) -> np.ndarray:
        positions = self.world.get_uav_positions()
        velocities = np.asarray([uav.velocity for uav in self.world.uavs], dtype=np.float32)
        if self.num_agents <= 1:
            return np.zeros((0, 4), dtype=np.float32)
        others = [idx for idx in range(self.num_agents) if idx != agent_idx]
        rel_pos = (positions[others] - positions[agent_idx][None, :]) / max(self.world.map_size, 1e-6)
        rel_vel = (velocities[others] - velocities[agent_idx][None, :]) / max(self.world.max_speed, 1e-6)
        return np.concatenate([rel_pos, rel_vel], axis=-1).astype(np.float32)

    def _build_agent_observations(self) -> dict[str, dict[str, np.ndarray]]:
        global_state = self.get_global_state()
        features = global_state["candidate_features"]
        all_states = global_state["all_uav_states"]
        observations = {}
        for idx, agent in enumerate(self.possible_agents):
            observations[agent] = {
                "self_state": all_states[idx].astype(np.float32),
                "neighbor_relative_states": self._neighbor_relative_states(idx),
                "task_field": global_state["task_field"].astype(np.float32),
                "candidate_waypoints": self.last_candidate_waypoints[idx].astype(np.float32),
                "candidate_features": features[idx].astype(np.float32),
                "candidate_mask": self.last_candidate_mask[idx].astype(np.int8),
                "task_id": global_state["task_id"].astype(np.float32),
                "global_summary": global_state["global_info"].astype(np.float32),
                "all_uav_states": all_states.astype(np.float32),
                "comm_adjacency": self.world.communication_graph.astype(np.int8),
            }
        return observations

    def _build_agent_info(self, agent_idx: int) -> dict[str, Any]:
        reward_result = self.last_reward_result or {}
        metrics = dict(reward_result.get("metrics", self.current_scenario.get_metrics(self.world, self.task_state)))
        info = {
            "team_reward": float(reward_result.get("team_reward", 0.0)),
            "per_agent_rewards": np.asarray(reward_result.get("per_agent_rewards", np.zeros(self.num_agents, dtype=np.float32)), dtype=np.float32),
            "reward_components": dict(reward_result.get("components", {})),
            "task_metrics": metrics,
            "candidate_mask": self.last_candidate_mask[agent_idx].copy(),
            "selected_waypoints": None if self.last_selected_waypoints is None else self.last_selected_waypoints.copy(),
            "action_validity": bool(self.world.uavs[agent_idx].last_action_valid),
            "task_name": self.current_scenario.name,
            "num_agents": self.num_agents,
            "success": bool(reward_result.get("success", metrics.get("success", False))),
            "path_length": float(sum(uav.path_length for uav in self.world.uavs)),
            "collision_rate": float(self.last_transition_info.get("safety_violation_count", 0) / max(self.world.step_count * self.num_agents, 1)),
            "invalid_action_count": int(self.last_transition_info.get("invalid_action_count", 0)),
            "candidate_mask_empty_count": int(np.count_nonzero(~self.last_candidate_mask.any(axis=1))),
        }
        info.update(metrics)
        return info

    def get_global_state(self) -> dict[str, np.ndarray]:
        if self.current_scenario is None:
            task_field = np.zeros((5, self.world.grid_size, self.world.grid_size), dtype=np.float32)
            task_id = np.zeros(len(WAYPOINT_TASKS), dtype=np.float32)
        else:
            task_field = self.current_scenario.build_task_field(self.world, self.task_state)
            task_id = self.current_scenario.task_one_hot()
        return {
            "task_field": task_field.astype(np.float32),
            "global_task_field": task_field.astype(np.float32),
            "all_uav_states": normalized_positions(self.world).astype(np.float32),
            "candidate_waypoints": self.last_candidate_waypoints.astype(np.float32),
            "candidate_features": self._candidate_features().astype(np.float32),
            "candidate_mask": self.last_candidate_mask.astype(np.int8),
            "task_id": task_id.astype(np.float32),
            "global_info": self.world.get_global_summary().astype(np.float32),
            "comm_adjacency": self.world.communication_graph.astype(np.int8),
            "agent_mask": np.ones(self.num_agents, dtype=np.int8),
        }

    state = get_global_state

    def render(self, mode: str = "rgb_array"):
        image = render_world(
            self.world,
            self.current_scenario.name if self.current_scenario is not None else "uninitialized",
            self.task_state,
            candidate_waypoints=self.last_candidate_waypoints if bool(self.config.get("render_candidates", True)) else None,
            selected_waypoints=self.last_selected_waypoints,
            metrics=(self.last_reward_result or {}).get("metrics", {}),
        )
        if mode == "rgb_array":
            return image
        if mode == "human":
            try:
                import matplotlib.pyplot as plt

                plt.imshow(image)
                plt.axis("off")
                plt.pause(0.001)
                return None
            except Exception:
                return image
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self) -> None:
        return None


__all__ = ["WaypointMultiUAVEnv"]

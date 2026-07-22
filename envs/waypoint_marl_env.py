from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.waypoint.control import build_action_adapter, build_dynamics_backend
from envs.waypoint.mission_goals import GOAL_DIM, goal_progress, resolve_episode_goal
from envs.waypoint.randomization import build_episode_config
from envs.waypoint.rendering import render_world
from envs.waypoint.safety import SafetyLayer
from envs.waypoint.task_element_heatmap import generate_task_element_heatmap
from envs.waypoint.task_element_provider import resolve_task_elements
from envs.waypoint.world import MissionWorld
from tasks.waypoint import WAYPOINT_TASKS, build_waypoint_scenario
from tasks.waypoint.base import normalized_positions


class WaypointMultiUAVEnv:
    """PettingZoo-style Parallel API for waypoint-level multi-UAV missions.

    The main action API accepts final per-UAV waypoint coordinates. Candidate
    waypoint generation and candidate-index selection are policy/planner
    responsibilities, not environment responsibilities. A legacy candidate-index
    mode is kept only for diagnostics.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict[str, Any]):
        self.config = deepcopy(config)
        self.num_agents = int(self.config.get("num_agents", 4))
        self.action_interface = str(self.config.get("action_interface", self.config.get("action_mode", "waypoint")))
        self.action_mode = str(self.config.get("action_mode", self.action_interface))
        self.waypoint_dim = int(self.config.get("waypoint_dim", self.config.get("world_dim", 2)))
        self.waypoint_dim = 3 if self.waypoint_dim == 3 else 2
        self.candidate_count = int(self.config.get("candidate_count", self.config.get("tasks", {}).get("candidate_count", 12)))
        self.expose_candidates_for_debug = bool(self.config.get("expose_candidates_for_debug", False))
        heatmap_config = dict(self.config.get("heatmap_observation", {}))
        heatmap_provider_config = dict(heatmap_config.get("provider", {}))
        self.heatmap_observation_enabled = bool(heatmap_config.get("enabled", False))
        self.heatmap_provider_enabled = bool(heatmap_provider_config.get("enabled", False))
        self.heatmap_provider_refresh_on_step = bool(
            heatmap_provider_config.get("refresh_on_step", False)
        )
        self.heatmap_observation_channels = int(heatmap_config.get("channels", 1))
        self.heatmap_task_elements = deepcopy(heatmap_config.get("task_elements", []))
        if self.heatmap_observation_channels < 1:
            raise ValueError("heatmap_observation.channels must be at least 1")
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
        self.action_adapter = build_action_adapter(self._action_adapter_config())
        self.dynamics_backend = build_dynamics_backend(self._dynamics_backend_config())
        self.current_scenario = None
        self.task_state: dict[str, Any] = {}
        self.current_mission_goal = np.zeros(GOAL_DIM, dtype=np.float32)
        self.current_goal_mask = np.zeros(GOAL_DIM, dtype=np.float32)
        self.current_goal_split = "fixed"
        self.current_goal_id = "fixed_00"
        self.last_candidate_waypoints = np.zeros((self.num_agents, self.candidate_count, 2), dtype=np.float32)
        self.last_candidate_mask = np.zeros((self.num_agents, self.candidate_count), dtype=bool)
        self.last_selected_waypoints: np.ndarray | None = None
        self.last_raw_waypoints: np.ndarray | None = None
        self.last_reward_result: dict[str, Any] = {}
        self.last_transition_info: dict[str, Any] = {}
        self.episode_config = deepcopy(self.config)
        self.episode_randomization_info: dict[str, Any] = {"enabled": False}
        self.task_element_heatmap = np.zeros(
            (self.heatmap_observation_channels, self.world.grid_size, self.world.grid_size),
            dtype=np.float32,
        )
        self._global_state_cache: dict[str, np.ndarray] | None = None
        self._shared_info_cache: dict[str, Any] | None = None
        self._build_spaces()

    def _invalidate_runtime_caches(self) -> None:
        self._global_state_cache = None
        self._shared_info_cache = None

    def _refresh_task_element_heatmap(self) -> None:
        resolved_task_elements = resolve_task_elements(
            heatmap_enabled=self.heatmap_observation_enabled,
            provider_enabled=self.heatmap_provider_enabled,
            task_name=self.current_scenario.name,
            static_task_elements=self.heatmap_task_elements,
            scenario=self.current_scenario,
            world=self.world,
            task_state=self.task_state,
        )
        if self.heatmap_observation_enabled and (
            resolved_task_elements or self.heatmap_observation_channels == 1
        ):
            self.task_element_heatmap = generate_task_element_heatmap(
                grid_size=self.world.grid_size,
                task_elements=resolved_task_elements,
                channels=self.heatmap_observation_channels,
            )

    def _action_adapter_config(self) -> dict[str, Any]:
        physical_scale = dict(self.config.get("physical_scale", {}))
        cfg = {
            "name": "waypoint_velocity_tracker",
            "max_command_distance": float(self.config.get("max_waypoint_distance", 0.22)),
            "meters_per_unit": float(physical_scale.get("meters_per_unit", 100.0)),
            "max_speed": float(self.config.get("max_speed", 0.04)),
            "slowdown_radius": 0.08,
            "velocity_gain": 4.0,
            "max_accel": 0.10,
            "control_smoothing": 0.45,
            "acceptance_radius": 0.020,
            "hover_when_arrived": True,
            "command_latency_steps": 0,
            "tracking_noise_std": 0.0,
            "clip_to_geofence": True,
            "reject_no_fly_waypoints": True,
            "strict_action_validation": bool(self.config.get("strict_action_validation", False)),
        }
        cfg.update(dict(self.config.get("action_adapter", {})))
        return cfg

    def _dynamics_backend_config(self) -> dict[str, Any]:
        cfg = {
            "name": "mpe_core",
            "source": "third_party_openai_mpe",
            "dt": 0.1,
            "substeps": 20,
            "mass": 1.0,
            "damping": 0.25,
            "max_speed": float(self.config.get("max_speed", 0.04)),
            "agent_size": 0.02,
            "agent_accel": 1.0,
            "action_scale": 1.0,
            "contact_force": 100.0,
            "contact_margin": 0.01,
            "enable_boundary_projection": True,
            "enable_no_fly_projection": True,
        }
        cfg.update(dict(self.config.get("dynamics_backend", {})))
        return cfg

    def _task_probs(self, task_names: list[str]) -> np.ndarray:
        raw = self.config.get("task_sampling_probs", {})
        probs = np.asarray([float(raw.get(name, 1.0)) for name in task_names], dtype=np.float32)
        probs = probs / max(float(probs.sum()), 1e-8)
        return probs

    def _build_spaces(self) -> None:
        n = self.num_agents
        g = int(self.config.get("grid_size", 32))
        map_size = float(self.config.get("map_size", 1.0))
        obs_items: dict[str, gym.Space] = {
            "self_state": spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32),
            "neighbor_relative_states": spaces.Box(low=-10.0, high=10.0, shape=(max(n - 1, 0), 4), dtype=np.float32),
            "task_field": spaces.Box(low=0.0, high=1.0, shape=(5, g, g), dtype=np.float32),
            "task_id": spaces.Box(low=0.0, high=1.0, shape=(len(WAYPOINT_TASKS),), dtype=np.float32),
            "mission_goal": spaces.Box(low=0.0, high=10.0, shape=(GOAL_DIM,), dtype=np.float32),
            "goal_mask": spaces.Box(low=0.0, high=1.0, shape=(GOAL_DIM,), dtype=np.float32),
            "global_summary": spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32),
            "all_uav_states": spaces.Box(low=-10.0, high=10.0, shape=(n, 8), dtype=np.float32),
            "comm_adjacency": spaces.MultiBinary((n + 1, n + 1)),
        }
        global_items: dict[str, gym.Space] = {
            "task_field": spaces.Box(low=0.0, high=1.0, shape=(5, g, g), dtype=np.float32),
            "global_task_field": spaces.Box(low=0.0, high=1.0, shape=(5, g, g), dtype=np.float32),
            "all_uav_states": spaces.Box(low=-10.0, high=10.0, shape=(n, 8), dtype=np.float32),
            "task_id": spaces.Box(low=0.0, high=1.0, shape=(len(WAYPOINT_TASKS),), dtype=np.float32),
            "mission_goal": spaces.Box(low=0.0, high=10.0, shape=(GOAL_DIM,), dtype=np.float32),
            "goal_mask": spaces.Box(low=0.0, high=1.0, shape=(GOAL_DIM,), dtype=np.float32),
            "global_info": spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32),
            "comm_adjacency": spaces.MultiBinary((n + 1, n + 1)),
            "agent_mask": spaces.MultiBinary(n),
        }
        if self.heatmap_observation_enabled:
            heatmap_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.heatmap_observation_channels, g, g),
                dtype=np.float32,
            )
            obs_items["task_element_heatmap"] = heatmap_space
            global_items["task_element_heatmap"] = heatmap_space
        if self._exposes_candidates():
            k = self.candidate_count
            obs_items.update(
                {
                    "candidate_waypoints": spaces.Box(low=0.0, high=map_size, shape=(k, 2), dtype=np.float32),
                    "candidate_features": spaces.Box(low=-10.0, high=10.0, shape=(k, 6), dtype=np.float32),
                    "candidate_mask": spaces.MultiBinary(k),
                }
            )
            global_items.update(
                {
                    "candidate_waypoints": spaces.Box(low=0.0, high=map_size, shape=(n, k, 2), dtype=np.float32),
                    "candidate_features": spaces.Box(low=-10.0, high=10.0, shape=(n, k, 6), dtype=np.float32),
                    "candidate_mask": spaces.MultiBinary((n, k)),
                }
            )
        self._single_observation_space = spaces.Dict(obs_items)
        self.global_observation_space = spaces.Dict(global_items)
        if self.action_mode == "candidate_index_legacy":
            self._action_space = spaces.Discrete(self.candidate_count)
            self.action_space_n = spaces.MultiDiscrete(np.full(n, self.candidate_count, dtype=np.int64))
            return
        low, high = self._waypoint_bounds()
        self._action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space_n = spaces.Box(
            low=np.repeat(low[None, :], n, axis=0),
            high=np.repeat(high[None, :], n, axis=0),
            dtype=np.float32,
        )

    def _waypoint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        map_size = float(self.config.get("map_size", 1.0))
        if self.waypoint_dim == 3:
            min_alt = float(self.config.get("min_altitude", self.config.get("default_altitude", 30.0)))
            max_alt = float(self.config.get("max_altitude", self.config.get("default_altitude", 30.0)))
            return np.asarray([0.0, 0.0, min_alt], dtype=np.float32), np.asarray([map_size, map_size, max_alt], dtype=np.float32)
        return np.asarray([0.0, 0.0], dtype=np.float32), np.asarray([map_size, map_size], dtype=np.float32)

    def _exposes_candidates(self) -> bool:
        return self.action_mode == "candidate_index_legacy" or self.expose_candidates_for_debug

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
        self._invalidate_runtime_caches()
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)
        options = options or {}
        self.episode_config, self.episode_randomization_info = build_episode_config(self.config, self.rng)
        self.world = MissionWorld.from_config(self.episode_config, rng=self.rng)
        # Domain randomization must not alter sensing, communication, or dynamics scales.
        self.world.sensor_radius = float(self.config.get("sensor_radius", 0.12))
        self.world.comm_radius = float(self.config.get("comm_radius", 0.35))
        self.world.base_comm_radius = float(self.config.get("base_comm_radius", 0.45))
        self.world.reset_common(self.num_agents, rng=self.rng, initial_positions=options.get("initial_positions"))
        self.action_adapter.reset(self.world)
        self.dynamics_backend.reset(self.world)
        task_name = str(options.get("task_name") or self.rng.choice(self.task_names, p=self.task_sampling_probs))
        self.current_scenario = build_waypoint_scenario(task_name, self.episode_config.get("tasks", self.episode_config))
        self.task_state = self.current_scenario.reset(self.rng, self.world)
        goal_config = dict(self.episode_config.get("mission_goals", {}))
        requested_split = str(options.get("goal_split", goal_config.get("train_split", "fixed")))
        (
            self.current_mission_goal,
            self.current_goal_mask,
            self.current_goal_split,
            self.current_goal_id,
        ) = resolve_episode_goal(
            task_name,
            self.episode_config,
            self.rng,
            split=requested_split,
            forced_goal=options.get("mission_goal"),
        )
        self.task_state["_mission_goal"] = self.current_mission_goal.copy()
        self.task_state["_goal_mask"] = self.current_goal_mask.copy()
        self.task_state["_goal_split"] = self.current_goal_split
        self.task_state["_goal_id"] = self.current_goal_id
        self._record_episode_randomization(task_name)
        self.agents = self.possible_agents.copy()
        self.last_selected_waypoints = None
        self.last_raw_waypoints = None
        self.last_reward_result = {}
        self.last_transition_info = {}
        self._refresh_task_element_heatmap()
        if self._exposes_candidates():
            self._refresh_candidates()
        self._invalidate_runtime_caches()
        observations = self._build_agent_observations()
        infos = {agent: self._build_agent_info(agent_idx) for agent_idx, agent in enumerate(self.possible_agents)}
        return observations, infos

    def _record_episode_randomization(self, task_name: str) -> None:
        metadata = self.episode_randomization_info
        metadata["task_name"] = task_name
        metadata["mission_goal"] = self.current_mission_goal.astype(float).tolist()
        metadata["goal_mask"] = self.current_goal_mask.astype(float).tolist()
        metadata["goal_split"] = self.current_goal_split
        metadata["goal_id"] = self.current_goal_id
        metadata["base_position"] = self.world.base_position.astype(float).tolist()
        metadata["spawn_positions"] = self.world.get_uav_positions().astype(float).tolist()
        metadata["fixed_parameters"] = {
            "sensor_radius": float(self.world.sensor_radius),
            "comm_radius": float(self.world.comm_radius),
            "base_comm_radius": float(self.world.base_comm_radius),
        }
        if task_name == "belief_search":
            metadata["belief_peak_centers"] = np.asarray(
                self.task_state.get("peak_centers", []),
                dtype=np.float32,
            ).astype(float).tolist()
            metadata["belief_target_position"] = np.asarray(
                self.task_state.get("target_position", []),
                dtype=np.float32,
            ).astype(float).tolist()
        elif task_name == "priority_inspection":
            metadata["poi_positions"] = np.asarray(
                self.task_state.get("pois", []),
                dtype=np.float32,
            ).astype(float).tolist()
            metadata["poi_weights"] = np.asarray(
                self.task_state.get("weights", []),
                dtype=np.float32,
            ).astype(float).tolist()
            metadata["poi_deadlines"] = np.asarray(
                self.task_state.get("deadlines", []),
                dtype=np.int64,
            ).astype(int).tolist()
        elif task_name in {"dynamic_target_escort", "target_interception"}:
            metadata["target_positions"] = np.asarray(
                self.task_state.get("target_positions", []),
                dtype=np.float32,
            ).astype(float).tolist()
            metadata["target_count"] = int(len(self.task_state.get("target_positions", [])))

    def step(self, actions: dict[str, np.ndarray | int]):
        self._invalidate_runtime_caches()
        if self.current_scenario is None:
            raise RuntimeError("reset() must be called before step().")

        if self.action_mode == "candidate_index_legacy":
            raw_waypoints, validity, safety_info = self._legacy_actions_to_waypoints(actions)
            safety_result = self.safety.validate_waypoints(
                self.world,
                raw_waypoints,
                config=self._action_adapter_config(),
                require_connectivity=self.current_scenario.name == "connectivity_expansion" and bool(self.config.get("connectivity_action_filter", False)),
            )
            validity &= safety_result.valid_mask
        else:
            raw_waypoints, action_shape_valid = self._actions_to_waypoints(actions)
            if self.action_mode == "waypoint_delta":
                raw_waypoints = self.world.get_uav_positions() + raw_waypoints
            require_connectivity = self.current_scenario.name == "connectivity_expansion" and bool(self.config.get("connectivity_action_filter", False))
            raw_waypoints_2d = raw_waypoints[..., :2].astype(np.float32)
            action_connectivity_info = self._action_connectivity_diagnostics(raw_waypoints_2d)
            safety_result = self.safety.validate_waypoints(
                self.world,
                raw_waypoints_2d,
                config=self._action_adapter_config(),
                require_connectivity=require_connectivity,
            )
            validity = safety_result.valid_mask & action_shape_valid
            raw_waypoints = raw_waypoints_2d
            raw_waypoints[~action_shape_valid] = self.world.get_uav_positions()[~action_shape_valid]
            if not np.all(validity) and bool(self.config.get("strict_action_validation", False)):
                raise ValueError(f"Illegal waypoint action: {safety_result.to_info()}")
        if self.action_mode == "candidate_index_legacy":
            action_connectivity_info = self._action_connectivity_diagnostics(raw_waypoints[..., :2].astype(np.float32))
        safety_result.valid_mask &= validity
        safety_result.rejected_mask |= ~validity
        safety_info = safety_result.to_info()

        adapter_output = self.action_adapter.adapt(self.world, safety_result.safe_waypoints, safety_result)
        self.last_raw_waypoints = raw_waypoints[..., :2].copy()
        self.last_selected_waypoints = safety_result.safe_waypoints.copy()
        prev_world = self.world.snapshot_for_reward()
        self.current_scenario.step_update(self.world, self.task_state)
        dynamics_info = self.dynamics_backend.step(self.world, adapter_output.controls, adapter_output, safety_result)
        transition_info = self._merge_transition_info(safety_result, adapter_output, dynamics_info)
        transition_info["invalid_action_count"] = int((~validity).sum())
        transition_info["connectivity_action_filter_enabled"] = bool(
            self.current_scenario.name == "connectivity_expansion"
            and bool(self.config.get("connectivity_action_filter", False))
        )
        transition_info["connectivity_filter_replacement_count"] = int(
            safety_info.get("connectivity_action_rejected_count", 0)
        )
        transition_info.update(action_connectivity_info)
        reward_result = self.current_scenario.compute_rewards(prev_world, self.world, self.task_state, transition_info)
        self.last_transition_info = transition_info
        self.last_reward_result = reward_result
        if (
            self.heatmap_observation_enabled
            and self.heatmap_provider_enabled
            and self.heatmap_provider_refresh_on_step
        ):
            self._refresh_task_element_heatmap()
        if self._exposes_candidates():
            self._refresh_candidates()
        self._invalidate_runtime_caches()

        success = bool(reward_result.get("success", False))
        failure = bool(reward_result.get("failure", False))
        terminated = success or failure
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
            for agent in self.possible_agents:
                infos[agent]["terminal_observation"] = terminal_observations[agent]
                infos[agent]["terminal_info"] = deepcopy(infos[agent])
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def step_global(self, actions: dict[str, np.ndarray | int]):
        """Trainer fast path that avoids building per-agent observations.

        The public PettingZoo-style ``step`` API remains unchanged. Rollout
        collection only consumes centralized global observations and one info
        dictionary per environment, so this path skips the expensive
        per-agent observation/info expansion used by external callers.
        """
        self._invalidate_runtime_caches()
        if self.current_scenario is None:
            raise RuntimeError("reset() must be called before step().")

        if self.action_mode == "candidate_index_legacy":
            raw_waypoints, validity, safety_info = self._legacy_actions_to_waypoints(actions)
            safety_result = self.safety.validate_waypoints(
                self.world,
                raw_waypoints,
                config=self._action_adapter_config(),
                require_connectivity=self.current_scenario.name == "connectivity_expansion" and bool(self.config.get("connectivity_action_filter", False)),
            )
            validity &= safety_result.valid_mask
            action_connectivity_info = self._action_connectivity_diagnostics(raw_waypoints[..., :2].astype(np.float32))
        else:
            raw_waypoints, action_shape_valid = self._actions_to_waypoints(actions)
            if self.action_mode == "waypoint_delta":
                raw_waypoints = self.world.get_uav_positions() + raw_waypoints
            require_connectivity = self.current_scenario.name == "connectivity_expansion" and bool(self.config.get("connectivity_action_filter", False))
            raw_waypoints_2d = raw_waypoints[..., :2].astype(np.float32)
            action_connectivity_info = self._action_connectivity_diagnostics(raw_waypoints_2d)
            safety_result = self.safety.validate_waypoints(
                self.world,
                raw_waypoints_2d,
                config=self._action_adapter_config(),
                require_connectivity=require_connectivity,
            )
            validity = safety_result.valid_mask & action_shape_valid
            raw_waypoints = raw_waypoints_2d
            raw_waypoints[~action_shape_valid] = self.world.get_uav_positions()[~action_shape_valid]
            if not np.all(validity) and bool(self.config.get("strict_action_validation", False)):
                raise ValueError(f"Illegal waypoint action: {safety_result.to_info()}")

        safety_result.valid_mask &= validity
        safety_result.rejected_mask |= ~validity
        safety_info = safety_result.to_info()
        adapter_output = self.action_adapter.adapt(self.world, safety_result.safe_waypoints, safety_result)
        self.last_raw_waypoints = raw_waypoints[..., :2].copy()
        self.last_selected_waypoints = safety_result.safe_waypoints.copy()
        prev_world = self.world.snapshot_for_reward()
        self.current_scenario.step_update(self.world, self.task_state)
        dynamics_info = self.dynamics_backend.step(self.world, adapter_output.controls, adapter_output, safety_result)
        transition_info = self._merge_transition_info(safety_result, adapter_output, dynamics_info)
        transition_info["invalid_action_count"] = int((~validity).sum())
        transition_info["connectivity_action_filter_enabled"] = bool(
            self.current_scenario.name == "connectivity_expansion"
            and bool(self.config.get("connectivity_action_filter", False))
        )
        transition_info["connectivity_filter_replacement_count"] = int(
            safety_info.get("connectivity_action_rejected_count", 0)
        )
        transition_info.update(action_connectivity_info)
        reward_result = self.current_scenario.compute_rewards(prev_world, self.world, self.task_state, transition_info)
        self.last_transition_info = transition_info
        self.last_reward_result = reward_result
        if (
            self.heatmap_observation_enabled
            and self.heatmap_provider_enabled
            and self.heatmap_provider_refresh_on_step
        ):
            self._refresh_task_element_heatmap()
        if self._exposes_candidates():
            self._refresh_candidates()
        self._invalidate_runtime_caches()

        success = bool(reward_result.get("success", False))
        failure = bool(reward_result.get("failure", False))
        terminated = success or failure
        truncated = bool(self.world.step_count >= self.world.max_steps)
        current_global = self.get_global_state()
        info = self._build_shared_info()
        if terminated or truncated:
            info = dict(info)
            info["terminal_info"] = dict(info)
            info["terminal_global_state"] = {key: value.copy() for key, value in current_global.items()}
            self.agents = []
        return (
            current_global,
            current_global,
            float(reward_result.get("team_reward", 0.0)),
            np.asarray(reward_result.get("per_agent_rewards", np.zeros(self.num_agents)), dtype=np.float32),
            terminated,
            truncated,
            terminated or truncated,
            info,
        )

    def _merge_transition_info(self, safety_result, adapter_output, dynamics_info) -> dict[str, Any]:
        transition = dynamics_info.to_transition_info()
        safety_info = safety_result.to_info()
        adapter_info = adapter_output.to_info()
        for key, value in safety_info.items():
            transition[f"safety_{key}"] = value
        for key, value in adapter_info.items():
            transition[f"adapter_{key}"] = value
        transition["safe_waypoints"] = safety_result.safe_waypoints.astype(np.float32)
        transition["selected_waypoints"] = safety_result.safe_waypoints.astype(np.float32)
        transition["adapter_info"] = {
            key: value for key, value in adapter_info.items() if isinstance(value, (int, float, bool, str, np.integer, np.floating, np.bool_))
        }
        transition["dynamics_info"] = {
            key: value for key, value in dynamics_info.to_transition_info().items() if isinstance(value, (int, float, bool, str, np.integer, np.floating, np.bool_))
        }
        transition["safety_info"] = {
            key: value for key, value in safety_info.items() if isinstance(value, (int, float, bool, str, np.integer, np.floating, np.bool_))
        }
        return transition

    def _action_connectivity_diagnostics(self, predicted_positions: np.ndarray) -> dict[str, Any]:
        if self.current_scenario is None or self.current_scenario.name != "connectivity_expansion":
            return {}
        if not bool(self.current_scenario.cfg("action_risk_use_predicted_graph", self.config.get("action_risk_use_predicted_graph", False))):
            return {}
        positions = np.asarray(predicted_positions, dtype=np.float32)
        connected = self.world.connected_to_base_mask_for_positions(positions)
        violation = 1.0 - float(np.mean(connected)) if len(connected) else 0.0
        disconnected = ~connected
        return {
            "predicted_positions": positions.astype(np.float32),
            "predicted_connected_mask": connected.astype(bool),
            "predicted_disconnected_mask": disconnected.astype(bool),
            "predicted_connected_to_base_ratio": float(np.mean(connected)) if len(connected) else 1.0,
            "predicted_connectivity_violation": float(violation),
            "action_disconnect_risk": float(violation),
            "unsafe_action_ratio": float(np.mean(disconnected)) if len(disconnected) else 0.0,
        }

    def _actions_to_waypoints(self, actions: dict[str, np.ndarray | int]) -> tuple[np.ndarray, np.ndarray]:
        strict = bool(self.config.get("strict_action_validation", False))
        waypoints = self.world.get_uav_positions().copy()
        valid = np.ones(self.num_agents, dtype=bool)
        for idx, agent in enumerate(self.possible_agents):
            if agent not in actions:
                valid[idx] = False
                if strict:
                    raise ValueError(f"Missing action for {agent}")
                continue
            raw = np.asarray(actions[agent], dtype=np.float32)
            if raw.shape != (self.waypoint_dim,):
                valid[idx] = False
                if strict:
                    raise ValueError(f"Waypoint action for {agent} must have shape {(self.waypoint_dim,)}, got {raw.shape}")
                continue
            if not np.isfinite(raw).all():
                valid[idx] = False
                if strict:
                    raise ValueError(f"Waypoint action for {agent} contains non-finite values")
                continue
            waypoints[idx] = raw[:2]
        return waypoints.astype(np.float32), valid.astype(bool)

    def _legacy_actions_to_waypoints(self, actions: dict[str, np.ndarray | int]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        if self.current_scenario is None:
            raise RuntimeError("reset() must be called before step().")
        if not self.last_candidate_mask.any():
            self._refresh_candidates()
        strict = bool(self.config.get("strict_action_validation", False))
        action_array = np.zeros(self.num_agents, dtype=np.int64)
        valid = np.ones(self.num_agents, dtype=bool)
        fallback = self._fallback_indices()
        for idx, agent in enumerate(self.possible_agents):
            raw = actions.get(agent, int(fallback[idx]))
            action = int(raw)
            if action < 0 or action >= self.candidate_count or not bool(self.last_candidate_mask[idx, action]):
                if strict:
                    raise ValueError(f"Illegal legacy candidate-index action {action} for {agent}")
                action = int(fallback[idx])
                valid[idx] = False
            action_array[idx] = action
        selected = self.last_candidate_waypoints[np.arange(self.num_agents), action_array].copy()
        return selected.astype(np.float32), valid, {"legacy_invalid_index_count": int((~valid).sum())}

    def _fallback_indices(self) -> np.ndarray:
        out = np.zeros(self.num_agents, dtype=np.int64)
        for idx in range(self.num_agents):
            valid = np.where(self.last_candidate_mask[idx])[0]
            out[idx] = int(valid[0]) if len(valid) else 0
        return out

    def _refresh_candidates(self) -> None:
        if self.current_scenario is None:
            return
        candidates, raw_mask = self.current_scenario.generate_candidate_waypoints(self.world, self.task_state)
        require_connectivity = self.current_scenario.name == "connectivity_expansion" and bool(self.config.get("connectivity_candidate_filter", True))
        candidates, mask = self.safety.filter_candidates(self.world, candidates, raw_mask, require_connectivity=require_connectivity)
        if candidates.shape[1] != self.candidate_count:
            padded = np.zeros((self.num_agents, self.candidate_count, 2), dtype=np.float32)
            padded_mask = np.zeros((self.num_agents, self.candidate_count), dtype=bool)
            take = min(self.candidate_count, candidates.shape[1])
            padded[:, :take] = candidates[:, :take]
            padded_mask[:, :take] = mask[:, :take]
            candidates, mask = padded, padded_mask
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
        all_states = global_state["all_uav_states"]
        observations = {}
        for idx, agent in enumerate(self.possible_agents):
            obs = {
                "self_state": all_states[idx].astype(np.float32),
                "neighbor_relative_states": self._neighbor_relative_states(idx),
                "task_field": global_state["task_field"].astype(np.float32),
                "task_id": global_state["task_id"].astype(np.float32),
                "mission_goal": global_state["mission_goal"].astype(np.float32),
                "goal_mask": global_state["goal_mask"].astype(np.float32),
                "global_summary": global_state["global_info"].astype(np.float32),
                "all_uav_states": all_states.astype(np.float32),
                "comm_adjacency": self.world.communication_graph.astype(np.int8),
            }
            if self.heatmap_observation_enabled:
                obs["task_element_heatmap"] = global_state["task_element_heatmap"].astype(np.float32)
            if self._exposes_candidates():
                features = global_state["candidate_features"]
                obs.update(
                    {
                        "candidate_waypoints": self.last_candidate_waypoints[idx].astype(np.float32),
                        "candidate_features": features[idx].astype(np.float32),
                        "candidate_mask": self.last_candidate_mask[idx].astype(np.int8),
                    }
                )
            observations[agent] = obs
        return observations

    def _build_agent_info(self, agent_idx: int) -> dict[str, Any]:
        shared = self._build_shared_info()
        info = dict(shared)
        info["action_validity"] = bool(self.world.uavs[agent_idx].last_action_valid)
        if self._exposes_candidates():
            info["candidate_mask"] = self.last_candidate_mask[agent_idx].copy()
        return info

    def _build_shared_info(self) -> dict[str, Any]:
        if self._shared_info_cache is not None:
            return self._shared_info_cache
        reward_result = self.last_reward_result or {}
        if "metrics" in reward_result:
            metrics = dict(reward_result["metrics"])
        else:
            metrics = dict(self.current_scenario.get_metrics(self.world, self.task_state))
        progress = goal_progress(
            self.current_scenario.name,
            self.current_mission_goal,
            metrics,
        )
        metrics["goal_progress"] = progress
        metrics["goal_achieved"] = bool(reward_result.get("success", metrics.get("success", False)))
        adapter_info = dict(self.last_transition_info.get("adapter_info", {}))
        info = {
            "team_reward": float(reward_result.get("team_reward", 0.0)),
            "per_agent_rewards": np.asarray(reward_result.get("per_agent_rewards", np.zeros(self.num_agents, dtype=np.float32)), dtype=np.float32),
            "reward_components": dict(reward_result.get("components", {})),
            "task_metrics": metrics,
            "selected_waypoints": None if self.last_selected_waypoints is None else self.last_selected_waypoints.copy(),
            "safe_waypoints": None if self.last_selected_waypoints is None else self.last_selected_waypoints.copy(),
            "raw_waypoints": None if self.last_raw_waypoints is None else self.last_raw_waypoints.copy(),
            "action_validity": True,
            "action_interface": self.action_interface,
            "action_mode": self.action_mode,
            "action_adapter": str(self._action_adapter_config().get("name", "waypoint_velocity_tracker")),
            "dynamics_backend": str(self._dynamics_backend_config().get("name", "mpe_core")),
            "adapter_info": adapter_info,
            "dynamics_info": dict(self.last_transition_info.get("dynamics_info", {})),
            "safety_info": dict(self.last_transition_info.get("safety_info", {})),
            "mean_speed": float(self.last_transition_info.get("mean_speed", 0.0)),
            "max_speed_observed": float(self.last_transition_info.get("max_speed_observed", 0.0)),
            "mean_control_norm": float(self.last_transition_info.get("mean_control_norm", 0.0)),
            "mean_distance_to_waypoint_m": float(adapter_info.get("mean_distance_to_waypoint_m", 0.0)),
            "arrival_rate": float(adapter_info.get("arrival_rate", 0.0)),
            "command_update_rate": float(adapter_info.get("command_update_rate", 1.0)),
            "command_reject_rate": float(adapter_info.get("command_reject_rate", 0.0)),
            "mean_desired_speed_mps": float(adapter_info.get("mean_desired_speed_mps", 0.0)),
            "collision_count": int(self.last_transition_info.get("collision_count", 0)),
            "no_fly_violation_count": int(self.last_transition_info.get("no_fly_violation_count", 0)),
            "geofence_violation_count": int(self.last_transition_info.get("geofence_violation_count", 0)),
            "task_name": self.current_scenario.name,
            "mission_goal": self.current_mission_goal.copy(),
            "goal_mask": self.current_goal_mask.copy(),
            "goal_split": self.current_goal_split,
            "goal_id": self.current_goal_id,
            "goal_progress": progress,
            "goal_achieved": bool(reward_result.get("success", metrics.get("success", False))),
            "num_agents": self.num_agents,
            "success": bool(reward_result.get("success", metrics.get("success", False))),
            "failure": bool(reward_result.get("failure", False)),
            "path_length": float(sum(uav.path_length for uav in self.world.uavs)),
            "collision_rate": float(self.last_transition_info.get("safety_violation_count", 0) / max(self.world.step_count * self.num_agents, 1)),
            "invalid_action_count": int(self.last_transition_info.get("invalid_action_count", 0)),
            "connectivity_action_filter_enabled": bool(self.last_transition_info.get("connectivity_action_filter_enabled", False)),
            "connectivity_filter_replacement_count": int(self.last_transition_info.get("connectivity_filter_replacement_count", 0)),
            "candidate_mask_empty_count": int(np.count_nonzero(~self.last_candidate_mask.any(axis=1))) if self._exposes_candidates() else 0,
            "episode_seed": self.seed_value,
            "domain_randomization": deepcopy(self.episode_randomization_info),
            "domain_randomization_enabled": bool(self.episode_randomization_info.get("enabled", False)),
            "episode_base_position": self.world.base_position.copy(),
            "no_fly_zone_count": len(self.world.no_fly_zones),
        }
        info.update(metrics)
        self._shared_info_cache = info
        return info

    def get_global_state(self) -> dict[str, np.ndarray]:
        if self._global_state_cache is not None:
            return self._global_state_cache
        if self.current_scenario is None:
            task_field = np.zeros((5, self.world.grid_size, self.world.grid_size), dtype=np.float32)
            task_id = np.zeros(len(WAYPOINT_TASKS), dtype=np.float32)
        else:
            task_field = self.current_scenario.build_task_field(self.world, self.task_state)
            task_id = self.current_scenario.task_one_hot()
        state = {
            "task_field": task_field.astype(np.float32),
            "global_task_field": task_field.astype(np.float32),
            "all_uav_states": normalized_positions(self.world).astype(np.float32),
            "task_id": task_id.astype(np.float32),
            "mission_goal": self.current_mission_goal.astype(np.float32),
            "goal_mask": self.current_goal_mask.astype(np.float32),
            "global_info": self.world.get_global_summary().astype(np.float32),
            "comm_adjacency": self.world.communication_graph.astype(np.int8),
            "agent_mask": np.ones(self.num_agents, dtype=np.int8),
        }
        if self.heatmap_observation_enabled:
            state["task_element_heatmap"] = self.task_element_heatmap.astype(np.float32)
        if self._exposes_candidates():
            state.update(
                {
                    "candidate_waypoints": self.last_candidate_waypoints.astype(np.float32),
                    "candidate_features": self._candidate_features().astype(np.float32),
                    "candidate_mask": self.last_candidate_mask.astype(np.int8),
                }
            )
        self._global_state_cache = state
        return state

    state = get_global_state

    def render(self, mode: str = "rgb_array"):
        image = render_world(
            self.world,
            self.current_scenario.name if self.current_scenario is not None else "uninitialized",
            self.task_state,
            candidate_waypoints=self.last_candidate_waypoints if (self._exposes_candidates() and bool(self.config.get("render_candidates", True))) else None,
            selected_waypoints=self.last_selected_waypoints,
            metrics=(self.last_reward_result or {}).get("metrics", {}),
            render_detail=str(self.config.get("rendering", {}).get("render_detail", "full")),
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

from __future__ import annotations

import numpy as np
import yaml
from gymnasium import spaces

from envs.waypoint_marl_env import WaypointMultiUAVEnv


def _config(task_name: str = "area_coverage") -> dict:
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["num_agents"] = 3
    config["max_steps"] = 5
    config["task_name"] = task_name
    config["task_names"] = [task_name]
    return config


def test_waypoint_parallel_api_reset_step_and_global_state():
    env = WaypointMultiUAVEnv(_config())
    observations, infos = env.reset(seed=123)
    assert set(observations) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)
    first_obs = observations["uav_0"]
    assert "candidate_waypoints" not in first_obs
    assert "candidate_mask" not in first_obs
    global_state = env.get_global_state()
    assert "candidate_waypoints" not in global_state
    assert "candidate_mask" not in global_state
    assert global_state["mission_goal"].shape == (3,)
    assert global_state["goal_mask"].shape == (3,)
    assert first_obs["mission_goal"].shape == (3,)
    assert first_obs["goal_mask"].shape == (3,)
    assert isinstance(env.action_space("uav_0"), spaces.Box)
    assert env.action_space("uav_0").shape == (2,)
    assert env.observation_space("uav_0").contains(first_obs)
    positions = env.world.get_uav_positions()
    actions = {agent: positions[idx].astype(np.float32) for idx, agent in enumerate(env.possible_agents)}
    next_obs, rewards, terminations, truncations, step_infos = env.step(actions)
    assert set(next_obs) == set(env.possible_agents)
    assert set(rewards) == set(env.possible_agents)
    assert set(terminations) == set(env.possible_agents)
    assert set(truncations) == set(env.possible_agents)
    info = step_infos["uav_0"]
    for key in [
        "team_reward",
        "per_agent_rewards",
        "reward_components",
        "task_metrics",
        "selected_waypoints",
        "safe_waypoints",
        "action_validity",
        "adapter_info",
        "dynamics_info",
        "mean_speed",
        "max_speed_observed",
        "mean_control_norm",
        "mean_distance_to_waypoint_m",
        "arrival_rate",
        "command_update_rate",
        "command_reject_rate",
        "mean_desired_speed_mps",
        "mission_goal",
        "goal_mask",
        "goal_split",
        "goal_progress",
        "goal_achieved",
    ]:
        assert key in info
    assert np.asarray(info["per_agent_rewards"]).shape == (3,)
    assert np.asarray(info["selected_waypoints"]).shape == (3, 2)
    assert info["action_mode"] == "waypoint"
    assert info["action_adapter"] == "waypoint_velocity_tracker"
    assert info["dynamics_backend"] == "mpe_core"
    assert info["dynamics_info"]["uses_real_mpe_core"] is True
    assert info["dynamics_info"]["mpe_world_step_calls"] > 0
    assert info["dynamics_info"]["mpe_source"]
    assert np.isfinite(info["mean_speed"])
    assert np.isfinite(info["max_speed_observed"])
    assert np.isfinite(info["mean_control_norm"])
    assert np.isfinite(info["mean_distance_to_waypoint_m"])
    assert 0.0 <= info["arrival_rate"] <= 1.0
    assert 0.0 <= info["command_update_rate"] <= 1.0
    assert 0.0 <= info["command_reject_rate"] <= 1.0
    assert env.state()["all_uav_states"].shape == (3, 8)

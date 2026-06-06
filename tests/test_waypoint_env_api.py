from __future__ import annotations

import numpy as np
import yaml

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
    assert "candidate_waypoints" in first_obs
    assert "candidate_mask" in first_obs
    assert first_obs["candidate_waypoints"].shape == (env.candidate_count, 2)
    assert first_obs["candidate_mask"].any()
    global_state = env.get_global_state()
    assert global_state["candidate_waypoints"].shape == (3, env.candidate_count, 2)
    assert global_state["candidate_mask"].shape == (3, env.candidate_count)
    assert env.observation_space("uav_0").contains(first_obs)
    actions = {
        agent: int(np.where(env.last_candidate_mask[idx])[0][0])
        for idx, agent in enumerate(env.possible_agents)
    }
    next_obs, rewards, terminations, truncations, step_infos = env.step(actions)
    assert set(next_obs) == set(env.possible_agents)
    assert set(rewards) == set(env.possible_agents)
    assert set(terminations) == set(env.possible_agents)
    assert set(truncations) == set(env.possible_agents)
    info = step_infos["uav_0"]
    for key in ["team_reward", "per_agent_rewards", "reward_components", "task_metrics", "candidate_mask", "selected_waypoints", "action_validity"]:
        assert key in info
    assert np.asarray(info["per_agent_rewards"]).shape == (3,)
    assert env.state()["all_uav_states"].shape == (3, 8)

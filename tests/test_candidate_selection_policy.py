from __future__ import annotations

import torch
import yaml

from policies import build_policy
from utils.waypoint_vector_env import make_waypoint_env_batch


def _configs(num_agents: int):
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    with open("configs/policy/mappo_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        policy_config = yaml.safe_load(handle)
    policy_config["policy_class"] = "candidate_selection_waypoint"
    policy_config["candidate_hidden_dim"] = 32
    policy_config["candidate_count"] = 12
    policy_config["candidate_generator"] = "task_aware"
    env_config["num_agents"] = num_agents
    policy_config["map_size"] = env_config["map_size"]
    policy_config["grid_size"] = env_config["grid_size"]
    policy_config["max_waypoint_distance"] = env_config["max_waypoint_distance"]
    policy_config["base_position"] = env_config["base_position"]
    policy_config["no_fly_zones"] = env_config["no_fly_zones"]
    return env_config, policy_config


def test_candidate_selection_policy_returns_final_waypoints_for_variable_agent_counts():
    for num_agents in [4, 10]:
        env_config, policy_config = _configs(num_agents)
        batch = make_waypoint_env_batch(env_config, 1)
        obs, _ = batch.reset()
        policy = build_policy(policy_config, batch.envs[0].global_observation_space, batch.envs[0].action_space_n)
        obs_t = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}
        output = policy.get_action_and_value(obs_t)
        assert output.env_action.shape == (1, num_agents, 2)
        assert output.train_action.shape == (1, num_agents)
        assert output.logprob.shape == (1, num_agents)
        assert output.entropy.shape == (1, num_agents)
        assert output.value.shape == (1,)
        assert output.train_action.dtype == torch.int64
        selected_mask = output.aux["candidate_mask"].gather(-1, output.train_action.unsqueeze(-1)).squeeze(-1)
        selected_waypoint = output.aux["candidate_waypoints"].gather(
            2,
            output.train_action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2),
        ).squeeze(2)
        assert selected_mask.all()
        assert torch.allclose(output.env_action, selected_waypoint)
        deterministic = policy.act_deterministic(obs_t)
        assert deterministic.shape == (1, num_agents, 2)
        assert torch.isfinite(output.logprob).all()
        assert torch.isfinite(output.entropy).all()
        assert torch.isfinite(output.value).all()
        batch.close()

from __future__ import annotations

import torch
import yaml

from policies import build_policy
from utils.waypoint_vector_env import make_waypoint_env_batch


def test_direct_waypoint_policy_outputs_bounded_final_waypoints():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    with open("configs/policy/direct_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        policy_config = yaml.safe_load(handle)
    env_config["num_agents"] = 5
    policy_config["map_size"] = env_config["map_size"]
    policy_config["max_delta"] = env_config["max_waypoint_distance"]
    batch = make_waypoint_env_batch(env_config, 1)
    obs, _ = batch.reset()
    policy = build_policy(policy_config, batch.envs[0].global_observation_space, batch.envs[0].action_space_n)
    obs_t = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}
    output = policy.get_action_and_value(obs_t)
    assert output.env_action.shape == (1, 5, 2)
    assert output.train_action.shape == (1, 5, 2)
    assert output.logprob.shape == (1, 5)
    assert output.entropy.shape == (1, 5)
    assert output.value.shape == (1,)
    assert torch.isfinite(output.env_action).all()
    assert torch.isfinite(output.logprob).all()
    assert torch.isfinite(output.entropy).all()
    assert torch.isfinite(output.value).all()
    assert torch.all(output.env_action >= 0.0)
    assert torch.all(output.env_action <= float(env_config["map_size"]))
    deterministic = policy.act_deterministic(obs_t)
    assert deterministic.shape == (1, 5, 2)
    batch.close()

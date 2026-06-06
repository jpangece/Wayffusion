from __future__ import annotations

import torch
import yaml

from policies import build_policy
from utils.waypoint_vector_env import make_waypoint_env_batch


def test_mappo_waypoint_policy_shapes_and_masked_actions():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    with open("configs/policy/mappo_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        policy_config = yaml.safe_load(handle)
    env_config["num_agents"] = 4
    batch = make_waypoint_env_batch(env_config, 1)
    obs, _ = batch.reset()
    obs["candidate_mask"][:] = 0
    obs["candidate_mask"][:, :, 2] = 1
    policy = build_policy(policy_config, batch.envs[0].global_observation_space, batch.envs[0].action_space_n)
    obs_t = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}
    action, logprob, entropy, value = policy.get_action_and_value(obs_t)
    assert action.shape == (1, 4)
    assert logprob.shape == (1, 4)
    assert entropy.shape == (1, 4)
    assert value.shape == (1,)
    assert action.dtype == torch.int64
    assert torch.equal(action, torch.full_like(action, 2))
    deterministic = policy.act_deterministic(obs_t)
    assert torch.equal(deterministic, torch.full_like(deterministic, 2))
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()

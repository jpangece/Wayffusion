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
    for key in [
        "delta_raw_norm_mean",
        "delta_raw_norm_max",
        "delta_mean_norm_mean",
        "clipped_delta_norm_mean",
        "clipped_delta_norm_max",
        "delta_clip_fraction",
    ]:
        assert key in output.aux
        assert torch.isfinite(torch.as_tensor(output.aux[key])).all()
    assert 0.0 <= float(output.aux["delta_clip_fraction"]) <= 1.0
    deterministic = policy.act_deterministic(obs_t)
    assert deterministic.shape == (1, 5, 2)
    batch.close()


def test_direct_waypoint_policy_clips_large_train_delta():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    with open("configs/policy/direct_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        policy_config = yaml.safe_load(handle)
    env_config["num_agents"] = 2
    policy_config["map_size"] = env_config["map_size"]
    policy_config["max_delta"] = 0.10
    batch = make_waypoint_env_batch(env_config, 1)
    obs, _ = batch.reset()
    policy = build_policy(policy_config, batch.envs[0].global_observation_space, batch.envs[0].action_space_n)
    obs_t = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}
    large_delta = torch.full((1, 2, 2), 1.0)
    output = policy.get_action_and_value(obs_t, train_action=large_delta)
    assert float(output.aux["delta_clip_fraction"]) == 1.0
    assert float(output.aux["clipped_delta_norm_max"]) <= 0.10001
    batch.close()


def test_direct_waypoint_policy_supports_task_relevant_spatial_context():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    with open("configs/policy/direct_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        policy_config = yaml.safe_load(handle)
    env_config["num_agents"] = 3
    env_config["task_name"] = "belief_search"
    env_config["task_names"] = ["belief_search"]
    policy_config["map_size"] = env_config["map_size"]
    policy_config["max_delta"] = 0.15
    policy_config["task_field_channel_mode"] = "task_relevant"
    policy_config["use_spatial_field_context"] = True
    policy_config["spatial_context_radii"] = [0.05, 0.10]
    batch = make_waypoint_env_batch(env_config, 1)
    obs, _ = batch.reset()
    policy = build_policy(policy_config, batch.envs[0].global_observation_space, batch.envs[0].action_space_n)
    obs_t = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}
    output = policy.get_action_and_value(obs_t)
    assert output.env_action.shape == (1, 3, 2)
    assert output.train_action.shape == (1, 3, 2)
    assert output.logprob.shape == (1, 3)
    assert output.entropy.shape == (1, 3)
    assert output.value.shape == (1,)
    assert torch.isfinite(output.env_action).all()
    assert torch.isfinite(output.logprob).all()
    assert torch.isfinite(output.value).all()
    assert policy.spatial_context_dim > 0
    batch.close()


def test_direct_waypoint_policy_supports_field_moment_context():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    with open("configs/policy/direct_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        policy_config = yaml.safe_load(handle)
    env_config["num_agents"] = 3
    env_config["task_name"] = "priority_inspection"
    env_config["task_names"] = ["priority_inspection"]
    policy_config["map_size"] = env_config["map_size"]
    policy_config["max_delta"] = 0.15
    policy_config["task_field_channel_mode"] = "task_relevant"
    policy_config["use_spatial_field_context"] = True
    policy_config["use_field_moment_context"] = True
    policy_config["field_moment_include_inverse"] = True
    batch = make_waypoint_env_batch(env_config, 1)
    obs, _ = batch.reset()
    policy = build_policy(policy_config, batch.envs[0].global_observation_space, batch.envs[0].action_space_n)
    obs_t = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}
    output = policy.get_action_and_value(obs_t)
    assert output.env_action.shape == (1, 3, 2)
    assert output.logprob.shape == (1, 3)
    assert output.value.shape == (1,)
    assert torch.isfinite(output.env_action).all()
    assert torch.isfinite(output.logprob).all()
    assert torch.isfinite(output.value).all()
    assert policy.field_moment_dim > 0
    batch.close()


def test_direct_waypoint_policy_supports_existing_comm_graph_encoder():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    with open("configs/policy/direct_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        policy_config = yaml.safe_load(handle)
    env_config["num_agents"] = 4
    env_config["task_name"] = "connectivity_expansion"
    env_config["task_names"] = ["connectivity_expansion"]
    env_config["task_sampling_probs"] = {"connectivity_expansion": 1.0}
    policy_config["map_size"] = env_config["map_size"]
    policy_config["max_delta"] = 0.15
    policy_config["use_comm_graph_encoder"] = True
    policy_config["use_connectivity_auxiliary_loss"] = True
    batch = make_waypoint_env_batch(env_config, 1)
    obs, _ = batch.reset()
    assert "comm_adjacency" in obs
    policy = build_policy(policy_config, batch.envs[0].global_observation_space, batch.envs[0].action_space_n)
    obs_t = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}
    output = policy.get_action_and_value(obs_t)
    loss = output.value.mean() + output.logprob.mean() + output.aux["aux_pred_connectivity_violation"].mean()
    loss.backward()
    graph_grads = [
        param.grad
        for name, param in policy.named_parameters()
        if name.startswith("comm_graph_update")
    ]
    assert output.env_action.shape == (1, 4, 2)
    assert torch.isfinite(output.env_action).all()
    assert graph_grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in graph_grads)
    batch.close()

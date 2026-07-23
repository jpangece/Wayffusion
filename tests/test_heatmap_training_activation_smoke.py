from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml

from algorithms.mappo_waypoint import MAPPOWaypointTrainer
from envs.waypoint.priority_inspection_task_element_provider import (
    register_priority_inspection_task_element_provider,
)
from envs.waypoint.task_element_provider import clear_task_element_provider_registry
from policies import build_policy
from scripts import train_mappo_waypoint
from utils.waypoint_vector_env import make_waypoint_env_batch


@pytest.fixture(autouse=True)
def isolated_provider_registry():
    clear_task_element_provider_registry()
    yield
    clear_task_element_provider_registry()


def _environment_config() -> dict:
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.update(
        {
            "seed": 83,
            "task_name": "priority_inspection",
            "task_names": ["priority_inspection"],
            "num_agents": 2,
            "grid_size": 8,
            "max_steps": 16,
        }
    )
    config["domain_randomization"]["enabled"] = False
    config["tasks"]["priority_inspection"]["num_pois"] = 4
    config["tasks"]["priority_inspection"]["num_pois_range"] = None
    config["heatmap_observation"] = {
        "enabled": True,
        "channels": 1,
        "task_elements": [{"x": 0, "y": 0, "priority": 1.0}],
        "provider": {
            "enabled": True,
            "refresh_on_step": True,
        },
    }
    return config


def _policy_config(*, use_heatmap: bool) -> dict:
    with open("configs/policy/direct_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.update(
        {
            "policy_class": "direct_waypoint",
            "use_task_element_heatmap": use_heatmap,
            "cnn_channels": [4, 8],
            "agent_hidden_dim": 16,
            "joint_hidden_dim": 24,
            "attention_heads": 1,
            "use_attention": False,
            "use_spatial_field_context": False,
            "use_field_moment_context": False,
            "use_connectivity_auxiliary_loss": False,
            "use_comm_graph_encoder": False,
            "num_envs": 1,
            "rollout_steps": 2,
            "total_updates": 1,
            "epochs": 1,
            "minibatch_size": 2,
            "learning_rate": 1.0e-3,
            "reward_norm": False,
            "advantage_norm": True,
            "lr_schedule": "constant",
        }
    )
    return config


def _build_training_stack(*, use_heatmap: bool):
    env_config = _environment_config()
    policy_config = _policy_config(use_heatmap=use_heatmap)
    policy_config["map_size"] = env_config["map_size"]
    policy_config["grid_size"] = env_config["grid_size"]
    policy_config["max_delta"] = env_config["max_waypoint_distance"]
    policy_config["max_waypoint_distance"] = env_config["max_waypoint_distance"]
    register_priority_inspection_task_element_provider()
    env_batch = make_waypoint_env_batch(env_config, num_envs=1, backend="sync")
    observations, _ = env_batch.reset(seeds=[83])
    policy = build_policy(
        policy_config,
        env_batch.envs[0].global_observation_space,
        env_batch.envs[0].action_space_n,
    )
    trainer = MAPPOWaypointTrainer(env_batch, policy, policy_config, device="cpu")
    return env_batch, observations, policy, trainer


def test_enabled_heatmap_completes_real_rollout_and_optimizer_update():
    env_batch, observations, policy, trainer = _build_training_stack(use_heatmap=True)
    try:
        heatmap = observations["task_element_heatmap"]
        declared = env_batch.envs[0].global_observation_space["task_element_heatmap"]
        assert heatmap.shape == (1, 1, 8, 8)
        assert declared.shape == (1, 8, 8)
        assert heatmap.dtype == np.float32
        assert np.isfinite(heatmap).all()
        assert policy.use_task_element_heatmap is True
        assert hasattr(policy, "heatmap_encoder")

        obs_t = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in observations.items()
        }
        output = policy.get_action_and_value(obs_t)
        assert output.env_action.shape == (1, 2, 2)
        assert output.value.shape == (1,)

        rollout, _ = trainer.collect_rollout()
        assert "task_element_heatmap" in rollout.observations
        stored_heatmap = rollout.observations["task_element_heatmap"]
        assert stored_heatmap.shape == (2, 1, 1, 8, 8)
        assert np.isfinite(stored_heatmap).all()

        before = {
            name: parameter.detach().clone()
            for name, parameter in policy.heatmap_encoder.named_parameters()
        }
        update_stats = trainer.update(rollout)
        for key in ["policy_loss", "value_loss", "entropy", "grad_norm"]:
            assert np.isfinite(update_stats[key])
        changed = [
            not torch.allclose(before[name], parameter.detach(), rtol=0.0, atol=1.0e-10)
            for name, parameter in policy.heatmap_encoder.named_parameters()
        ]
        assert any(changed)
        gradients = [parameter.grad for parameter in policy.heatmap_encoder.parameters()]
        if any(gradient is not None for gradient in gradients):
            assert all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in gradients
            )
    finally:
        env_batch.close()


def test_disabled_policy_completes_same_rollout_and_update_without_encoder():
    env_batch, observations, policy, trainer = _build_training_stack(use_heatmap=False)
    try:
        assert "task_element_heatmap" in observations
        assert policy.use_task_element_heatmap is False
        assert not hasattr(policy, "heatmap_encoder")

        rollout, _ = trainer.collect_rollout()
        assert rollout.observations["task_element_heatmap"].shape == (2, 1, 1, 8, 8)
        update_stats = trainer.update(rollout)
        for key in ["policy_loss", "value_loss", "entropy", "grad_norm"]:
            assert np.isfinite(update_stats[key])
    finally:
        env_batch.close()


def test_unregistered_provider_preserves_static_fallback():
    env_config = _environment_config()
    env_batch = make_waypoint_env_batch(env_config, num_envs=1, backend="sync")
    try:
        observations, _ = env_batch.reset(seeds=[83])
        heatmap = observations["task_element_heatmap"]
        expected = np.zeros((1, 1, 8, 8), dtype=np.float32)
        expected[0, 0, 0, 0] = 1.0
        assert np.array_equal(heatmap, expected)
    finally:
        env_batch.close()


def test_launcher_registers_provider_for_explicit_heatmap_activation():
    env_config = _environment_config()

    register_hook = train_mappo_waypoint.register_heatmap_task_element_providers
    register_hook(deepcopy(env_config))
    env_batch = make_waypoint_env_batch(env_config, num_envs=1, backend="sync")
    try:
        observations, _ = env_batch.reset(seeds=[83])
        static = np.zeros((1, 1, 8, 8), dtype=np.float32)
        static[0, 0, 0, 0] = 1.0
        assert not np.array_equal(observations["task_element_heatmap"], static)
    finally:
        env_batch.close()

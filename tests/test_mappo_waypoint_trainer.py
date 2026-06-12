from __future__ import annotations

import numpy as np
import pytest
import yaml

from algorithms.mappo_waypoint import MAPPOWaypointTrainer, compute_gae_returns
from policies import build_policy
from utils.waypoint_vector_env import make_waypoint_env_batch


def _base_env_config():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    env_config["num_agents"] = 3
    env_config["max_steps"] = 1
    env_config["task_name"] = "area_coverage"
    env_config["task_names"] = ["area_coverage"]
    env_config["tasks"]["area_coverage"]["success_ratio"] = 2.0
    return env_config


def _train_config(path: str, policy_class: str | None = None):
    with open(path, "r", encoding="utf-8") as handle:
        train_config = yaml.safe_load(handle)
    if policy_class is not None:
        train_config["policy_class"] = policy_class
    if train_config.get("policy_class") == "candidate_selection_waypoint":
        train_config.setdefault("candidate_hidden_dim", 32)
        train_config.setdefault("candidate_count", 12)
        train_config.setdefault("candidate_generator", "task_aware")
    train_config["num_envs"] = 2
    train_config["rollout_steps"] = 4
    train_config["total_updates"] = 1
    train_config["epochs"] = 1
    train_config["minibatch_size"] = 4
    return train_config


def _run_collect_update(env_config: dict, train_config: dict):
    train_config["map_size"] = env_config["map_size"]
    train_config["grid_size"] = env_config["grid_size"]
    train_config["max_waypoint_distance"] = env_config["max_waypoint_distance"]
    train_config["base_position"] = env_config["base_position"]
    train_config["no_fly_zones"] = env_config["no_fly_zones"]
    batch_env = make_waypoint_env_batch(env_config, 2)
    policy = build_policy(train_config, batch_env.envs[0].global_observation_space, batch_env.envs[0].action_space_n)
    trainer = MAPPOWaypointTrainer(batch_env, policy, train_config)
    batch, stats = trainer.collect_rollout()
    assert batch.old_logprobs.shape == (4, 2, 3)
    assert batch.env_actions.shape == (4, 2, 3, 2)
    assert batch.train_actions.shape[0:3] == (4, 2, 3)
    assert batch.values.shape == (4, 2)
    assert batch.truncated.any()
    assert np.all(batch.bootstrap_mask[batch.truncated] == 1.0)
    assert np.isfinite(stats["mean_speed"])
    assert np.isfinite(stats["max_speed_observed"])
    assert np.isfinite(stats["mean_control_norm"])
    assert np.isfinite(stats["mean_distance_to_waypoint_m"])
    assert np.isfinite(stats["mean_desired_speed_mps"])
    assert np.isfinite(stats["delta_clip_fraction"])
    assert np.isfinite(stats["delta_raw_norm_mean"])
    assert np.isfinite(stats["clipped_delta_norm_mean"])
    assert 0.0 <= stats["arrival_rate"] <= 1.0
    assert 0.0 <= stats["command_update_rate"] <= 1.0
    assert 0.0 <= stats["command_reject_rate"] <= 1.0
    assert stats["uses_real_mpe_core"] == 1.0
    assert stats["mpe_world_step_calls"] > 0.0
    update_stats = trainer.update(batch)
    for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm"]:
        assert np.isfinite(update_stats[key])
    assert stats["candidate_mask_empty_count"] == 0.0
    batch_env.close()


def test_mappo_waypoint_collect_update_and_truncated_bootstrap_candidate_selection():
    _run_collect_update(_base_env_config(), _train_config("configs/policy/mappo_waypoint_debug.yaml", policy_class="candidate_selection_waypoint"))


def test_mappo_waypoint_collect_update_direct_waypoint():
    _run_collect_update(_base_env_config(), _train_config("configs/policy/direct_waypoint_debug.yaml"))


def test_mappo_waypoint_collect_update_uvfa_direct_waypoint():
    config = _train_config("configs/policy/direct_waypoint_debug.yaml")
    config["policy_class"] = "direct_waypoint_uvfa"
    config["reward_norm_scope"] = "task"
    _run_collect_update(_base_env_config(), config)


def test_batched_fast_env_backend_collect_update_direct_waypoint():
    env_config = _base_env_config()
    env_config["dynamics_backend"] = {
        "name": "waypoint_behavior_fast",
        "dt": 1.0,
        "max_speed": float(env_config.get("max_speed", 0.04)),
        "acceptance_radius": 0.025,
        "agent_size": 0.02,
        "enable_boundary_projection": True,
        "enable_no_fly_projection": True,
    }
    env_config["runtime_acceleration"] = {"sensor_footprint_mode": "disk_stamp"}
    train_config = _train_config("configs/policy/direct_waypoint_debug.yaml")
    train_config["map_size"] = env_config["map_size"]
    train_config["grid_size"] = env_config["grid_size"]
    train_config["max_waypoint_distance"] = env_config["max_waypoint_distance"]
    train_config["base_position"] = env_config["base_position"]
    train_config["no_fly_zones"] = env_config["no_fly_zones"]
    batch_env = make_waypoint_env_batch(env_config, 2, backend="batched_fast")
    policy = build_policy(train_config, batch_env.envs[0].global_observation_space, batch_env.envs[0].action_space_n)
    trainer = MAPPOWaypointTrainer(batch_env, policy, train_config)
    batch, stats = trainer.collect_rollout()
    assert batch.old_logprobs.shape == (4, 2, 3)
    assert batch.env_actions.shape == (4, 2, 3, 2)
    assert np.isfinite(stats["mean_speed"])
    assert stats["uses_real_mpe_core"] == 0.0
    update_stats = trainer.update(batch)
    assert np.isfinite(update_stats["policy_loss"])
    batch_env.close()


def test_batched_fast_env_backend_rejects_non_fast_dynamics():
    env_config = _base_env_config()
    with pytest.raises(ValueError, match="waypoint_behavior_fast"):
        make_waypoint_env_batch(env_config, 2, backend="batched_fast")


def test_gae_truncated_bootstraps_value_but_does_not_cross_episode():
    rewards = np.array([[1.0], [2.0]], dtype=np.float32)
    values = np.array([[0.5], [0.25]], dtype=np.float32)
    next_values = np.array([[10.0], [4.0]], dtype=np.float32)
    terminated = np.array([[False], [False]], dtype=bool)
    done_for_reset = np.array([[True], [False]], dtype=bool)
    advantages, returns = compute_gae_returns(
        rewards,
        values,
        next_values,
        terminated,
        done_for_reset,
        gamma=0.9,
        gae_lambda=0.95,
    )

    expected_t1 = 2.0 + 0.9 * 4.0 - 0.25
    expected_t0 = 1.0 + 0.9 * 10.0 - 0.5
    assert np.allclose(advantages[1, 0], expected_t1)
    assert np.allclose(advantages[0, 0], expected_t0)
    assert np.allclose(returns, advantages + values)


def test_gae_terminated_does_not_bootstrap_value():
    rewards = np.array([[1.0]], dtype=np.float32)
    values = np.array([[0.5]], dtype=np.float32)
    next_values = np.array([[10.0]], dtype=np.float32)
    terminated = np.array([[True]], dtype=bool)
    done_for_reset = np.array([[True]], dtype=bool)
    advantages, _ = compute_gae_returns(
        rewards,
        values,
        next_values,
        terminated,
        done_for_reset,
        gamma=0.9,
        gae_lambda=0.95,
    )
    assert np.allclose(advantages[0, 0], 0.5)

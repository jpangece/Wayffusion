from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import yaml

from algorithms.mappo_waypoint import MAPPOWaypointTrainer
from envs.waypoint.mission_goals import goal_values_for_split
from envs.waypoint_marl_env import WaypointMultiUAVEnv
from policies import build_policy
from utils.waypoint_vector_env import make_waypoint_env_batch


STATIC_TASKS = [
    "area_coverage",
    "belief_search",
    "priority_inspection",
    "connectivity_expansion",
]


def _env_config(task_name: str = "area_coverage") -> dict:
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["num_agents"] = 3
    config["max_steps"] = 10
    config["task_name"] = task_name
    config["task_names"] = [task_name]
    config["mission_goals"]["train_split"] = "train"
    return config


def _policy_config(policy_class: str) -> dict:
    with open("configs/policy/direct_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["policy_class"] = policy_class
    config["goal_embedding_dim"] = 24
    config["goal_hidden_dim"] = 32
    return config


def test_episode_goal_sampling_is_reproducible_and_forced_goal_wins():
    env = WaypointMultiUAVEnv(_env_config())
    env.reset(seed=19)
    first = env.current_mission_goal.copy()
    first_id = env.current_goal_id
    env.reset(seed=19)
    assert np.allclose(env.current_mission_goal, first)
    assert env.current_goal_id == first_id

    forced = np.asarray([0.67, 0.0, 0.0], dtype=np.float32)
    observations, infos = env.reset(
        seed=19,
        options={"goal_split": "interpolation", "mission_goal": forced},
    )
    assert np.allclose(env.current_mission_goal, forced)
    assert np.allclose(observations["uav_0"]["mission_goal"], forced)
    assert infos["uav_0"]["goal_split"] == "interpolation"
    env.close()


def test_goal_sets_exclude_formal_targets_from_training_sets():
    config = _env_config()
    for task_name in STATIC_TASKS:
        train = goal_values_for_split(task_name, config, "train")
        formal = goal_values_for_split(task_name, config, "formal")[0]
        assert all(not np.allclose(item, formal) for item in train)


def test_four_tasks_use_forced_episode_goal_for_success():
    for task_name in STATIC_TASKS:
        config = _env_config(task_name)
        env = WaypointMultiUAVEnv(config)
        if task_name == "connectivity_expansion":
            forced = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            forced = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        env.reset(seed=4, options={"goal_split": "formal", "mission_goal": forced})
        positions = env.world.get_uav_positions()
        actions = {agent: positions[idx] for idx, agent in enumerate(env.possible_agents)}
        _, _, terminations, _, infos = env.step(actions)
        assert all(terminations.values()), task_name
        assert infos["uav_0"]["goal_achieved"] is True
        assert infos["uav_0"]["goal_progress"] == 1.0
        env.close()


def test_uvfa_policy_goal_stream_changes_actor_and_value_and_receives_gradients():
    config = _env_config()
    batch = make_waypoint_env_batch(config, 2)
    obs, _ = batch.reset(seeds=[3, 7])
    policy = build_policy(
        _policy_config("direct_waypoint_uvfa"),
        batch.envs[0].global_observation_space,
        batch.envs[0].action_space_n,
    )
    obs_t = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}
    changed = {key: value.clone() for key, value in obs_t.items()}
    changed["mission_goal"][:, 0] = torch.clamp(changed["mission_goal"][:, 0] - 0.2, min=0.05)

    mean_a, value_a, _ = policy.forward(obs_t)
    mean_b, value_b, _ = policy.forward(changed)
    assert not torch.allclose(mean_a, mean_b)
    assert not torch.allclose(value_a, value_b)

    loss = mean_a.square().mean() + value_a.square().mean()
    loss.backward()
    goal_grads = [
        parameter.grad
        for name, parameter in policy.named_parameters()
        if name.startswith("goal_encoder")
    ]
    assert goal_grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in goal_grads)
    batch.close()


def test_legacy_direct_policy_state_dict_remains_strictly_compatible():
    config = _env_config()
    batch = make_waypoint_env_batch(config, 1)
    policy_config = _policy_config("direct_waypoint")
    first = build_policy(policy_config, batch.envs[0].global_observation_space, batch.envs[0].action_space_n)
    second = build_policy(policy_config, batch.envs[0].global_observation_space, batch.envs[0].action_space_n)
    second.load_state_dict(deepcopy(first.state_dict()), strict=True)
    assert not any(name.startswith("goal_encoder") for name, _ in first.named_parameters())
    batch.close()


def test_task_scoped_reward_normalization_keeps_independent_statistics():
    trainer = object.__new__(MAPPOWaypointTrainer)
    trainer.train_config = {"reward_norm_scope": "task"}
    trainer.reward_rms_by_task = {}
    trainer.reward_rms = None
    rewards = np.asarray([1.0, 3.0, 100.0, 120.0], dtype=np.float32)
    infos = [
        {"task_name": "area_coverage"},
        {"task_name": "area_coverage"},
        {"task_name": "belief_search"},
        {"task_name": "belief_search"},
    ]
    normalized = trainer._normalize_rewards(rewards, infos)
    assert np.isfinite(normalized).all()
    assert set(trainer.reward_rms_by_task) == {"area_coverage", "belief_search"}
    assert trainer.reward_rms_by_task["area_coverage"].mean < 10.0
    assert trainer.reward_rms_by_task["belief_search"].mean > 50.0

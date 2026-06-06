from __future__ import annotations

import numpy as np
import yaml

from algorithms.mappo_waypoint import MAPPOWaypointTrainer
from policies import build_policy
from utils.waypoint_vector_env import make_waypoint_env_batch


def test_mappo_waypoint_collect_update_and_truncated_bootstrap():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    with open("configs/policy/mappo_waypoint_debug.yaml", "r", encoding="utf-8") as handle:
        train_config = yaml.safe_load(handle)
    env_config["num_agents"] = 3
    env_config["max_steps"] = 1
    env_config["task_name"] = "area_coverage"
    env_config["task_names"] = ["area_coverage"]
    env_config["tasks"]["area_coverage"]["success_ratio"] = 2.0
    train_config["num_envs"] = 2
    train_config["rollout_steps"] = 4
    train_config["total_updates"] = 1
    train_config["epochs"] = 1
    train_config["minibatch_size"] = 4
    batch_env = make_waypoint_env_batch(env_config, 2)
    policy = build_policy(train_config, batch_env.envs[0].global_observation_space, batch_env.envs[0].action_space_n)
    trainer = MAPPOWaypointTrainer(batch_env, policy, train_config)
    batch, stats = trainer.collect_rollout()
    assert batch.old_logprobs.shape == (4, 2, 3)
    assert batch.actions.shape == (4, 2, 3)
    assert batch.values.shape == (4, 2)
    assert batch.truncated.any()
    assert np.all(batch.bootstrap_mask[batch.truncated] == 1.0)
    update_stats = trainer.update(batch)
    for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm"]:
        assert np.isfinite(update_stats[key])
    assert stats["candidate_mask_empty_count"] == 0.0
    batch_env.close()

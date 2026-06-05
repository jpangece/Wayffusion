import math

from algorithms import MAPPOTrainer
from policies import build_policy
from scripts._common import load_generic_config, prepare_env_config
from utils import make_env_batch


def test_mappo_collect_rollout_stores_per_agent_logprobs_and_updates():
    env_config = prepare_env_config(
        "configs/env/multitask.yaml",
        tasks=["goal_nav"],
        num_agents=4,
        scaling_mode="fixed_map",
        extra_override={"max_steps": 8},
    )
    env_batch = make_env_batch(env_config, num_envs=2, backend="sync")
    train_config = load_generic_config("configs/policy/mappo_shared.yaml")
    train_config.update(
        {
            "num_envs": 2,
            "rollout_steps": 2,
            "total_updates": 1,
            "epochs": 1,
            "minibatch_size": 4,
            "reward_norm": False,
            "advantage_norm": True,
            "learning_rate": 1e-4,
        }
    )
    policy = build_policy(train_config, env_batch.envs[0].observation_space, env_batch.envs[0].action_space)
    trainer = MAPPOTrainer(env_batch, policy, train_config, device="cpu")

    batch, rollout_stats = trainer.collect_rollout()
    assert batch.logprobs.shape == (2, 2, 4)
    assert batch.actions.shape == (2, 2, 4, 2)
    assert "mean_rollout_reward" in rollout_stats

    update_stats = trainer.update(batch)
    for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "ratio_mean", "grad_norm"]:
        assert key in update_stats
        assert math.isfinite(float(update_stats[key]))

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from algorithms.mappo_waypoint import WaypointRolloutBuffer
from envs.waypoint_marl_env import WaypointMultiUAVEnv
from policies import build_policy
from policies.candidate_selection_policy import CandidateSelectionWaypointPolicy
from policies.direct_waypoint_policy import DirectWaypointPolicy


GLOBAL_KEYS = {
    "task_field",
    "global_task_field",
    "all_uav_states",
    "task_id",
    "mission_goal",
    "goal_mask",
    "global_info",
    "comm_adjacency",
    "agent_mask",
}

AGENT_KEYS = {
    "self_state",
    "neighbor_relative_states",
    "task_field",
    "task_id",
    "mission_goal",
    "goal_mask",
    "global_summary",
    "all_uav_states",
    "comm_adjacency",
}


@pytest.fixture
def baseline_env():
    with open("configs/env/waypoint_missions.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.update(
        {
            "num_agents": 3,
            "grid_size": 8,
            "max_steps": 2,
            "task_name": "area_coverage",
            "task_names": ["area_coverage"],
            "candidate_count": 4,
        }
    )
    config["domain_randomization"]["enabled"] = False
    env = WaypointMultiUAVEnv(config)
    observations, _ = env.reset(seed=17)
    yield env, observations
    env.close()


def _batched_global_state(env: WaypointMultiUAVEnv, batch_size: int = 2) -> dict[str, torch.Tensor]:
    state = env.get_global_state()
    return {
        key: torch.as_tensor(np.repeat(value[None, ...], batch_size, axis=0), dtype=torch.float32)
        for key, value in state.items()
    }


def _small_policy_config(policy_class: str) -> dict:
    return {
        "policy_class": policy_class,
        "cnn_channels": [4, 8],
        "agent_hidden_dim": 16,
        "joint_hidden_dim": 24,
        "attention_heads": 2,
        "use_attention": True,
        "map_size": 1.0,
        "grid_size": 8,
        "max_delta": 0.2,
        "max_waypoint_distance": 0.2,
        "candidate_hidden_dim": 16,
        "candidate_count": 4,
    }


def test_default_observation_contract_has_existing_keys_and_shapes(baseline_env):
    env, observations = baseline_env
    global_state = env.get_global_state()

    assert set(global_state) == GLOBAL_KEYS
    assert global_state["task_field"].shape == (5, 8, 8)
    assert global_state["global_task_field"].shape == (5, 8, 8)
    assert global_state["all_uav_states"].shape == (3, 8)
    assert global_state["task_id"].shape == (6,)
    assert global_state["mission_goal"].shape == (3,)
    assert global_state["goal_mask"].shape == (3,)
    assert global_state["global_info"].shape == (8,)
    assert global_state["comm_adjacency"].shape == (4, 4)
    assert global_state["agent_mask"].shape == (3,)
    assert not any("heatmap" in key for key in global_state)

    agent_obs = observations["uav_0"]
    assert set(agent_obs) == AGENT_KEYS
    assert agent_obs["self_state"].shape == (8,)
    assert agent_obs["neighbor_relative_states"].shape == (2, 4)
    assert agent_obs["task_field"].shape == (5, 8, 8)
    assert agent_obs["task_id"].shape == (6,)
    assert agent_obs["mission_goal"].shape == (3,)
    assert agent_obs["goal_mask"].shape == (3,)
    assert agent_obs["global_summary"].shape == (8,)
    assert agent_obs["all_uav_states"].shape == (3, 8)
    assert agent_obs["comm_adjacency"].shape == (4, 4)
    assert not any("heatmap" in key for key in agent_obs)


def test_baseline_policy_classes_remain_selectable_with_current_output_contracts(baseline_env):
    env, _ = baseline_env
    torch.manual_seed(23)
    observations = _batched_global_state(env)

    direct = build_policy(_small_policy_config("direct_waypoint"), env.global_observation_space, env.action_space_n)
    assert isinstance(direct, DirectWaypointPolicy)
    direct_output = direct.get_action_and_value(observations)
    assert direct_output.env_action.shape == (2, 3, 2)
    assert direct_output.train_action.shape == (2, 3, 2)
    assert direct_output.logprob.shape == (2, 3)
    assert direct_output.entropy.shape == (2, 3)
    assert direct_output.value.shape == (2,)

    candidate = build_policy(
        _small_policy_config("candidate_selection_waypoint"),
        env.global_observation_space,
        env.action_space_n,
    )
    assert isinstance(candidate, CandidateSelectionWaypointPolicy)
    candidate_output = candidate.get_action_and_value(observations)
    assert candidate_output.train_action.shape == (2, 3)
    assert candidate_output.env_action.shape == (2, 3, 2)
    assert candidate_output.logprob.shape == (2, 3)
    assert candidate_output.entropy.shape == (2, 3)
    assert candidate_output.value.shape == (2,)


@pytest.mark.parametrize("train_action_shape", [(2, 3, 2), (2, 3)])
def test_rollout_buffer_preserves_existing_tensor_shapes(train_action_shape):
    horizon, env_count, agent_count = 3, 2, 3
    buffer = WaypointRolloutBuffer()
    observation = {
        "task_field": np.zeros((env_count, 5, 8, 8), dtype=np.float32),
        "all_uav_states": np.zeros((env_count, agent_count, 8), dtype=np.float32),
    }

    for _ in range(horizon):
        buffer.add(
            observation,
            env_action=np.zeros((env_count, agent_count, 2), dtype=np.float32),
            train_action=np.zeros(train_action_shape, dtype=np.float32),
            logprob=np.zeros((env_count, agent_count), dtype=np.float32),
            value=np.zeros(env_count, dtype=np.float32),
            team_reward=np.zeros(env_count, dtype=np.float32),
            per_agent_reward=np.zeros((env_count, agent_count), dtype=np.float32),
            terminated=np.zeros(env_count, dtype=bool),
            truncated=np.zeros(env_count, dtype=bool),
            bootstrap_mask=np.ones(env_count, dtype=np.float32),
            done_for_reset=np.zeros(env_count, dtype=bool),
            infos=[{} for _ in range(env_count)],
        )

    batch = buffer.build(
        advantages=np.zeros((horizon, env_count), dtype=np.float32),
        returns=np.zeros((horizon, env_count), dtype=np.float32),
    )

    assert batch.observations["task_field"].shape == (3, 2, 5, 8, 8)
    assert batch.observations["all_uav_states"].shape == (3, 2, 3, 8)
    assert batch.env_actions.shape == (3, 2, 3, 2)
    assert batch.train_actions.shape == (3, *train_action_shape)
    assert batch.old_logprobs.shape == (3, 2, 3)
    assert batch.values.shape == (3, 2)
    assert batch.team_rewards.shape == (3, 2)
    assert batch.per_agent_rewards.shape == (3, 2, 3)
    assert batch.advantages.shape == (3, 2)
    assert batch.returns.shape == (3, 2)

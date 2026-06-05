import numpy as np

from policies import build_policy, observation_to_tensor
from scripts._common import load_env_config, load_generic_config
from envs import CentralizedMultiUAVEnv


def test_variable_agent_policies_output_joint_waypoints():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 10, "task_name": "goal_nav"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=5)

    for policy_path in [
        "configs/policy/ppo_cnn_deepsets.yaml",
        "configs/policy/ppo_attention.yaml",
    ]:
        policy_config = load_generic_config(policy_path)
        policy = build_policy(policy_config, env.observation_space, env.action_space)
        obs_tensor = observation_to_tensor(observation, device="cpu")
        action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
        assert action.shape == (1, config["num_agents"], 2)
        assert np.all(np.isfinite(action))


def test_mlp_policy_stays_fixed_n():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=7)
    policy_config = load_generic_config("configs/policy/ppo_mlp.yaml")
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)


def test_cnn_deepsets_policy_supports_optional_coordination_repulsion():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=9)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["coordination_repulsion_strength"] = 0.25
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_optional_spatial_action_head():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=11)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_spatial_action_head"] = True
    policy_config["spatial_action_strength"] = 0.5
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_spatial_target_suppression():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=13)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_spatial_action_head"] = True
    policy_config["spatial_action_strength"] = 0.5
    policy_config["spatial_target_suppression_strength"] = 1.0
    policy_config["coordination_repulsion_strength"] = 0.25
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_angular_slot_embeddings():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=15)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_spatial_action_head"] = True
    policy_config["spatial_action_strength"] = 0.5
    policy_config["coordination_repulsion_strength"] = 0.25
    policy_config["use_angular_slot_embeddings"] = True
    policy_config["slot_embedding_strength"] = 1.0
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_sector_target_bias():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=17)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_spatial_action_head"] = True
    policy_config["spatial_action_strength"] = 0.5
    policy_config["coordination_repulsion_strength"] = 0.25
    policy_config["use_angular_slot_embeddings"] = True
    policy_config["slot_embedding_strength"] = 1.0
    policy_config["sector_target_bias_strength"] = 0.35
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_global_slot_head():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=19)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_global_slot_head"] = True
    policy_config["global_slot_strength"] = 0.5
    policy_config["use_angular_slot_embeddings"] = True
    policy_config["slot_embedding_strength"] = 1.0
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_global_slot_head_is_compatible_across_tasks():
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_global_slot_head"] = True
    policy_config["global_slot_strength"] = 0.4
    policy_config["use_angular_slot_embeddings"] = True
    policy_config["slot_embedding_strength"] = 1.0
    for task_name in ["goal_nav", "coverage", "formation", "risk_nav"]:
        config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": task_name})
        env = CentralizedMultiUAVEnv(config)
        observation, _ = env.reset(seed=23)
        policy = build_policy(policy_config, env.observation_space, env.action_space)
        obs_tensor = observation_to_tensor(observation, device="cpu")
        action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
        assert action.shape == (1, 4, 2)
        assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_slot_dominant_actor():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=29)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_global_slot_head"] = True
    policy_config["global_slot_strength"] = 0.8
    policy_config["use_angular_slot_embeddings"] = True
    policy_config["slot_embedding_strength"] = 1.0
    policy_config["actor_mean_residual_weight"] = 0.0
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_global_spatial_slot_head():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=31)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_global_spatial_slot_head"] = True
    policy_config["global_spatial_slot_strength"] = 0.5
    policy_config["use_angular_slot_embeddings"] = True
    policy_config["slot_embedding_strength"] = 1.0
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_coverage_utility_slot_head():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=37)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_coverage_utility_slot_head"] = True
    policy_config["coverage_utility_slot_strength"] = 0.8
    policy_config["coverage_utility_pool_size"] = 16
    policy_config["actor_mean_residual_weight"] = 0.2
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_coverage_utility_slot_head_keeps_task_compatibility():
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_coverage_utility_slot_head"] = True
    policy_config["coverage_utility_slot_strength"] = 0.7
    policy_config["coverage_utility_pool_size"] = 16
    policy_config["actor_mean_residual_weight"] = 0.2
    for task_name in ["goal_nav", "coverage", "formation", "risk_nav"]:
        config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": task_name})
        env = CentralizedMultiUAVEnv(config)
        observation, _ = env.reset(seed=41)
        policy = build_policy(policy_config, env.observation_space, env.action_space)
        obs_tensor = observation_to_tensor(observation, device="cpu")
        action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
        assert action.shape == (1, 4, 2)
        assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_coverage_frontier_slot_head():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=42)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_coverage_frontier_slot_head"] = True
    policy_config["coverage_frontier_slot_strength"] = 0.6
    policy_config["coverage_frontier_pool_size"] = 16
    policy_config["actor_mean_residual_weight"] = 0.2
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_factorized_group_policy_supports_coverage_frontier_slot_head():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=44)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["policy_class"] = "factorized_group"
    policy_config["num_groups"] = 2
    policy_config["group_hidden_dim"] = 96
    policy_config["group_action_strength"] = 0.4
    policy_config["use_coverage_frontier_slot_head"] = True
    policy_config["coverage_frontier_slot_strength"] = 0.5
    policy_config["coverage_frontier_pool_size"] = 16
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    assert policy.use_coverage_frontier_slot_head
    assert policy.coverage_frontier_slot_strength == 0.5
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action, logprob, entropy, value = policy.get_action_and_value(obs_tensor)
    assert action.shape == (1, 4, 2)
    assert logprob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1,)
    assert np.all(np.isfinite(action.detach().cpu().numpy()))


def test_factorized_group_policy_supports_sequential_group_context():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 5, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=45)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["policy_class"] = "factorized_group"
    policy_config["num_groups"] = 3
    policy_config["group_hidden_dim"] = 96
    policy_config["group_action_strength"] = 0.5
    policy_config["use_group_spatial_slots"] = True
    policy_config["use_sequential_group_context"] = True
    policy_config["sequential_group_context_strength"] = 0.6
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    assert policy.use_sequential_group_context
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action, logprob, entropy, value = policy.get_action_and_value(obs_tensor)
    assert action.shape == (1, 5, 2)
    assert logprob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1,)
    assert np.all(np.isfinite(action.detach().cpu().numpy()))


def test_factorized_group_policy_supports_coverage_lawnmower_route_head():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=46)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["policy_class"] = "factorized_group"
    policy_config["num_groups"] = 2
    policy_config["group_hidden_dim"] = 96
    policy_config["group_action_strength"] = 0.4
    policy_config["use_coverage_lawnmower_route_head"] = True
    policy_config["coverage_lawnmower_route_strength"] = 0.5
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    assert policy.use_coverage_lawnmower_route_head
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action, logprob, entropy, value = policy.get_action_and_value(obs_tensor)
    assert action.shape == (1, 4, 2)
    assert logprob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1,)
    assert np.all(np.isfinite(action.detach().cpu().numpy()))


def test_factorized_group_policy_supports_coverage_route_hint_head():
    config = load_env_config(
        "configs/env/multitask.yaml",
        override={
            "num_agents": 4,
            "task_name": "coverage",
            "coverage": {"route_hint_enabled": True},
        },
    )
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=48)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["policy_class"] = "factorized_group"
    policy_config["num_groups"] = 2
    policy_config["group_hidden_dim"] = 96
    policy_config["group_action_strength"] = 0.4
    policy_config["use_coverage_route_hint_head"] = True
    policy_config["coverage_route_hint_strength"] = 0.5
    policy_config["coverage_route_hint_pool_size"] = 16
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    assert policy.use_coverage_route_hint_head
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action, logprob, entropy, value = policy.get_action_and_value(obs_tensor)
    assert action.shape == (1, 4, 2)
    assert logprob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1,)
    assert np.all(np.isfinite(action.detach().cpu().numpy()))


def test_factorized_group_policy_supports_route_target_agent_features():
    config = load_env_config(
        "configs/env/multitask.yaml",
        override={
            "num_agents": 4,
            "task_name": "coverage",
            "include_route_targets_in_agents": True,
            "coverage": {"route_hint_enabled": True},
        },
    )
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=49)
    assert observation["agents"].shape[-1] == 10
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["policy_class"] = "factorized_group"
    policy_config["num_groups"] = 2
    policy_config["group_hidden_dim"] = 96
    policy_config["group_action_strength"] = 0.4
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action, logprob, entropy, value = policy.get_action_and_value(obs_tensor)
    assert action.shape == (1, 4, 2)
    assert logprob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1,)
    assert np.all(np.isfinite(action.detach().cpu().numpy()))


def test_factorized_group_policy_outputs_joint_waypoints():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 6, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=43)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["policy_class"] = "factorized_group"
    policy_config["num_groups"] = 3
    policy_config["group_hidden_dim"] = 96
    policy_config["group_action_strength"] = 0.6
    policy_config["use_group_spatial_slots"] = True
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action, logprob, entropy, value = policy.get_action_and_value(obs_tensor)
    assert action.shape == (1, 6, 2)
    assert logprob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1,)
    assert np.all(np.isfinite(action.detach().cpu().numpy()))


def test_factorized_group_policy_keeps_task_compatibility():
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["policy_class"] = "factorized_group"
    policy_config["num_groups"] = 2
    policy_config["group_hidden_dim"] = 96
    policy_config["group_action_strength"] = 0.5
    for task_name in ["goal_nav", "coverage", "formation", "risk_nav"]:
        config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": task_name})
        env = CentralizedMultiUAVEnv(config)
        observation, _ = env.reset(seed=47)
        policy = build_policy(policy_config, env.observation_space, env.action_space)
        obs_tensor = observation_to_tensor(observation, device="cpu")
        action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
        assert action.shape == (1, 4, 2)
        assert np.all(np.isfinite(action))


def test_mappo_shared_policy_outputs_variable_n_waypoints():
    policy_config = load_generic_config("configs/policy/mappo_shared.yaml")
    for num_agents in [4, 10]:
        config = load_env_config("configs/env/multitask.yaml", override={"num_agents": num_agents, "task_name": "goal_nav"})
        env = CentralizedMultiUAVEnv(config)
        observation, _ = env.reset(seed=51)
        policy = build_policy(policy_config, env.observation_space, env.action_space)
        obs_tensor = observation_to_tensor(observation, device="cpu")
        action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
        assert action.shape == (1, num_agents, 2)
        assert np.all(np.isfinite(action))


def test_mappo_shared_policy_returns_per_agent_logprob_entropy():
    policy_config = load_generic_config("configs/policy/mappo_shared.yaml")
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 6, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=53)
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action, logprob, entropy, value = policy.get_action_and_value(obs_tensor)
    assert action.shape == (1, 6, 2)
    assert logprob.shape == (1, 6)
    assert entropy.shape == (1, 6)
    assert value.shape == (1,)
    assert np.all(np.isfinite(action.detach().cpu().numpy()))
    assert np.all(np.isfinite(logprob.detach().cpu().numpy()))
    assert np.all(np.isfinite(entropy.detach().cpu().numpy()))

from __future__ import annotations

import numpy as np
import torch

from policies.attention_policy import CNNAttentionPolicy
from policies.action_distribution import SquashedNormal
from policies.cnn_deepsets_policy import CNNDeepSetsPolicy
from policies.factorized_group_policy import FactorizedGroupPolicy
from policies.mappo_shared_policy import MAPPOSharedPolicy
from policies.mlp_policy import MLPPolicy


def observation_to_tensor(observation: dict, device: torch.device, already_batched: bool = False) -> dict[str, torch.Tensor]:
    tensors = {}
    for key, value in observation.items():
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if not already_batched and tensor.ndim == len(np.asarray(value).shape):
            tensor = tensor.unsqueeze(0)
        tensors[key] = tensor
    return tensors


def build_policy(policy_config: dict, observation_space, action_space):
    policy_class = policy_config.get("policy_class", "mlp")
    if policy_class == "mlp":
        return MLPPolicy(observation_space, action_space, hidden_dims=policy_config.get("hidden_dims"))
    if policy_class == "cnn_deepsets":
        return CNNDeepSetsPolicy(
            observation_space,
            action_space,
            cnn_channels=policy_config.get("cnn_channels"),
            agent_hidden_dim=int(policy_config.get("agent_hidden_dim", 64)),
            joint_hidden_dim=int(policy_config.get("joint_hidden_dim", 256)),
            decoder_hidden_dim=int(policy_config.get("decoder_hidden_dim", 128)),
            use_spatial_attention=bool(policy_config.get("use_spatial_attention", False)),
            use_spatial_action_head=bool(policy_config.get("use_spatial_action_head", False)),
            spatial_pool_size=int(policy_config.get("spatial_pool_size", 8)),
            attention_heads=int(policy_config.get("attention_heads", 4)),
            coordination_repulsion_strength=float(policy_config.get("coordination_repulsion_strength", 0.0)),
            spatial_action_strength=float(policy_config.get("spatial_action_strength", 0.0)),
            spatial_target_suppression_strength=float(policy_config.get("spatial_target_suppression_strength", 0.0)),
            spatial_target_suppression_sigma=float(policy_config.get("spatial_target_suppression_sigma", 0.15)),
            use_angular_slot_embeddings=bool(policy_config.get("use_angular_slot_embeddings", False)),
            slot_embedding_strength=float(policy_config.get("slot_embedding_strength", 1.0)),
            sector_target_bias_strength=float(policy_config.get("sector_target_bias_strength", 0.0)),
            use_global_slot_head=bool(policy_config.get("use_global_slot_head", False)),
            global_slot_strength=float(policy_config.get("global_slot_strength", 0.0)),
            use_global_spatial_slot_head=bool(policy_config.get("use_global_spatial_slot_head", False)),
            global_spatial_slot_strength=float(policy_config.get("global_spatial_slot_strength", 0.0)),
            use_coverage_utility_slot_head=bool(policy_config.get("use_coverage_utility_slot_head", False)),
            coverage_utility_slot_strength=float(policy_config.get("coverage_utility_slot_strength", 0.0)),
            coverage_utility_temperature=float(policy_config.get("coverage_utility_temperature", 8.0)),
            coverage_utility_pool_size=int(policy_config.get("coverage_utility_pool_size", 16)),
            coverage_utility_sector_bias_strength=float(policy_config.get("coverage_utility_sector_bias_strength", 1.2)),
            coverage_utility_suppression_strength=float(policy_config.get("coverage_utility_suppression_strength", 3.0)),
            coverage_utility_suppression_sigma=float(policy_config.get("coverage_utility_suppression_sigma", 0.25)),
            coverage_utility_target_weight=float(policy_config.get("coverage_utility_target_weight", 0.6)),
            coverage_utility_desired_weight=float(policy_config.get("coverage_utility_desired_weight", 1.8)),
            coverage_utility_visited_weight=float(policy_config.get("coverage_utility_visited_weight", -0.4)),
            coverage_utility_risk_weight=float(policy_config.get("coverage_utility_risk_weight", -0.2)),
            coverage_utility_obstacle_weight=float(policy_config.get("coverage_utility_obstacle_weight", -0.8)),
            coverage_utility_agent_density_weight=float(policy_config.get("coverage_utility_agent_density_weight", -0.3)),
            use_coverage_frontier_slot_head=bool(policy_config.get("use_coverage_frontier_slot_head", False)),
            coverage_frontier_slot_strength=float(policy_config.get("coverage_frontier_slot_strength", 0.0)),
            coverage_frontier_temperature=float(policy_config.get("coverage_frontier_temperature", 8.0)),
            coverage_frontier_pool_size=int(policy_config.get("coverage_frontier_pool_size", 16)),
            coverage_frontier_sector_bias_strength=float(policy_config.get("coverage_frontier_sector_bias_strength", 1.4)),
            coverage_frontier_suppression_strength=float(policy_config.get("coverage_frontier_suppression_strength", 3.5)),
            coverage_frontier_suppression_sigma=float(policy_config.get("coverage_frontier_suppression_sigma", 0.22)),
            coverage_frontier_distance_weight=float(policy_config.get("coverage_frontier_distance_weight", 0.6)),
            coverage_frontier_target_weight=float(policy_config.get("coverage_frontier_target_weight", 0.8)),
            coverage_frontier_desired_weight=float(policy_config.get("coverage_frontier_desired_weight", 2.4)),
            coverage_frontier_visited_power=float(policy_config.get("coverage_frontier_visited_power", 1.0)),
            coverage_frontier_obstacle_weight=float(policy_config.get("coverage_frontier_obstacle_weight", -1.2)),
            coverage_frontier_agent_density_weight=float(policy_config.get("coverage_frontier_agent_density_weight", -0.5)),
            use_coverage_lawnmower_route_head=bool(policy_config.get("use_coverage_lawnmower_route_head", False)),
            coverage_lawnmower_route_strength=float(policy_config.get("coverage_lawnmower_route_strength", 0.0)),
            coverage_lawnmower_target_weight=float(policy_config.get("coverage_lawnmower_target_weight", 1.0)),
            coverage_lawnmower_desired_weight=float(policy_config.get("coverage_lawnmower_desired_weight", 3.0)),
            coverage_lawnmower_visited_power=float(policy_config.get("coverage_lawnmower_visited_power", 1.2)),
            coverage_lawnmower_obstacle_weight=float(policy_config.get("coverage_lawnmower_obstacle_weight", -2.0)),
            coverage_lawnmower_distance_weight=float(policy_config.get("coverage_lawnmower_distance_weight", 0.25)),
            coverage_lawnmower_stripe_bonus=float(policy_config.get("coverage_lawnmower_stripe_bonus", 2.0)),
            coverage_lawnmower_ahead_bonus=float(policy_config.get("coverage_lawnmower_ahead_bonus", 0.8)),
            use_coverage_route_hint_head=bool(policy_config.get("use_coverage_route_hint_head", False)),
            coverage_route_hint_strength=float(policy_config.get("coverage_route_hint_strength", 0.0)),
            coverage_route_hint_temperature=float(policy_config.get("coverage_route_hint_temperature", 10.0)),
            coverage_route_hint_distance_weight=float(policy_config.get("coverage_route_hint_distance_weight", 0.6)),
            coverage_route_hint_suppression_strength=float(policy_config.get("coverage_route_hint_suppression_strength", 6.0)),
            coverage_route_hint_suppression_sigma=float(policy_config.get("coverage_route_hint_suppression_sigma", 0.08)),
            coverage_route_hint_pool_size=int(policy_config.get("coverage_route_hint_pool_size", 16)),
            actor_mean_residual_weight=float(policy_config.get("actor_mean_residual_weight", 1.0)),
            log_std_min=float(policy_config.get("log_std_min", -1.5)),
            log_std_max=float(policy_config.get("log_std_max", 0.5)),
            log_std_init=float(policy_config.get("log_std_init", 0.0)),
        )
    if policy_class == "factorized_group":
        return FactorizedGroupPolicy(
            observation_space,
            action_space,
            cnn_channels=policy_config.get("cnn_channels"),
            agent_hidden_dim=int(policy_config.get("agent_hidden_dim", 64)),
            joint_hidden_dim=int(policy_config.get("joint_hidden_dim", 256)),
            decoder_hidden_dim=int(policy_config.get("decoder_hidden_dim", 128)),
            use_spatial_attention=bool(policy_config.get("use_spatial_attention", False)),
            use_spatial_action_head=bool(policy_config.get("use_spatial_action_head", False)),
            spatial_pool_size=int(policy_config.get("spatial_pool_size", 8)),
            attention_heads=int(policy_config.get("attention_heads", 4)),
            coordination_repulsion_strength=float(policy_config.get("coordination_repulsion_strength", 0.0)),
            spatial_action_strength=float(policy_config.get("spatial_action_strength", 0.0)),
            spatial_target_suppression_strength=float(policy_config.get("spatial_target_suppression_strength", 0.0)),
            spatial_target_suppression_sigma=float(policy_config.get("spatial_target_suppression_sigma", 0.15)),
            use_angular_slot_embeddings=bool(policy_config.get("use_angular_slot_embeddings", False)),
            slot_embedding_strength=float(policy_config.get("slot_embedding_strength", 1.0)),
            sector_target_bias_strength=float(policy_config.get("sector_target_bias_strength", 0.0)),
            use_global_slot_head=bool(policy_config.get("use_global_slot_head", False)),
            global_slot_strength=float(policy_config.get("global_slot_strength", 0.0)),
            use_global_spatial_slot_head=bool(policy_config.get("use_global_spatial_slot_head", False)),
            global_spatial_slot_strength=float(policy_config.get("global_spatial_slot_strength", 0.0)),
            use_coverage_utility_slot_head=bool(policy_config.get("use_coverage_utility_slot_head", False)),
            coverage_utility_slot_strength=float(policy_config.get("coverage_utility_slot_strength", 0.0)),
            coverage_utility_temperature=float(policy_config.get("coverage_utility_temperature", 8.0)),
            coverage_utility_pool_size=int(policy_config.get("coverage_utility_pool_size", 16)),
            coverage_utility_sector_bias_strength=float(policy_config.get("coverage_utility_sector_bias_strength", 1.2)),
            coverage_utility_suppression_strength=float(policy_config.get("coverage_utility_suppression_strength", 3.0)),
            coverage_utility_suppression_sigma=float(policy_config.get("coverage_utility_suppression_sigma", 0.25)),
            coverage_utility_target_weight=float(policy_config.get("coverage_utility_target_weight", 0.6)),
            coverage_utility_desired_weight=float(policy_config.get("coverage_utility_desired_weight", 1.8)),
            coverage_utility_visited_weight=float(policy_config.get("coverage_utility_visited_weight", -0.4)),
            coverage_utility_risk_weight=float(policy_config.get("coverage_utility_risk_weight", -0.2)),
            coverage_utility_obstacle_weight=float(policy_config.get("coverage_utility_obstacle_weight", -0.8)),
            coverage_utility_agent_density_weight=float(policy_config.get("coverage_utility_agent_density_weight", -0.3)),
            use_coverage_frontier_slot_head=bool(policy_config.get("use_coverage_frontier_slot_head", False)),
            coverage_frontier_slot_strength=float(policy_config.get("coverage_frontier_slot_strength", 0.0)),
            coverage_frontier_temperature=float(policy_config.get("coverage_frontier_temperature", 8.0)),
            coverage_frontier_pool_size=int(policy_config.get("coverage_frontier_pool_size", 16)),
            coverage_frontier_sector_bias_strength=float(policy_config.get("coverage_frontier_sector_bias_strength", 1.4)),
            coverage_frontier_suppression_strength=float(policy_config.get("coverage_frontier_suppression_strength", 3.5)),
            coverage_frontier_suppression_sigma=float(policy_config.get("coverage_frontier_suppression_sigma", 0.22)),
            coverage_frontier_distance_weight=float(policy_config.get("coverage_frontier_distance_weight", 0.6)),
            coverage_frontier_target_weight=float(policy_config.get("coverage_frontier_target_weight", 0.8)),
            coverage_frontier_desired_weight=float(policy_config.get("coverage_frontier_desired_weight", 2.4)),
            coverage_frontier_visited_power=float(policy_config.get("coverage_frontier_visited_power", 1.0)),
            coverage_frontier_obstacle_weight=float(policy_config.get("coverage_frontier_obstacle_weight", -1.2)),
            coverage_frontier_agent_density_weight=float(policy_config.get("coverage_frontier_agent_density_weight", -0.5)),
            use_coverage_lawnmower_route_head=bool(policy_config.get("use_coverage_lawnmower_route_head", False)),
            coverage_lawnmower_route_strength=float(policy_config.get("coverage_lawnmower_route_strength", 0.0)),
            coverage_lawnmower_target_weight=float(policy_config.get("coverage_lawnmower_target_weight", 1.0)),
            coverage_lawnmower_desired_weight=float(policy_config.get("coverage_lawnmower_desired_weight", 3.0)),
            coverage_lawnmower_visited_power=float(policy_config.get("coverage_lawnmower_visited_power", 1.2)),
            coverage_lawnmower_obstacle_weight=float(policy_config.get("coverage_lawnmower_obstacle_weight", -2.0)),
            coverage_lawnmower_distance_weight=float(policy_config.get("coverage_lawnmower_distance_weight", 0.25)),
            coverage_lawnmower_stripe_bonus=float(policy_config.get("coverage_lawnmower_stripe_bonus", 2.0)),
            coverage_lawnmower_ahead_bonus=float(policy_config.get("coverage_lawnmower_ahead_bonus", 0.8)),
            use_coverage_route_hint_head=bool(policy_config.get("use_coverage_route_hint_head", False)),
            coverage_route_hint_strength=float(policy_config.get("coverage_route_hint_strength", 0.0)),
            coverage_route_hint_temperature=float(policy_config.get("coverage_route_hint_temperature", 10.0)),
            coverage_route_hint_distance_weight=float(policy_config.get("coverage_route_hint_distance_weight", 0.6)),
            coverage_route_hint_suppression_strength=float(policy_config.get("coverage_route_hint_suppression_strength", 6.0)),
            coverage_route_hint_suppression_sigma=float(policy_config.get("coverage_route_hint_suppression_sigma", 0.08)),
            coverage_route_hint_pool_size=int(policy_config.get("coverage_route_hint_pool_size", 16)),
            actor_mean_residual_weight=float(policy_config.get("actor_mean_residual_weight", 1.0)),
            log_std_min=float(policy_config.get("log_std_min", -1.5)),
            log_std_max=float(policy_config.get("log_std_max", 0.5)),
            log_std_init=float(policy_config.get("log_std_init", 0.0)),
            num_groups=policy_config.get("num_groups"),
            group_hidden_dim=int(policy_config.get("group_hidden_dim", 128)),
            group_assignment_temperature=float(policy_config.get("group_assignment_temperature", 1.0)),
            group_action_strength=float(policy_config.get("group_action_strength", 0.5)),
            use_group_spatial_slots=bool(policy_config.get("use_group_spatial_slots", True)),
            use_sequential_group_context=bool(policy_config.get("use_sequential_group_context", False)),
            sequential_group_context_strength=float(policy_config.get("sequential_group_context_strength", 0.5)),
        )
    if policy_class == "mappo_shared":
        return MAPPOSharedPolicy(
            observation_space,
            action_space,
            cnn_channels=policy_config.get("cnn_channels"),
            agent_hidden_dim=int(policy_config.get("agent_hidden_dim", 64)),
            joint_hidden_dim=int(policy_config.get("joint_hidden_dim", 256)),
            decoder_hidden_dim=int(policy_config.get("decoder_hidden_dim", 128)),
            log_std_min=float(policy_config.get("log_std_min", -1.5)),
            log_std_max=float(policy_config.get("log_std_max", 0.5)),
            log_std_init=float(policy_config.get("log_std_init", 0.0)),
        )
    if policy_class == "attention":
        return CNNAttentionPolicy(
            observation_space,
            action_space,
            embed_dim=int(policy_config.get("embed_dim", 128)),
            num_heads=int(policy_config.get("num_heads", 4)),
            num_layers=int(policy_config.get("num_layers", 2)),
            cnn_channels=policy_config.get("cnn_channels"),
        )
    raise ValueError(f"Unsupported policy class: {policy_class}")


__all__ = [
    "MLPPolicy",
    "CNNDeepSetsPolicy",
    "FactorizedGroupPolicy",
    "MAPPOSharedPolicy",
    "CNNAttentionPolicy",
    "SquashedNormal",
    "build_policy",
    "observation_to_tensor",
]

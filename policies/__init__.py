from __future__ import annotations

import torch

from policies.candidate_selection_policy import CandidateSelectionWaypointPolicy, MAPPOWaypointPolicy
from policies.direct_waypoint_policy import DirectWaypointPolicy, UVFADirectWaypointPolicy
from policies.policy_output import PolicyOutput


def observation_to_tensor(
    observation: dict,
    device: torch.device,
    already_batched: bool = False,
) -> dict[str, torch.Tensor]:
    tensors = {}
    for key, value in observation.items():
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if not already_batched:
            tensor = tensor.unsqueeze(0)
        tensors[key] = tensor
    return tensors


def build_policy(policy_config: dict, observation_space, action_space):
    policy_class = str(policy_config.get("policy_class", "candidate_selection_waypoint"))
    common = {
        "cnn_channels": policy_config.get("cnn_channels"),
        "agent_hidden_dim": int(policy_config.get("agent_hidden_dim", 64)),
        "joint_hidden_dim": int(policy_config.get("joint_hidden_dim", 192)),
        "attention_heads": int(policy_config.get("attention_heads", 4)),
        "use_attention": bool(policy_config.get("use_attention", True)),
        "map_size": float(policy_config.get("map_size", 1.0)),
    }
    if policy_class in {"candidate_selection_waypoint", "mappo_waypoint"}:
        return CandidateSelectionWaypointPolicy(
            observation_space,
            action_space,
            **common,
            candidate_hidden_dim=int(policy_config.get("candidate_hidden_dim", 64)),
            candidate_count=int(policy_config.get("candidate_count", 12)),
            grid_size=int(policy_config.get("grid_size", 32)),
            max_waypoint_distance=float(policy_config.get("max_waypoint_distance", 0.22)),
            min_uav_distance=float(policy_config.get("min_uav_distance", 0.035)),
            comm_radius=float(policy_config.get("comm_radius", 0.36)),
            base_comm_radius=float(policy_config.get("base_comm_radius", 0.45)),
            no_fly_zones=list(policy_config.get("no_fly_zones", [])),
            base_position=list(policy_config.get("base_position", [0.1, 0.1])),
            candidate_generator=str(policy_config.get("candidate_generator", "task_aware")),
            connectivity_candidate_filter=bool(policy_config.get("connectivity_candidate_filter", True)),
            candidate_prior_coef=float(policy_config.get("candidate_prior_coef", 0.0)),
            connectivity_chain_candidates=bool(policy_config.get("connectivity_chain_candidates", False)),
        )
    if policy_class in {"direct_waypoint", "direct_waypoint_uvfa"}:
        policy_type = UVFADirectWaypointPolicy if policy_class == "direct_waypoint_uvfa" else DirectWaypointPolicy
        return policy_type(
            observation_space,
            action_space,
            **common,
            max_delta=float(policy_config.get("max_delta", policy_config.get("max_waypoint_distance", 0.22))),
            log_std_init=float(policy_config.get("log_std_init", -1.0)),
            log_std_min=float(policy_config.get("log_std_min", -2.0)),
            log_std_max=float(policy_config.get("log_std_max", 0.5)),
            task_field_channel_mode=str(policy_config.get("task_field_channel_mode", "all")),
            task_field_channel_masks=policy_config.get("task_field_channel_masks"),
            use_spatial_field_context=bool(policy_config.get("use_spatial_field_context", False)),
            spatial_context_radii=list(policy_config.get("spatial_context_radii", [0.06, 0.12])),
            use_field_moment_context=bool(policy_config.get("use_field_moment_context", False)),
            field_moment_include_inverse=bool(policy_config.get("field_moment_include_inverse", False)),
            use_connectivity_auxiliary_loss=bool(policy_config.get("use_connectivity_auxiliary_loss", False)),
            use_comm_graph_encoder=bool(policy_config.get("use_comm_graph_encoder", False)),
            goal_embedding_dim=int(policy_config.get("goal_embedding_dim", 64)),
            goal_hidden_dim=int(policy_config.get("goal_hidden_dim", 64)),
            use_task_element_heatmap=bool(policy_config.get("use_task_element_heatmap", False)),
        )
    raise ValueError(f"Unsupported waypoint MARL policy class: {policy_class}")


__all__ = [
    "CandidateSelectionWaypointPolicy",
    "DirectWaypointPolicy",
    "UVFADirectWaypointPolicy",
    "MAPPOWaypointPolicy",
    "PolicyOutput",
    "build_policy",
    "observation_to_tensor",
]

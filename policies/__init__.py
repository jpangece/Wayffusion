from __future__ import annotations

import torch

from policies.mappo_waypoint_policy import MAPPOWaypointPolicy


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
    policy_class = policy_config.get("policy_class", "mappo_waypoint")
    if policy_class != "mappo_waypoint":
        raise ValueError(f"Unsupported waypoint MARL policy class: {policy_class}")
    return MAPPOWaypointPolicy(
        observation_space,
        action_space,
        cnn_channels=policy_config.get("cnn_channels"),
        agent_hidden_dim=int(policy_config.get("agent_hidden_dim", 64)),
        candidate_hidden_dim=int(policy_config.get("candidate_hidden_dim", 64)),
        joint_hidden_dim=int(policy_config.get("joint_hidden_dim", 192)),
        attention_heads=int(policy_config.get("attention_heads", 4)),
        use_attention=bool(policy_config.get("use_attention", True)),
    )


__all__ = ["MAPPOWaypointPolicy", "build_policy", "observation_to_tensor"]

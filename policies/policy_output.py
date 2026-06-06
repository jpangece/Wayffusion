from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PolicyOutput:
    """Common policy output for waypoint-level MAPPO.

    env_action is the final waypoint sent to the environment. train_action is
    the policy-internal sampled action used for PPO log-prob recomputation.
    """

    env_action: torch.Tensor
    train_action: torch.Tensor | None
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    aux: dict[str, Any] = field(default_factory=dict)


__all__ = ["PolicyOutput"]

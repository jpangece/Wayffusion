from __future__ import annotations

import torch
from torch import nn

from policies.action_distribution import SquashedNormal
from policies.cnn_deepsets_policy import masked_mean


class MAPPOSharedPolicy(nn.Module):
    """Shared per-agent actor with a centralized critic for MAPPO-style PPO.

    The policy keeps the existing centralized environment API: observations are
    batched dictionaries and actions are joint waypoint tensors [B, N, 2]. The
    difference from the joint-action PPO policies is that log-probabilities and
    entropies are returned per agent [B, N], so the trainer can form agent-wise
    PPO ratios instead of one high-dimensional joint ratio per environment.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        cnn_channels: list[int] | None = None,
        agent_hidden_dim: int = 64,
        joint_hidden_dim: int = 256,
        decoder_hidden_dim: int = 128,
        log_std_min: float = -1.5,
        log_std_max: float = 0.5,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        del action_space
        cnn_channels = cnn_channels or [16, 32, 64]
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        in_channels = int(observation_space["task_field"].shape[0])
        conv_layers = []
        last_channels = in_channels
        for out_channels in cnn_channels:
            conv_layers += [
                nn.Conv2d(last_channels, int(out_channels), kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ]
            last_channels = int(out_channels)
        self.field_encoder = nn.Sequential(*conv_layers)
        self.field_global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        agent_input_dim = int(observation_space["agents"].shape[-1])
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_input_dim, agent_hidden_dim),
            nn.ReLU(),
            nn.Linear(agent_hidden_dim, agent_hidden_dim),
            nn.ReLU(),
        )

        task_dim = int(observation_space["task_id"].shape[0])
        global_dim = int(observation_space["global_info"].shape[0])
        actor_input_dim = agent_hidden_dim + last_channels + task_dim + global_dim
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, 2),
        )

        critic_input_dim = last_channels + agent_hidden_dim + task_dim + global_dim
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, 1),
        )

        init_log_std = float(min(max(log_std_init, self.log_std_min), self.log_std_max))
        self.log_std = nn.Parameter(torch.full((2,), init_log_std))

    def _agent_mask(self, obs: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if "agent_mask" not in obs:
            return None
        return obs["agent_mask"].bool()

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action means [B, N, 2] and centralized state values [B]."""

        field_map = self.field_encoder(obs["task_field"])
        field_feat = self.field_global_pool(field_map)
        agent_tokens = self.agent_encoder(obs["agents"])
        agent_mask = self._agent_mask(obs)

        batch_size, num_agents, _ = agent_tokens.shape
        field_per_agent = field_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        task_per_agent = obs["task_id"].unsqueeze(1).expand(batch_size, num_agents, -1)
        global_per_agent = obs["global_info"].unsqueeze(1).expand(batch_size, num_agents, -1)
        actor_input = torch.cat([agent_tokens, field_per_agent, task_per_agent, global_per_agent], dim=-1)
        mean = self.actor(actor_input)

        pooled_agents = masked_mean(agent_tokens, agent_mask, dim=1)
        critic_input = torch.cat([field_feat, pooled_agents, obs["task_id"], obs["global_info"]], dim=-1)
        value = self.critic(critic_input).squeeze(-1)

        if agent_mask is not None:
            mean = mean * agent_mask.unsqueeze(-1).float()
        return mean, value

    def _distribution(self, mean: torch.Tensor) -> SquashedNormal:
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).view(1, 1, -1).expand_as(mean)
        return SquashedNormal(mean, std)

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample/evaluate actions and return per-agent logprob and entropy.

        log_prob and entropy sum over the 2-D waypoint action only. They keep the
        agent dimension so MAPPO can optimize one PPO ratio per UAV.
        """

        mean, value = self.forward(obs)
        dist = self._distribution(mean)
        if action is None:
            action, raw_action = dist.rsample(return_raw=True)
        else:
            action = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
            raw_action = dist.atanh(action)

        log_prob = dist.log_prob(action, raw_action=raw_action, reduce=False).sum(dim=-1)
        entropy = dist.base_entropy(reduce=False).sum(dim=-1)

        agent_mask = self._agent_mask(obs)
        if agent_mask is not None:
            mask = agent_mask.float()
            action = action * mask.unsqueeze(-1)
            log_prob = log_prob * mask
            entropy = entropy * mask
        return action, log_prob, entropy, value

    def act_deterministic(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        mean, _ = self.forward(obs)
        action = torch.tanh(mean)
        agent_mask = self._agent_mask(obs)
        if agent_mask is not None:
            action = action * agent_mask.unsqueeze(-1).float()
        return action

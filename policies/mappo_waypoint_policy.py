from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=dim)
    weights = mask.float().unsqueeze(-1)
    return (values * weights).sum(dim=dim) / torch.clamp(weights.sum(dim=dim), min=1.0)


class MAPPOWaypointPolicy(nn.Module):
    """Shared categorical waypoint actor with centralized critic."""

    def __init__(
        self,
        observation_space,
        action_space=None,
        cnn_channels: list[int] | None = None,
        agent_hidden_dim: int = 64,
        candidate_hidden_dim: int = 64,
        joint_hidden_dim: int = 192,
        attention_heads: int = 4,
        use_attention: bool = True,
    ):
        super().__init__()
        del action_space
        cnn_channels = cnn_channels or [16, 32, 64]
        task_field_space = observation_space["task_field"] if "task_field" in observation_space else observation_space["global_task_field"]
        in_channels = int(task_field_space.shape[0])
        conv_layers = []
        last_channels = in_channels
        for out_channels in cnn_channels:
            conv_layers.extend(
                [
                    nn.Conv2d(last_channels, int(out_channels), kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                ]
            )
            last_channels = int(out_channels)
        self.field_encoder = nn.Sequential(*conv_layers)
        self.field_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.field_dim = last_channels

        agent_dim = int(observation_space["all_uav_states"].shape[-1])
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_dim, agent_hidden_dim),
            nn.ReLU(),
            nn.Linear(agent_hidden_dim, agent_hidden_dim),
            nn.ReLU(),
        )
        self.use_attention = bool(use_attention)
        if self.use_attention:
            heads = max(1, min(int(attention_heads), int(agent_hidden_dim)))
            while agent_hidden_dim % heads != 0 and heads > 1:
                heads -= 1
            self.agent_attention = nn.MultiheadAttention(agent_hidden_dim, heads, batch_first=True)
            self.agent_attention_norm = nn.LayerNorm(agent_hidden_dim)

        candidate_dim = int(observation_space["candidate_waypoints"].shape[-1]) + int(observation_space["candidate_features"].shape[-1])
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, candidate_hidden_dim),
            nn.ReLU(),
            nn.Linear(candidate_hidden_dim, candidate_hidden_dim),
            nn.ReLU(),
        )
        task_dim = int(observation_space["task_id"].shape[0])
        global_dim = int(observation_space["global_info"].shape[0])
        actor_context_dim = agent_hidden_dim + self.field_dim + task_dim + global_dim
        self.actor_context = nn.Sequential(
            nn.Linear(actor_context_dim, candidate_hidden_dim),
            nn.ReLU(),
            nn.Linear(candidate_hidden_dim, candidate_hidden_dim),
            nn.ReLU(),
        )
        self.logit_head = nn.Linear(candidate_hidden_dim, 1)
        critic_dim = self.field_dim + agent_hidden_dim + task_dim + global_dim
        self.critic = nn.Sequential(
            nn.Linear(critic_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, 1),
        )

    def _task_field(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["task_field"] if "task_field" in obs else obs["global_task_field"]

    def _agent_mask(self, obs: dict[str, torch.Tensor], batch_size: int, num_agents: int) -> torch.Tensor:
        if "agent_mask" in obs:
            return obs["agent_mask"].bool()
        return torch.ones((batch_size, num_agents), dtype=torch.bool, device=self._task_field(obs).device)

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        field = self._task_field(obs)
        field_feat = self.field_pool(self.field_encoder(field))
        agents = obs["all_uav_states"]
        batch_size, num_agents, _ = agents.shape
        agent_mask = self._agent_mask(obs, batch_size, num_agents)
        agent_tokens = self.agent_encoder(agents)
        if self.use_attention:
            key_padding_mask = ~agent_mask
            attended, _ = self.agent_attention(agent_tokens, agent_tokens, agent_tokens, key_padding_mask=key_padding_mask)
            agent_tokens = self.agent_attention_norm(agent_tokens + attended)

        candidates = obs["candidate_waypoints"]
        candidate_features = obs["candidate_features"]
        candidate_input = torch.cat([candidates, candidate_features], dim=-1)
        candidate_tokens = self.candidate_encoder(candidate_input)
        _, _, candidate_count, _ = candidate_tokens.shape

        field_per_agent = field_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        task_per_agent = obs["task_id"].unsqueeze(1).expand(batch_size, num_agents, -1)
        global_per_agent = obs["global_info"].unsqueeze(1).expand(batch_size, num_agents, -1)
        actor_context = torch.cat([agent_tokens, field_per_agent, task_per_agent, global_per_agent], dim=-1)
        actor_context = self.actor_context(actor_context).unsqueeze(2).expand(-1, -1, candidate_count, -1)
        logits = self.logit_head(torch.tanh(actor_context + candidate_tokens)).squeeze(-1)

        candidate_mask = obs["candidate_mask"].bool()
        if candidate_mask.ndim != 3:
            raise ValueError(f"candidate_mask must be [B, N, K], got {tuple(candidate_mask.shape)}")
        safe_mask = candidate_mask.clone()
        empty = ~safe_mask.any(dim=-1)
        if empty.any():
            safe_mask[empty, 0] = True
        logits = logits.masked_fill(~safe_mask, -1.0e9)
        logits = logits.masked_fill(~agent_mask.unsqueeze(-1), -1.0e9)

        pooled_agents = _masked_mean(agent_tokens, agent_mask, dim=1)
        critic_input = torch.cat([field_feat, pooled_agents, obs["task_id"], obs["global_info"]], dim=-1)
        value = self.critic(critic_input).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        batch_size, num_agents, candidate_count = logits.shape
        dist = Categorical(logits=logits.reshape(batch_size * num_agents, candidate_count))
        if action is None:
            flat_action = dist.sample()
            action = flat_action.view(batch_size, num_agents)
        else:
            action = action.long()
            flat_action = action.reshape(-1)
        logprob = dist.log_prob(flat_action).view(batch_size, num_agents)
        entropy = dist.entropy().view(batch_size, num_agents)
        agent_mask = self._agent_mask(obs, batch_size, num_agents)
        logprob = logprob * agent_mask.float()
        entropy = entropy * agent_mask.float()
        action = action.masked_fill(~agent_mask, 0)
        return action.long(), logprob, entropy, value

    def act_deterministic(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        logits, _ = self.forward(obs)
        action = torch.argmax(logits, dim=-1)
        batch_size, num_agents = action.shape
        agent_mask = self._agent_mask(obs, batch_size, num_agents)
        return action.masked_fill(~agent_mask, 0).long()

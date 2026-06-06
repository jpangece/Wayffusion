from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from policies.candidate_selection_policy import _masked_mean
from policies.policy_output import PolicyOutput


class DirectWaypointPolicy(nn.Module):
    """Gaussian delta policy that directly outputs final waypoint coordinates."""

    def __init__(
        self,
        observation_space,
        action_space=None,
        cnn_channels: list[int] | None = None,
        agent_hidden_dim: int = 64,
        joint_hidden_dim: int = 192,
        attention_heads: int = 4,
        use_attention: bool = True,
        map_size: float = 1.0,
        max_delta: float = 0.22,
        log_std_init: float = -1.0,
        log_std_min: float = -2.0,
        log_std_max: float = 0.5,
    ):
        super().__init__()
        del action_space
        self.map_size = float(map_size)
        self.max_delta = float(max_delta)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

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

        task_dim = int(observation_space["task_id"].shape[0])
        global_dim = int(observation_space["global_info"].shape[0])
        actor_dim = agent_hidden_dim + self.field_dim + task_dim + global_dim
        self.actor = nn.Sequential(
            nn.Linear(actor_dim, agent_hidden_dim),
            nn.ReLU(),
            nn.Linear(agent_hidden_dim, agent_hidden_dim),
            nn.ReLU(),
            nn.Linear(agent_hidden_dim, 2),
        )
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
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

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        field = self._task_field(obs)
        field_feat = self.field_pool(self.field_encoder(field))
        agents = obs["all_uav_states"]
        batch_size, num_agents, _ = agents.shape
        agent_mask = self._agent_mask(obs, batch_size, num_agents)
        agent_tokens = self.agent_encoder(agents)
        if self.use_attention:
            attended, _ = self.agent_attention(agent_tokens, agent_tokens, agent_tokens, key_padding_mask=~agent_mask)
            agent_tokens = self.agent_attention_norm(agent_tokens + attended)
        field_per_agent = field_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        task_per_agent = obs["task_id"].unsqueeze(1).expand(batch_size, num_agents, -1)
        global_per_agent = obs["global_info"].unsqueeze(1).expand(batch_size, num_agents, -1)
        actor_input = torch.cat([agent_tokens, field_per_agent, task_per_agent, global_per_agent], dim=-1)
        delta_mean = torch.tanh(self.actor(actor_input)) * self.max_delta
        pooled_agents = _masked_mean(agent_tokens, agent_mask, dim=1)
        critic_input = torch.cat([field_feat, pooled_agents, obs["task_id"], obs["global_info"]], dim=-1)
        value = self.critic(critic_input).squeeze(-1)
        return delta_mean, value, agent_mask

    def _delta_to_waypoint(self, obs: dict[str, torch.Tensor], delta: torch.Tensor) -> torch.Tensor:
        positions = obs["all_uav_states"][..., :2] * self.map_size
        norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(self.max_delta / torch.clamp(norm, min=1e-8), max=1.0)
        clipped_delta = delta * scale
        return torch.clamp(positions + clipped_delta, 0.0, self.map_size)

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        train_action: torch.Tensor | None = None,
    ) -> PolicyOutput:
        delta_mean, value, agent_mask = self.forward(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).view(1, 1, 2).expand_as(delta_mean)
        dist = Normal(delta_mean, std)
        if train_action is None:
            delta = dist.rsample()
        else:
            delta = train_action.float()
        logprob = dist.log_prob(delta).sum(dim=-1) * agent_mask.float()
        entropy = dist.entropy().sum(dim=-1) * agent_mask.float()
        env_action = self._delta_to_waypoint(obs, delta)
        return PolicyOutput(
            env_action=env_action.float(),
            train_action=delta.float(),
            logprob=logprob,
            entropy=entropy,
            value=value,
            aux={"delta_mean": delta_mean, "policy_std_mean": torch.exp(log_std).mean(), "agent_mask": agent_mask},
        )

    def act_deterministic(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        delta_mean, _, _ = self.forward(obs)
        return self._delta_to_waypoint(obs, delta_mean).float()


__all__ = ["DirectWaypointPolicy"]

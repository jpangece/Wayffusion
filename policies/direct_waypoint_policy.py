from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from policies.candidate_selection_policy import _masked_mean
from policies.policy_output import PolicyOutput


def _observation_space_entries(observation_space):
    spaces = getattr(observation_space, "spaces", None)
    return spaces if spaces is not None else observation_space


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
        task_field_channel_mode: str = "all",
        task_field_channel_masks: dict | None = None,
        use_spatial_field_context: bool = False,
        spatial_context_radii: list[float] | None = None,
        use_field_moment_context: bool = False,
        field_moment_include_inverse: bool = False,
        use_connectivity_auxiliary_loss: bool = False,
        use_comm_graph_encoder: bool = False,
        use_uvfa_goal: bool = False,
        goal_embedding_dim: int = 64,
        goal_hidden_dim: int = 64,
        use_task_element_heatmap: bool = False,
    ):
        super().__init__()
        del action_space
        self.map_size = float(map_size)
        self.max_delta = float(max_delta)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.task_field_channel_mode = str(task_field_channel_mode)
        self.use_spatial_field_context = bool(use_spatial_field_context)
        self.use_field_moment_context = bool(use_field_moment_context)
        self.field_moment_include_inverse = bool(field_moment_include_inverse)
        self.use_connectivity_auxiliary_loss = bool(use_connectivity_auxiliary_loss)
        self.use_comm_graph_encoder = bool(use_comm_graph_encoder)
        self.use_uvfa_goal = bool(use_uvfa_goal)
        self.use_task_element_heatmap = bool(use_task_element_heatmap)

        cnn_channels = cnn_channels or [16, 32, 64]
        observation_spaces = _observation_space_entries(observation_space)
        task_field_space = observation_spaces["task_field"] if "task_field" in observation_spaces else observation_spaces["global_task_field"]
        in_channels = int(task_field_space.shape[0])
        field_height = int(task_field_space.shape[-2])
        field_width = int(task_field_space.shape[-1])
        self.in_channels = in_channels
        self.register_buffer("task_field_masks", self._build_task_field_masks(in_channels, task_field_channel_masks), persistent=False)
        offsets = self._build_spatial_offsets(spatial_context_radii or [0.06, 0.12])
        self.register_buffer("spatial_offsets", offsets, persistent=False)
        self.spatial_context_dim = int(offsets.shape[0] * in_channels) if self.use_spatial_field_context else 0
        self.field_moment_dim = int(in_channels * (6 if self.field_moment_include_inverse else 3)) if self.use_field_moment_context else 0
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, field_height),
            torch.linspace(0.0, 1.0, field_width),
            indexing="ij",
        )
        self.register_buffer("field_moment_x", xx.view(1, 1, field_height, field_width), persistent=False)
        self.register_buffer("field_moment_y", yy.view(1, 1, field_height, field_width), persistent=False)
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
        heatmap_dim = 0
        if self.use_task_element_heatmap:
            if "task_element_heatmap" not in observation_spaces:
                raise ValueError(
                    "task_element_heatmap must be present in observation_space when enabled"
                )
            heatmap_shape = tuple(observation_spaces["task_element_heatmap"].shape)
            if (
                len(heatmap_shape) != 3
                or heatmap_shape[0] < 1
                or heatmap_shape[1] < 1
                or heatmap_shape[2] < 1
                or heatmap_shape[1] != heatmap_shape[2]
            ):
                raise ValueError(
                    "task_element_heatmap observation space must have shape [C, G, G]"
                )
            self.task_element_heatmap_shape = heatmap_shape
            heatmap_hidden = max(4, int(cnn_channels[0]) // 2)
            heatmap_out = max(4, int(cnn_channels[-1]) // 2)
            self.heatmap_encoder = nn.Sequential(
                nn.Conv2d(heatmap_shape[0], heatmap_hidden, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(heatmap_hidden, heatmap_out, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
            )
            heatmap_dim = heatmap_out * 4

        agent_dim = int(observation_spaces["all_uav_states"].shape[-1])
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
        if self.use_comm_graph_encoder:
            self.comm_graph_update = nn.Sequential(
                nn.Linear(agent_hidden_dim + 1, agent_hidden_dim),
                nn.ReLU(),
                nn.Linear(agent_hidden_dim, agent_hidden_dim),
            )
            self.comm_graph_norm = nn.LayerNorm(agent_hidden_dim)

        task_dim = int(observation_spaces["task_id"].shape[0])
        global_dim = int(observation_spaces["global_info"].shape[0])
        if self.use_uvfa_goal:
            mission_goal_dim = int(observation_spaces["mission_goal"].shape[0])
            goal_mask_dim = int(observation_spaces["goal_mask"].shape[0])
            goal_input_dim = task_dim + mission_goal_dim + goal_mask_dim
            self.goal_encoder = nn.Sequential(
                nn.Linear(goal_input_dim, goal_hidden_dim),
                nn.ReLU(),
                nn.Linear(goal_hidden_dim, goal_embedding_dim),
                nn.ReLU(),
            )
            state_input_dim = self.field_dim + agent_hidden_dim + global_dim + heatmap_dim
            self.uvfa_state_encoder = nn.Sequential(
                nn.Linear(state_input_dim, joint_hidden_dim),
                nn.ReLU(),
                nn.Linear(joint_hidden_dim, goal_embedding_dim),
                nn.ReLU(),
            )
            actor_goal_dim = goal_embedding_dim
            critic_dim = goal_embedding_dim * 3
        else:
            actor_goal_dim = task_dim
            critic_dim = self.field_dim + agent_hidden_dim + task_dim + global_dim + heatmap_dim
        actor_dim = agent_hidden_dim + self.field_dim + actor_goal_dim + global_dim + self.spatial_context_dim + self.field_moment_dim + heatmap_dim
        self.actor = nn.Sequential(
            nn.Linear(actor_dim, agent_hidden_dim),
            nn.ReLU(),
            nn.Linear(agent_hidden_dim, agent_hidden_dim),
            nn.ReLU(),
            nn.Linear(agent_hidden_dim, 2),
        )
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
        self.critic = nn.Sequential(
            nn.Linear(critic_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, 1),
        )
        if self.use_connectivity_auxiliary_loss:
            aux_hidden = max(32, int(joint_hidden_dim) // 2)
            self.aux_connectivity_head = nn.Sequential(
                nn.Linear(critic_dim, aux_hidden),
                nn.ReLU(),
                nn.Linear(aux_hidden, 1),
            )
            self.aux_coverage_head = nn.Sequential(
                nn.Linear(critic_dim, aux_hidden),
                nn.ReLU(),
                nn.Linear(aux_hidden, 1),
            )

    def _build_task_field_masks(self, channels: int, masks: dict | None) -> torch.Tensor:
        defaults = {
            "area_coverage": [1.0, 1.0, 0.0, 1.0, 0.0],
            "belief_search": [1.0, 1.0, 1.0, 1.0, 0.0],
            "priority_inspection": [0.0, 1.0, 0.0, 1.0, 1.0],
            "connectivity_expansion": [1.0, 1.0, 0.0, 1.0, 0.0],
            "dynamic_target_escort": [0.0, 0.0, 0.0, 1.0, 1.0],
            "target_interception": [0.0, 0.0, 1.0, 1.0, 1.0],
        }
        names = [
            "area_coverage",
            "belief_search",
            "priority_inspection",
            "connectivity_expansion",
            "dynamic_target_escort",
            "target_interception",
        ]
        source = masks or defaults
        out = []
        for idx, name in enumerate(names):
            values = source.get(name, source.get(str(idx), [1.0] * channels)) if isinstance(source, dict) else [1.0] * channels
            vector = list(values)[:channels]
            if len(vector) < channels:
                vector.extend([0.0] * (channels - len(vector)))
            out.append(vector)
        return torch.as_tensor(out, dtype=torch.float32)

    def _build_spatial_offsets(self, radii: list[float]) -> torch.Tensor:
        offsets = [[0.0, 0.0]]
        directions = [
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (0.70710678, 0.70710678),
            (0.70710678, -0.70710678),
            (-0.70710678, 0.70710678),
            (-0.70710678, -0.70710678),
        ]
        for radius in radii:
            for dx, dy in directions:
                offsets.append([float(radius) * dx, float(radius) * dy])
        return torch.as_tensor(offsets, dtype=torch.float32)

    def _task_field(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["task_field"] if "task_field" in obs else obs["global_task_field"]

    def _masked_task_field(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        field = self._task_field(obs)
        if self.task_field_channel_mode == "all":
            mask = torch.ones((field.shape[0], field.shape[1]), dtype=field.dtype, device=field.device)
            return field, mask
        task_idx = torch.argmax(obs["task_id"], dim=-1).long().clamp(min=0, max=self.task_field_masks.shape[0] - 1)
        mask = self.task_field_masks.to(device=field.device, dtype=field.dtype)[task_idx, : field.shape[1]]
        return field * mask[:, :, None, None], mask

    def _agent_mask(self, obs: dict[str, torch.Tensor], batch_size: int, num_agents: int) -> torch.Tensor:
        if "agent_mask" in obs:
            return obs["agent_mask"].bool()
        return torch.ones((batch_size, num_agents), dtype=torch.bool, device=self._task_field(obs).device)

    def _encode_task_element_heatmap(
        self,
        obs: dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor | None:
        if not self.use_task_element_heatmap:
            return None
        if "task_element_heatmap" not in obs:
            raise KeyError("task_element_heatmap is required when heatmap consumption is enabled")
        heatmap = obs["task_element_heatmap"]
        if not isinstance(heatmap, torch.Tensor) or not torch.is_floating_point(heatmap):
            raise ValueError("task_element_heatmap must be a floating-point tensor")
        expected_shape = self.task_element_heatmap_shape
        if heatmap.ndim != 4 or tuple(heatmap.shape[1:]) != expected_shape:
            raise ValueError(
                "task_element_heatmap must have batched shape "
                f"[B, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}]"
            )
        if heatmap.shape[0] != batch_size:
            raise ValueError("task_element_heatmap batch size must match the observation batch")
        return self.heatmap_encoder(heatmap)

    def _forward_impl(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        field, field_mask = self._masked_task_field(obs)
        field_feat = self.field_pool(self.field_encoder(field))
        agents = obs["all_uav_states"]
        batch_size, num_agents, _ = agents.shape
        heatmap_feat = self._encode_task_element_heatmap(obs, batch_size)
        agent_mask = self._agent_mask(obs, batch_size, num_agents)
        agent_tokens = self.agent_encoder(agents)
        if self.use_attention:
            attended, _ = self.agent_attention(agent_tokens, agent_tokens, agent_tokens, key_padding_mask=~agent_mask)
            agent_tokens = self.agent_attention_norm(agent_tokens + attended)
        if self.use_comm_graph_encoder and "comm_adjacency" in obs:
            agent_tokens = self._comm_graph_context(agent_tokens, obs["comm_adjacency"], agent_mask)
        field_per_agent = field_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        global_per_agent = obs["global_info"].unsqueeze(1).expand(batch_size, num_agents, -1)
        pooled_agents = _masked_mean(agent_tokens, agent_mask, dim=1)
        aux_predictions: dict[str, torch.Tensor] = {}
        if self.use_uvfa_goal:
            goal_embedding = self.encode_goal(obs)
            goal_per_agent = goal_embedding.unsqueeze(1).expand(batch_size, num_agents, -1)
            state_embedding = self.uvfa_state_encoder(
                torch.cat(
                    [field_feat, pooled_agents, obs["global_info"]]
                    + ([heatmap_feat] if heatmap_feat is not None else []),
                    dim=-1,
                )
            )
            critic_input = torch.cat(
                [state_embedding, goal_embedding, state_embedding * goal_embedding],
                dim=-1,
            )
            pieces = [agent_tokens, field_per_agent, goal_per_agent, global_per_agent]
            aux_predictions.update(
                {
                    "goal_embedding_norm": torch.linalg.norm(goal_embedding, dim=-1).mean(),
                    "goal_embedding_variance": goal_embedding.var(dim=0, unbiased=False).mean(),
                }
            )
        else:
            task_per_agent = obs["task_id"].unsqueeze(1).expand(batch_size, num_agents, -1)
            pieces = [agent_tokens, field_per_agent, task_per_agent, global_per_agent]
            critic_input = torch.cat(
                [field_feat, pooled_agents, obs["task_id"], obs["global_info"]]
                + ([heatmap_feat] if heatmap_feat is not None else []),
                dim=-1,
            )
        if heatmap_feat is not None:
            pieces.append(heatmap_feat.unsqueeze(1).expand(batch_size, num_agents, -1))
        if self.use_spatial_field_context:
            pieces.append(self._spatial_field_context(field, agents[..., :2]))
        if self.use_field_moment_context:
            pieces.append(self._field_moment_context(field, field_mask, agents[..., :2]))
        actor_input = torch.cat(pieces, dim=-1)
        delta_mean = torch.tanh(self.actor(actor_input)) * self.max_delta
        value = self.critic(critic_input).squeeze(-1)
        if self.use_connectivity_auxiliary_loss:
            aux_predictions["aux_pred_connectivity_violation"] = torch.sigmoid(self.aux_connectivity_head(critic_input)).squeeze(-1)
            aux_predictions["aux_pred_coverage_gain"] = torch.sigmoid(self.aux_coverage_head(critic_input)).squeeze(-1)
        return delta_mean, value, agent_mask, aux_predictions

    def encode_goal(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.use_uvfa_goal:
            raise RuntimeError("encode_goal is only available when use_uvfa_goal=True")
        goal_input = torch.cat(
            [obs["task_id"], obs["mission_goal"], obs["goal_mask"]],
            dim=-1,
        )
        return self.goal_encoder(goal_input)

    def _comm_graph_context(
        self,
        agent_tokens: torch.Tensor,
        comm_adjacency: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_agents, _ = agent_tokens.shape
        adjacency = comm_adjacency.to(device=agent_tokens.device, dtype=agent_tokens.dtype)
        if adjacency.ndim != 3 or adjacency.shape[1] < num_agents + 1 or adjacency.shape[2] < num_agents + 1:
            return agent_tokens
        agent_adj = adjacency[:, 1 : num_agents + 1, 1 : num_agents + 1]
        agent_adj = torch.maximum(agent_adj, torch.eye(num_agents, device=agent_tokens.device, dtype=agent_tokens.dtype).view(1, num_agents, num_agents))
        valid_pair = agent_mask[:, :, None].float() * agent_mask[:, None, :].float()
        agent_adj = agent_adj * valid_pair
        degree = torch.clamp(agent_adj.sum(dim=-1, keepdim=True), min=1.0)
        graph_msg = torch.bmm(agent_adj / degree, agent_tokens)
        base_link = adjacency[:, 1 : num_agents + 1, 0:1] * agent_mask[:, :, None].float()
        update = self.comm_graph_update(torch.cat([graph_msg, base_link], dim=-1))
        return self.comm_graph_norm(agent_tokens + update * agent_mask[:, :, None].float())

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_mean, value, agent_mask, _ = self._forward_impl(obs)
        return delta_mean, value, agent_mask

    def _spatial_field_context(self, field: torch.Tensor, normalized_positions: torch.Tensor) -> torch.Tensor:
        batch_size, num_agents, _ = normalized_positions.shape
        offsets = self.spatial_offsets.to(device=field.device, dtype=field.dtype)
        sample_points = torch.clamp(normalized_positions.unsqueeze(2) + offsets.view(1, 1, -1, 2), 0.0, 1.0)
        grid = sample_points.mul(2.0).sub(1.0).reshape(batch_size, num_agents * offsets.shape[0], 1, 2)
        sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.squeeze(-1).permute(0, 2, 1).reshape(batch_size, num_agents, offsets.shape[0] * field.shape[1])
        return sampled

    def _field_moment_context(
        self,
        field: torch.Tensor,
        field_mask: torch.Tensor,
        normalized_positions: torch.Tensor,
    ) -> torch.Tensor:
        high = torch.clamp(field, min=0.0)
        contexts = [self._field_centroid_delta(high, field_mask, normalized_positions)]
        if self.field_moment_include_inverse:
            inverse = torch.clamp(1.0 - field, min=0.0) * field_mask[:, :, None, None]
            contexts.append(self._field_centroid_delta(inverse, field_mask, normalized_positions))
        return torch.cat(contexts, dim=-1)

    def _field_centroid_delta(
        self,
        weights: torch.Tensor,
        field_mask: torch.Tensor,
        normalized_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_agents, _ = normalized_positions.shape
        active = field_mask.to(device=weights.device, dtype=weights.dtype)[:, :, None, None]
        weights = weights * active
        mass = weights.sum(dim=(-2, -1))
        safe_mass = torch.clamp(mass, min=1e-6)
        center_x = (weights * self.field_moment_x.to(weights)).sum(dim=(-2, -1)) / safe_mass
        center_y = (weights * self.field_moment_y.to(weights)).sum(dim=(-2, -1)) / safe_mass
        active_channels = (mass > 1e-6).to(weights.dtype)
        center_x = center_x * active_channels + 0.5 * (1.0 - active_channels)
        center_y = center_y * active_channels + 0.5 * (1.0 - active_channels)
        mass_mean = mass / max(float(weights.shape[-2] * weights.shape[-1]), 1.0)
        dx = center_x[:, None, :] - normalized_positions[..., 0:1]
        dy = center_y[:, None, :] - normalized_positions[..., 1:2]
        mass_per_agent = mass_mean[:, None, :].expand(batch_size, num_agents, -1)
        return torch.cat([dx, dy, mass_per_agent], dim=-1)

    def _clip_delta(self, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(self.max_delta / torch.clamp(norm, min=1e-8), max=1.0)
        clipped_delta = delta * scale
        clip_mask = norm.squeeze(-1) > self.max_delta + 1e-6
        return clipped_delta, clip_mask

    def _delta_to_waypoint(self, obs: dict[str, torch.Tensor], delta: torch.Tensor) -> torch.Tensor:
        positions = obs["all_uav_states"][..., :2] * self.map_size
        clipped_delta, _ = self._clip_delta(delta)
        return torch.clamp(positions + clipped_delta, 0.0, self.map_size)

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        train_action: torch.Tensor | None = None,
    ) -> PolicyOutput:
        delta_mean, value, agent_mask, aux_predictions = self._forward_impl(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).view(1, 1, 2).expand_as(delta_mean)
        dist = Normal(delta_mean, std)
        if train_action is None:
            delta = dist.rsample()
        else:
            delta = train_action.float()
        logprob = dist.log_prob(delta).sum(dim=-1) * agent_mask.float()
        entropy = dist.entropy().sum(dim=-1) * agent_mask.float()
        clipped_delta, clip_mask = self._clip_delta(delta)
        env_action = self._delta_to_waypoint(obs, delta)
        valid_weights = agent_mask.float()
        valid_count = torch.clamp(valid_weights.sum(), min=1.0)
        delta_raw_norm = torch.linalg.norm(delta, dim=-1)
        delta_mean_norm = torch.linalg.norm(delta_mean, dim=-1)
        clipped_delta_norm = torch.linalg.norm(clipped_delta, dim=-1)
        return PolicyOutput(
            env_action=env_action.float(),
            train_action=delta.float(),
            logprob=logprob,
            entropy=entropy,
            value=value,
            aux={
                "delta_mean": delta_mean,
                "policy_std_mean": torch.exp(log_std).mean(),
                "log_std_mean": log_std.mean(),
                "log_std_min": log_std.min(),
                "log_std_max": log_std.max(),
                "delta_raw_norm_mean": (delta_raw_norm * valid_weights).sum() / valid_count,
                "delta_raw_norm_max": torch.where(agent_mask, delta_raw_norm, torch.zeros_like(delta_raw_norm)).max(),
                "delta_mean_norm_mean": (delta_mean_norm * valid_weights).sum() / valid_count,
                "clipped_delta_norm_mean": (clipped_delta_norm * valid_weights).sum() / valid_count,
                "clipped_delta_norm_max": torch.where(agent_mask, clipped_delta_norm, torch.zeros_like(clipped_delta_norm)).max(),
                "delta_clip_fraction": (clip_mask.float() * valid_weights).sum() / valid_count,
                "agent_mask": agent_mask,
                **aux_predictions,
            },
        )

    def act_deterministic(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        delta_mean, _, _ = self.forward(obs)
        return self._delta_to_waypoint(obs, delta_mean).float()


class UVFADirectWaypointPolicy(DirectWaypointPolicy):
    """Direct waypoint actor with a two-stream UVFA state-goal critic."""

    def __init__(self, *args, **kwargs):
        kwargs["use_uvfa_goal"] = True
        super().__init__(*args, **kwargs)


__all__ = ["DirectWaypointPolicy", "UVFADirectWaypointPolicy"]

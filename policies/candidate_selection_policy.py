from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical

from policies.policy_output import PolicyOutput


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=dim)
    weights = mask.float().unsqueeze(-1)
    return (values * weights).sum(dim=dim) / torch.clamp(weights.sum(dim=dim), min=1.0)


class CandidateSelectionWaypointPolicy(nn.Module):
    """Shared categorical actor that internally chooses candidate waypoints.

    The environment only receives final waypoint coordinates. Candidate
    generation, masking, sampling, and candidate-index log-probs stay inside the
    policy.
    """

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
        candidate_count: int = 12,
        map_size: float = 1.0,
        grid_size: int = 32,
        max_waypoint_distance: float = 0.22,
        min_uav_distance: float = 0.035,
        comm_radius: float = 0.36,
        base_comm_radius: float = 0.45,
        no_fly_zones: list[dict] | None = None,
        base_position: list[float] | None = None,
        candidate_generator: str = "task_aware",
        connectivity_candidate_filter: bool = True,
        candidate_prior_coef: float = 0.0,
        connectivity_chain_candidates: bool = False,
    ):
        super().__init__()
        del action_space
        self.candidate_count = int(candidate_count)
        self.map_size = float(map_size)
        self.grid_size = int(grid_size)
        self.max_waypoint_distance = float(max_waypoint_distance)
        self.min_uav_distance = float(min_uav_distance)
        self.comm_radius = float(comm_radius)
        self.base_comm_radius = float(base_comm_radius)
        self.no_fly_zones = list(no_fly_zones or [])
        self.base_position = torch.tensor(base_position or [0.1, 0.1], dtype=torch.float32)
        self.candidate_generator = str(candidate_generator)
        self.connectivity_candidate_filter = bool(connectivity_candidate_filter)
        self.candidate_prior_coef = float(candidate_prior_coef)
        self.connectivity_chain_candidates = bool(connectivity_chain_candidates)

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

        candidate_dim = 2 + 9
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

    def _encode_context(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        field = self._task_field(obs)
        field_feat = self.field_pool(self.field_encoder(field))
        agents = obs["all_uav_states"]
        batch_size, num_agents, _ = agents.shape
        agent_mask = self._agent_mask(obs, batch_size, num_agents)
        agent_tokens = self.agent_encoder(agents)
        if self.use_attention:
            attended, _ = self.agent_attention(agent_tokens, agent_tokens, agent_tokens, key_padding_mask=~agent_mask)
            agent_tokens = self.agent_attention_norm(agent_tokens + attended)
        return field_feat, agent_tokens, agent_mask, agents

    def _task_aware_score(self, field: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        coverage = field[:, 0]
        visits = field[:, 1]
        belief = field[:, 2]
        target = field[:, 4]
        area_score = (1.0 - coverage) / (1.0 + visits)
        belief_score = belief * (1.0 - 0.35 * visits.clamp(0.0, 1.0))
        target_score = target + 0.5 * belief
        connectivity_score = area_score
        task_weights = task_id.float()
        score = (
            task_weights[:, 0, None, None] * area_score
            + task_weights[:, 1, None, None] * belief_score
            + task_weights[:, 2, None, None] * target_score
            + task_weights[:, 3, None, None] * connectivity_score
            + task_weights[:, 4, None, None] * target_score
            + task_weights[:, 5, None, None] * target_score
        )
        fallback = area_score + belief_score + target_score
        return torch.where(task_weights.sum(dim=-1, keepdim=True).unsqueeze(-1) > 0.0, score, fallback)

    def _sample_field_values(self, field: torch.Tensor, candidates: torch.Tensor, channel: int) -> torch.Tensor:
        batch_size, num_agents, candidate_count, _ = candidates.shape
        grid = (candidates / max(self.map_size, 1e-6) * self.grid_size).long().clamp(0, self.grid_size - 1)
        flat_idx = grid[..., 1] * self.grid_size + grid[..., 0]
        values = field[:, channel].reshape(batch_size, -1).gather(1, flat_idx.reshape(batch_size, -1))
        return values.reshape(batch_size, num_agents, candidate_count)

    def _no_fly_mask(self, candidates: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(candidates.shape[:-1], dtype=torch.bool, device=candidates.device)
        for zone in self.no_fly_zones:
            if "center" in zone and "radius" in zone:
                center = torch.as_tensor(zone["center"], dtype=candidates.dtype, device=candidates.device)[:2]
                radius = float(zone["radius"])
                mask &= torch.linalg.norm(candidates - center, dim=-1) > radius
            elif {"x_min", "x_max", "y_min", "y_max"}.issubset(zone):
                inside = (
                    (candidates[..., 0] >= float(zone["x_min"]))
                    & (candidates[..., 0] <= float(zone["x_max"]))
                    & (candidates[..., 1] >= float(zone["y_min"]))
                    & (candidates[..., 1] <= float(zone["y_max"]))
                )
                mask &= ~inside
        return mask

    def _connectivity_features(self, positions: torch.Tensor, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.base_position.to(dtype=candidates.dtype, device=candidates.device)
        current_radius = torch.linalg.norm(positions - base.view(1, 1, 2), dim=-1).unsqueeze(-1)
        candidate_radius = torch.linalg.norm(candidates - base.view(1, 1, 1, 2), dim=-1)
        radial_gain = (candidate_radius - current_radius) / max(self.max_waypoint_distance, 1e-6)
        radius_norm = candidate_radius / max(self.map_size, 1e-6)

        base_margin = 1.0 - candidate_radius / max(self.base_comm_radius, 1e-6)
        pair_dist = torch.linalg.norm(candidates.unsqueeze(3) - positions.unsqueeze(1).unsqueeze(1), dim=-1)
        num_agents = positions.shape[1]
        if num_agents > 0:
            self_mask = torch.eye(num_agents, dtype=torch.bool, device=candidates.device).view(1, num_agents, 1, num_agents)
            pair_dist = pair_dist.masked_fill(self_mask, 1.0e6)
        nearest_neighbor = pair_dist.min(dim=-1).values
        neighbor_margin = 1.0 - nearest_neighbor / max(self.comm_radius, 1e-6)
        connectivity_margin = torch.maximum(base_margin, neighbor_margin)
        return radial_gain, radius_norm, connectivity_margin

    def _candidate_prior(self, task_id: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        distance = candidate_features[..., 2]
        unvisited = candidate_features[..., 3]
        belief = candidate_features[..., 4]
        radial_gain = candidate_features[..., 6]
        connectivity_margin = candidate_features[..., 8]
        stay_penalty = (distance < 1.0e-3).float()
        area_prior = 1.5 * unvisited + 0.1 * distance - 0.3 * stay_penalty
        belief_prior = 1.5 * belief + 0.5 * unvisited + 0.1 * distance - 0.3 * stay_penalty
        target_prior = 0.8 * belief + 0.4 * unvisited + 0.1 * distance - 0.3 * stay_penalty
        connectivity_prior = (
            1.0 * unvisited
            + 1.2 * radial_gain.clamp(min=0.0)
            + 1.0 * connectivity_margin.clamp(min=-0.5)
            + 0.2 * distance
            - 0.5 * stay_penalty
        )
        weights = task_id.float()
        priors = torch.stack([area_prior, belief_prior, target_prior, connectivity_prior, target_prior, target_prior], dim=-1)
        if weights.shape[-1] < priors.shape[-1]:
            pad = priors.shape[-1] - weights.shape[-1]
            weights = torch.nn.functional.pad(weights, (0, pad))
        weights = weights[:, : priors.shape[-1]]
        weighted = (priors * weights[:, None, None, :]).sum(dim=-1)
        fallback = area_prior + 0.5 * belief_prior
        has_task = weights.sum(dim=-1) > 0.0
        return torch.where(has_task[:, None, None], weighted, fallback)

    def _inject_connectivity_candidates(self, obs: dict[str, torch.Tensor], positions: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if not self.connectivity_chain_candidates:
            return candidates
        if obs["task_id"].shape[-1] <= 3 or candidates.shape[2] <= 1:
            return candidates
        is_connectivity = obs["task_id"][:, 3].float() > 0.5
        if not bool(is_connectivity.any()):
            return candidates

        batch_size, num_agents, candidate_count, _ = candidates.shape
        base = self.base_position.to(dtype=candidates.dtype, device=candidates.device)
        center = torch.full((2,), self.map_size * 0.5, dtype=candidates.dtype, device=candidates.device)
        primary = center - base
        primary_angle = torch.atan2(primary[1], primary[0])
        if num_agents > 1:
            role = torch.linspace(-0.7, 0.7, num_agents, dtype=candidates.dtype, device=candidates.device)
            rank = torch.linspace(0.0, 1.0, num_agents, dtype=candidates.dtype, device=candidates.device)
        else:
            role = torch.zeros(1, dtype=candidates.dtype, device=candidates.device)
            rank = torch.zeros(1, dtype=candidates.dtype, device=candidates.device)
        angles = primary_angle + role
        role_dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        lateral_dirs = torch.stack([-role_dirs[:, 1], role_dirs[:, 0]], dim=-1)
        target_radius = min(self.map_size * 0.85, self.base_comm_radius * 0.55) + rank * min(self.comm_radius * 0.85, self.map_size * 0.35)
        role_targets = (base.view(1, 2) + role_dirs * target_radius.unsqueeze(-1)).clamp(0.0, self.map_size)

        def local_step_to(target: torch.Tensor) -> torch.Tensor:
            delta = target.unsqueeze(0) - positions
            dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
            scale = torch.clamp(self.max_waypoint_distance / torch.clamp(dist, min=1.0e-6), max=1.0)
            return (positions + delta * scale).clamp(0.0, self.map_size)

        outward = positions - base.view(1, 1, 2)
        outward_norm = torch.linalg.norm(outward, dim=-1, keepdim=True)
        outward_dir = torch.where(outward_norm > 1.0e-6, outward / torch.clamp(outward_norm, min=1.0e-6), role_dirs.view(1, num_agents, 2))

        proposals = [
            local_step_to(role_targets),
            (positions + outward_dir * self.max_waypoint_distance).clamp(0.0, self.map_size),
            (positions + lateral_dirs.view(1, num_agents, 2) * self.max_waypoint_distance * 0.75).clamp(0.0, self.map_size),
            (positions - lateral_dirs.view(1, num_agents, 2) * self.max_waypoint_distance * 0.75).clamp(0.0, self.map_size),
        ]
        start = max(1, candidate_count - len(proposals))
        updated = candidates.clone()
        selector = is_connectivity.view(batch_size, 1, 1)
        for offset, proposal in enumerate(proposals[: candidate_count - start]):
            slot = start + offset
            updated[:, :, slot, :] = torch.where(selector, proposal, updated[:, :, slot, :])
        return updated

    def generate_candidates(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        field = self._task_field(obs)
        agents = obs["all_uav_states"]
        positions = agents[..., :2] * self.map_size
        batch_size, num_agents, _ = positions.shape
        k = self.candidate_count
        candidates = positions.unsqueeze(2).expand(batch_size, num_agents, k, 2).clone()

        radial_count = min(max(k // 2, 1), max(k - 1, 0))
        if radial_count > 0:
            idx = torch.arange(radial_count, dtype=positions.dtype, device=positions.device)
            angles = idx * (2.0 * math.pi / max(radial_count, 1))
            offsets = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) * self.max_waypoint_distance
            candidates[:, :, 1 : 1 + radial_count, :] = positions.unsqueeze(2) + offsets.view(1, 1, radial_count, 2)

        top_count = max(k - 1 - radial_count, 0)
        if top_count > 0:
            score = self._task_aware_score(field, obs["task_id"])
            flat = score.reshape(batch_size, -1)
            take = min(max(top_count * num_agents, top_count), flat.shape[-1])
            top_idx = torch.topk(flat, k=take, dim=-1).indices
            for agent_idx in range(num_agents):
                for local_idx in range(top_count):
                    src = top_idx[:, (agent_idx * top_count + local_idx) % take]
                    y = torch.div(src, self.grid_size, rounding_mode="floor")
                    x = src % self.grid_size
                    point = torch.stack([x.float() + 0.5, y.float() + 0.5], dim=-1) / float(self.grid_size) * self.map_size
                    candidates[:, agent_idx, 1 + radial_count + local_idx, :] = point.to(candidates.dtype)

        candidates = self._inject_connectivity_candidates(obs, positions, candidates)
        candidates = candidates.clamp(0.0, self.map_size)
        rel = candidates - positions.unsqueeze(2)
        distance = torch.linalg.norm(rel, dim=-1)
        mask = distance <= self.max_waypoint_distance + 1e-6
        mask &= self._no_fly_mask(candidates)
        mask &= self._agent_mask(obs, batch_size, num_agents).unsqueeze(-1)
        radial_gain, radius_norm, connectivity_margin = self._connectivity_features(positions, candidates)
        if self.connectivity_candidate_filter and obs["task_id"].shape[-1] > 3:
            is_connectivity = obs["task_id"][:, 3].float() > 0.5
            connected_candidate = connectivity_margin >= -1e-6
            mask = torch.where(is_connectivity.view(batch_size, 1, 1), mask & connected_candidate, mask)
        empty = ~mask.any(dim=-1)
        if empty.any():
            empty_b, empty_n = empty.nonzero(as_tuple=True)
            candidates[empty_b, empty_n, 0, :] = positions[empty_b, empty_n]
            mask[empty_b, empty_n, 0] = True

        rel = candidates - positions.unsqueeze(2)
        distance = torch.linalg.norm(rel, dim=-1)
        radial_gain, radius_norm, connectivity_margin = self._connectivity_features(positions, candidates)
        rel_norm = rel / max(self.map_size, 1e-6)
        dist_norm = distance.unsqueeze(-1) / max(self.max_waypoint_distance, 1e-6)
        unvisited = (1.0 - self._sample_field_values(field, candidates, 0)).unsqueeze(-1)
        belief = self._sample_field_values(field, candidates, 2).unsqueeze(-1)
        no_fly = self._no_fly_mask(candidates).float().unsqueeze(-1)
        features = torch.cat(
            [
                rel_norm,
                dist_norm,
                unvisited,
                belief,
                no_fly,
                radial_gain.clamp(-1.0, 1.0).unsqueeze(-1),
                radius_norm.clamp(0.0, 2.0).unsqueeze(-1),
                connectivity_margin.clamp(-1.0, 1.0).unsqueeze(-1),
            ],
            dim=-1,
        )
        return candidates.float(), features.float(), mask.bool()

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        field_feat, agent_tokens, agent_mask, _ = self._encode_context(obs)
        batch_size, num_agents, _ = agent_tokens.shape
        candidates, candidate_features, candidate_mask = self.generate_candidates(obs)
        candidate_input = torch.cat([candidates / max(self.map_size, 1e-6), candidate_features], dim=-1)
        candidate_tokens = self.candidate_encoder(candidate_input)
        candidate_count = candidate_tokens.shape[2]

        field_per_agent = field_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        task_per_agent = obs["task_id"].unsqueeze(1).expand(batch_size, num_agents, -1)
        global_per_agent = obs["global_info"].unsqueeze(1).expand(batch_size, num_agents, -1)
        actor_context = torch.cat([agent_tokens, field_per_agent, task_per_agent, global_per_agent], dim=-1)
        actor_context = self.actor_context(actor_context).unsqueeze(2).expand(-1, -1, candidate_count, -1)
        logits = self.logit_head(torch.tanh(actor_context + candidate_tokens)).squeeze(-1)
        if self.candidate_prior_coef != 0.0:
            logits = logits + float(self.candidate_prior_coef) * self._candidate_prior(obs["task_id"], candidate_features)
        logits = logits.masked_fill(~candidate_mask, -1.0e9)
        logits = logits.masked_fill(~agent_mask.unsqueeze(-1), -1.0e9)

        pooled_agents = _masked_mean(agent_tokens, agent_mask, dim=1)
        critic_input = torch.cat([field_feat, pooled_agents, obs["task_id"], obs["global_info"]], dim=-1)
        value = self.critic(critic_input).squeeze(-1)
        aux = {
            "candidate_waypoints": candidates,
            "candidate_features": candidate_features,
            "candidate_mask": candidate_mask,
            "agent_mask": agent_mask,
        }
        return logits, value, aux

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        train_action: torch.Tensor | None = None,
    ) -> PolicyOutput:
        logits, value, aux = self.forward(obs)
        batch_size, num_agents, candidate_count = logits.shape
        dist = Categorical(logits=logits.reshape(batch_size * num_agents, candidate_count))
        if train_action is None:
            flat_action = dist.sample()
            action = flat_action.view(batch_size, num_agents)
        else:
            action = train_action.long()
            flat_action = action.reshape(-1)
        logprob = dist.log_prob(flat_action).view(batch_size, num_agents)
        entropy = dist.entropy().view(batch_size, num_agents)
        agent_mask = aux["agent_mask"]
        logprob = logprob * agent_mask.float()
        entropy = entropy * agent_mask.float()
        action = action.masked_fill(~agent_mask, 0).long()
        gather_idx = action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        env_action = torch.gather(aux["candidate_waypoints"], dim=2, index=gather_idx).squeeze(2)
        aux = dict(aux)
        aux["selected_index"] = action
        return PolicyOutput(env_action=env_action.float(), train_action=action, logprob=logprob, entropy=entropy, value=value, aux=aux)

    def act_deterministic(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        logits, _, aux = self.forward(obs)
        action = self._deconflicted_argmax(logits, aux["candidate_waypoints"], aux["candidate_mask"], aux["agent_mask"])
        gather_idx = action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        return torch.gather(aux["candidate_waypoints"], dim=2, index=gather_idx).squeeze(2).float()

    def _deconflicted_argmax(
        self,
        logits: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_agents, candidate_count = logits.shape
        action = torch.zeros((batch_size, num_agents), dtype=torch.long, device=logits.device)
        for batch_idx in range(batch_size):
            selected_points: list[torch.Tensor] = []
            for agent_idx in range(num_agents):
                if not bool(agent_mask[batch_idx, agent_idx]):
                    continue
                order = torch.argsort(logits[batch_idx, agent_idx], descending=True)
                chosen = int(order[0].item())
                for cand_idx_t in order:
                    cand_idx = int(cand_idx_t.item())
                    if not bool(candidate_mask[batch_idx, agent_idx, cand_idx]):
                        continue
                    point = candidates[batch_idx, agent_idx, cand_idx]
                    if all(torch.linalg.norm(point - prev) >= self.min_uav_distance for prev in selected_points):
                        chosen = cand_idx
                        break
                action[batch_idx, agent_idx] = chosen
                selected_points.append(candidates[batch_idx, agent_idx, chosen])
        return action.masked_fill(~agent_mask, 0).long()


MAPPOWaypointPolicy = CandidateSelectionWaypointPolicy


__all__ = ["CandidateSelectionWaypointPolicy", "MAPPOWaypointPolicy"]

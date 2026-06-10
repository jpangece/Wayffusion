from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn


def _obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in obs.items()}


class RunningMeanStd:
    """Online scalar normalizer for waypoint MAPPO rewards."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = values.size
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / np.sqrt(self.var + 1e-8)


@dataclass
class MAPPOWaypointBatch:
    observations: dict[str, np.ndarray]
    env_actions: np.ndarray
    train_actions: np.ndarray
    old_logprobs: np.ndarray
    values: np.ndarray
    team_rewards: np.ndarray
    per_agent_rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    bootstrap_mask: np.ndarray
    done_for_reset: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    infos: list


class WaypointRolloutBuffer:
    def __init__(self):
        self.observations = []
        self.env_actions = []
        self.train_actions = []
        self.logprobs = []
        self.values = []
        self.team_rewards = []
        self.per_agent_rewards = []
        self.terminated = []
        self.truncated = []
        self.bootstrap_mask = []
        self.done_for_reset = []
        self.infos = []

    def add(self, obs, env_action, train_action, logprob, value, team_reward, per_agent_reward, terminated, truncated, bootstrap_mask, done_for_reset, infos):
        self.observations.append({key: np.asarray(item).copy() for key, item in obs.items()})
        self.env_actions.append(np.asarray(env_action, dtype=np.float32).copy())
        self.train_actions.append(np.asarray(train_action).copy())
        self.logprobs.append(np.asarray(logprob, dtype=np.float32).copy())
        self.values.append(np.asarray(value, dtype=np.float32).copy())
        self.team_rewards.append(np.asarray(team_reward, dtype=np.float32).copy())
        self.per_agent_rewards.append(np.asarray(per_agent_reward, dtype=np.float32).copy())
        self.terminated.append(np.asarray(terminated, dtype=bool).copy())
        self.truncated.append(np.asarray(truncated, dtype=bool).copy())
        self.bootstrap_mask.append(np.asarray(bootstrap_mask, dtype=np.float32).copy())
        self.done_for_reset.append(np.asarray(done_for_reset, dtype=bool).copy())
        self.infos.append(infos)

    def build(self, advantages: np.ndarray, returns: np.ndarray) -> MAPPOWaypointBatch:
        obs = {key: np.stack([item[key] for item in self.observations], axis=0) for key in self.observations[0]}
        return MAPPOWaypointBatch(
            observations=obs,
            env_actions=np.stack(self.env_actions, axis=0).astype(np.float32),
            train_actions=np.stack(self.train_actions, axis=0),
            old_logprobs=np.stack(self.logprobs, axis=0).astype(np.float32),
            values=np.stack(self.values, axis=0).astype(np.float32),
            team_rewards=np.stack(self.team_rewards, axis=0).astype(np.float32),
            per_agent_rewards=np.stack(self.per_agent_rewards, axis=0).astype(np.float32),
            terminated=np.stack(self.terminated, axis=0).astype(bool),
            truncated=np.stack(self.truncated, axis=0).astype(bool),
            bootstrap_mask=np.stack(self.bootstrap_mask, axis=0).astype(np.float32),
            done_for_reset=np.stack(self.done_for_reset, axis=0).astype(bool),
            advantages=advantages.astype(np.float32),
            returns=returns.astype(np.float32),
            infos=self.infos,
        )


def compute_gae_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    done_for_reset: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute team GAE without propagating advantages across reset episodes.

    Time-limit truncations still bootstrap value targets, but they must not let
    the recursive GAE state leak into the next auto-reset episode.
    """

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=bool)
    done_for_reset = np.asarray(done_for_reset, dtype=bool)
    value_bootstrap_mask = 1.0 - terminated.astype(np.float32)
    gae_continuation_mask = 1.0 - done_for_reset.astype(np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros((rewards.shape[1],), dtype=np.float32)
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + float(gamma) * next_values[t] * value_bootstrap_mask[t] - values[t]
        gae = delta + float(gamma) * float(gae_lambda) * gae_continuation_mask[t] * gae
        advantages[t] = gae
    return advantages, advantages + values


class MAPPOWaypointTrainer:
    """MAPPO trainer for final waypoint env actions and per-agent PPO ratios."""

    def __init__(self, env_batch, policy: nn.Module, train_config: dict, device: str | None = None):
        self.env_batch = env_batch
        self.policy = policy
        self.train_config = train_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(train_config["learning_rate"]))
        self.reward_rms = RunningMeanStd()
        self.current_obs, _ = self.env_batch.reset()
        self.global_step = 0
        self.completed_episodes = 0
        self.use_lagrangian_connectivity_cost = bool(train_config.get("use_lagrangian_connectivity_cost", False))
        self.lambda_connectivity_cost = float(train_config.get("lambda_cost_init", 1.0))
        self.lambda_connectivity_cost = float(
            np.clip(
                self.lambda_connectivity_cost,
                float(train_config.get("lambda_cost_min", 0.0)),
                float(train_config.get("lambda_cost_max", 50.0)),
            )
        )

    def _flatten_time_env(self, array: np.ndarray) -> np.ndarray:
        return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])

    def _flatten_obs(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: self._flatten_time_env(value) for key, value in observations.items()}

    def _set_lr(self, update_idx: int, total_updates: int) -> None:
        if self.train_config.get("lr_schedule", "constant") != "linear":
            return
        frac = 1.0 - ((update_idx - 1.0) / max(total_updates, 1))
        lr = float(self.train_config["learning_rate"]) * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _update_lagrangian_connectivity_cost(self, mean_cost: float) -> dict[str, float]:
        target = float(self.train_config.get("connectivity_cost_target", 0.05))
        error = float(mean_cost) - target
        if self.use_lagrangian_connectivity_cost:
            lr = float(self.train_config.get("lagrangian_lr", 0.01))
            self.lambda_connectivity_cost = float(
                np.clip(
                    self.lambda_connectivity_cost + lr * error,
                    float(self.train_config.get("lambda_cost_min", 0.0)),
                    float(self.train_config.get("lambda_cost_max", 50.0)),
                )
            )
        return {
            "lambda_connectivity_cost": float(self.lambda_connectivity_cost if self.use_lagrangian_connectivity_cost else 0.0),
            "connectivity_cost_target": float(target),
            "connectivity_cost_error": float(error),
        }

    def _info_scalar(self, info: dict, key: str, default: float = 0.0) -> float:
        value = info.get(key, None)
        if value is None:
            value = info.get("task_metrics", {}).get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def collect_rollout(self) -> tuple[MAPPOWaypointBatch, dict]:
        horizon = int(self.train_config["rollout_steps"])
        gamma = float(self.train_config["gamma"])
        gae_lambda = float(self.train_config["gae_lambda"])
        buffer = WaypointRolloutBuffer()
        next_values = []
        rollout_rewards = []
        valid_rates = []
        mask_empty_counts = []
        invalid_counts = []
        connectivity_violations = []
        coverage_values = []
        searched_values = []
        poi_values = []
        target_values = []
        interception_values = []
        mean_speeds = []
        max_speeds = []
        mean_control_norms = []
        mean_distance_to_waypoint_m = []
        arrival_rates = []
        command_update_rates = []
        command_reject_rates = []
        mean_desired_speed_mps = []
        collision_counts = []
        no_fly_counts = []
        no_fly_zone_counts = []
        domain_randomization_flags = []
        geofence_counts = []
        real_mpe_flags = []
        mpe_step_calls = []
        terminal_success = []
        terminal_path = []
        terminal_collision = []
        env_action_means = []
        env_action_stds = []
        env_action_mins = []
        env_action_maxs = []
        train_action_means = []
        train_action_stds = []
        train_action_mins = []
        train_action_maxs = []
        policy_std_means = []
        log_std_means = []
        log_std_mins = []
        log_std_maxs = []
        delta_raw_norm_means = []
        delta_raw_norm_maxs = []
        delta_mean_norm_means = []
        clipped_delta_norm_means = []
        clipped_delta_norm_maxs = []
        delta_clip_fractions = []
        raw_team_rewards = []
        penalized_team_rewards = []
        connectivity_costs = []
        lagrangian_penalties = []
        extra_metric_keys = [
            "connected_to_base_ratio",
            "connectivity_violation",
            "connectivity_margin_mean",
            "connectivity_margin_min",
            "connectivity_margin_penalty",
            "action_disconnect_risk",
            "action_disconnect_risk_mean",
            "predicted_connected_to_base_ratio",
            "predicted_connectivity_violation",
            "unsafe_action_ratio",
            "action_risk_penalty",
            "connectivity_cost",
            "raw_task_reward",
            "penalized_reward",
            "lagrangian_penalty",
            "coverage_gain",
            "connected_new_coverage_gain",
            "disconnected_new_coverage_gain",
            "connected_new_coverage_cells",
            "disconnected_new_coverage_cells",
            "connected_new_coverage_ratio",
            "disconnected_new_coverage_ratio",
            "connected_coverage_reward",
            "disconnected_coverage_penalty",
            "uncovered_approach_gain",
            "repeat_footprint_ratio",
            "connected_radius_hold",
            "connected_angular_spread",
            "relaxed_success",
            "relaxed_success_coverage",
            "relaxed_success_radius",
            "relaxed_max_violation_rate",
            "connectivity_action_filter_enabled",
            "connectivity_filter_replacement_count",
        ]
        extra_metrics = {key: [] for key in extra_metric_keys}
        start = time.perf_counter()
        for _ in range(horizon):
            obs_tensor = _obs_to_tensor(self.current_obs, self.device)
            with torch.no_grad():
                policy_output = self.policy.get_action_and_value(obs_tensor)
            env_actions = policy_output.env_action.detach().cpu().numpy().astype(np.float32)
            train_actions = policy_output.train_action
            if train_actions is None:
                train_actions = policy_output.env_action
            train_actions_np = train_actions.detach().cpu().numpy()
            env_action_means.append(float(np.mean(env_actions)))
            env_action_stds.append(float(np.std(env_actions)))
            env_action_mins.append(float(np.min(env_actions)))
            env_action_maxs.append(float(np.max(env_actions)))
            train_action_means.append(float(np.mean(train_actions_np)))
            train_action_stds.append(float(np.std(train_actions_np)))
            train_action_mins.append(float(np.min(train_actions_np)))
            train_action_maxs.append(float(np.max(train_actions_np)))
            aux = policy_output.aux or {}
            if "policy_std_mean" in aux:
                policy_std_means.append(float(torch.as_tensor(aux["policy_std_mean"]).detach().cpu().item()))
            if "log_std_mean" in aux:
                log_std_means.append(float(torch.as_tensor(aux["log_std_mean"]).detach().cpu().item()))
            if "log_std_min" in aux:
                log_std_mins.append(float(torch.as_tensor(aux["log_std_min"]).detach().cpu().item()))
            if "log_std_max" in aux:
                log_std_maxs.append(float(torch.as_tensor(aux["log_std_max"]).detach().cpu().item()))
            if "delta_raw_norm_mean" in aux:
                delta_raw_norm_means.append(float(torch.as_tensor(aux["delta_raw_norm_mean"]).detach().cpu().item()))
            if "delta_raw_norm_max" in aux:
                delta_raw_norm_maxs.append(float(torch.as_tensor(aux["delta_raw_norm_max"]).detach().cpu().item()))
            if "delta_mean_norm_mean" in aux:
                delta_mean_norm_means.append(float(torch.as_tensor(aux["delta_mean_norm_mean"]).detach().cpu().item()))
            if "clipped_delta_norm_mean" in aux:
                clipped_delta_norm_means.append(float(torch.as_tensor(aux["clipped_delta_norm_mean"]).detach().cpu().item()))
            if "clipped_delta_norm_max" in aux:
                clipped_delta_norm_maxs.append(float(torch.as_tensor(aux["clipped_delta_norm_max"]).detach().cpu().item()))
            if "delta_clip_fraction" in aux:
                delta_clip_fractions.append(float(torch.as_tensor(aux["delta_clip_fraction"]).detach().cpu().item()))
            step = self.env_batch.step(env_actions)
            with torch.no_grad():
                bootstrap_value = self.policy.get_action_and_value(_obs_to_tensor(step.bootstrap_observations, self.device)).value
            raw_rewards = step.team_rewards.copy()
            step_costs = np.asarray([self._info_scalar(info, "connectivity_cost", 0.0) for info in step.infos], dtype=np.float32)
            applied_lambda = float(self.lambda_connectivity_cost if self.use_lagrangian_connectivity_cost else 0.0)
            step_lagrangian_penalty = applied_lambda * step_costs
            train_rewards = raw_rewards - step_lagrangian_penalty
            raw_team_rewards.extend(float(value) for value in raw_rewards)
            penalized_team_rewards.extend(float(value) for value in train_rewards)
            connectivity_costs.extend(float(value) for value in step_costs)
            lagrangian_penalties.extend(float(value) for value in step_lagrangian_penalty)
            if self.train_config.get("reward_norm", True):
                self.reward_rms.update(train_rewards)
                train_rewards = self.reward_rms.normalize(train_rewards)
            buffer.add(
                self.current_obs,
                env_actions,
                train_actions.detach().cpu().numpy(),
                policy_output.logprob.detach().cpu().numpy(),
                policy_output.value.detach().cpu().numpy(),
                train_rewards,
                step.per_agent_rewards,
                step.terminated,
                step.truncated,
                step.bootstrap_mask,
                step.done_for_reset,
                step.infos,
            )
            next_values.append(bootstrap_value.detach().cpu().numpy())
            rollout_rewards.append(float(np.mean(train_rewards if self.use_lagrangian_connectivity_cost else step.team_rewards)))
            self.completed_episodes += int(np.count_nonzero(step.done_for_reset))
            for info in step.infos:
                num_agents = max(len(info.get("per_agent_rewards", [])), 1)
                valid_rates.append(float(1.0 - info.get("invalid_action_count", 0) / num_agents))
                mask_empty_counts.append(float(info.get("candidate_mask_empty_count", 0)))
                invalid_counts.append(float(info.get("invalid_action_count", 0)))
                connectivity_violations.append(float(info.get("connectivity_violation_rate", 0.0)))
                coverage_values.append(float(info.get("coverage_ratio", 0.0)))
                searched_values.append(float(info.get("searched_probability_mass", 0.0)))
                poi_values.append(float(info.get("weighted_poi_completion", 0.0)))
                target_values.append(float(info.get("target_coverage_ratio", 0.0)))
                interception_values.append(float(info.get("interception_success", 0.0)))
                mean_speeds.append(float(info.get("mean_speed", 0.0)))
                max_speeds.append(float(info.get("max_speed_observed", 0.0)))
                mean_control_norms.append(float(info.get("mean_control_norm", 0.0)))
                mean_distance_to_waypoint_m.append(float(info.get("mean_distance_to_waypoint_m", 0.0)))
                arrival_rates.append(float(info.get("arrival_rate", 0.0)))
                command_update_rates.append(float(info.get("command_update_rate", 1.0)))
                command_reject_rates.append(float(info.get("command_reject_rate", 0.0)))
                mean_desired_speed_mps.append(float(info.get("mean_desired_speed_mps", 0.0)))
                collision_counts.append(float(info.get("collision_count", 0.0)))
                no_fly_counts.append(float(info.get("no_fly_violation_count", 0.0)))
                no_fly_zone_counts.append(float(info.get("no_fly_zone_count", 0.0)))
                domain_randomization_flags.append(float(bool(info.get("domain_randomization_enabled", False))))
                geofence_counts.append(float(info.get("geofence_violation_count", 0.0)))
                for key in extra_metric_keys:
                    extra_metrics[key].append(self._info_scalar(info, key, 0.0))
                dyn = info.get("dynamics_info", {})
                real_mpe_flags.append(float(bool(dyn.get("uses_real_mpe_core", False))))
                mpe_step_calls.append(float(dyn.get("mpe_world_step_calls", 0.0)))
                if "terminal_info" in info:
                    terminal_success.append(float(info.get("success", False)))
                    terminal_path.append(float(info.get("path_length", 0.0)))
                    terminal_collision.append(float(info.get("collision_rate", 0.0)))
            self.current_obs = step.observations
            self.global_step += self.env_batch.num_envs

        rewards = np.stack(buffer.team_rewards, axis=0).astype(np.float32)
        values = np.stack(buffer.values, axis=0).astype(np.float32)
        next_values_arr = np.stack(next_values, axis=0).astype(np.float32)
        terminated = np.stack(buffer.terminated, axis=0).astype(bool)
        done_for_reset = np.stack(buffer.done_for_reset, axis=0).astype(bool)
        advantages, returns = compute_gae_returns(
            rewards,
            values,
            next_values_arr,
            terminated,
            done_for_reset,
            gamma,
            gae_lambda,
        )
        elapsed = time.perf_counter() - start
        mean_cost = float(np.mean(connectivity_costs)) if connectivity_costs else 0.0
        lagrangian_stats = self._update_lagrangian_connectivity_cost(mean_cost)
        rollout_result = {
            "mean_team_reward": float(np.mean(rollout_rewards)) if rollout_rewards else 0.0,
            "mean_rollout_reward": float(np.mean(rollout_rewards)) if rollout_rewards else 0.0,
            "mean_raw_team_reward": float(np.mean(raw_team_rewards)) if raw_team_rewards else 0.0,
            "mean_penalized_team_reward": float(np.mean(penalized_team_rewards)) if penalized_team_rewards else 0.0,
            "mean_per_agent_reward": float(np.mean(np.stack(buffer.per_agent_rewards))) if buffer.per_agent_rewards else 0.0,
            "action_validity_rate": float(np.mean(valid_rates)) if valid_rates else 1.0,
            "candidate_mask_empty_count": float(np.sum(mask_empty_counts)) if mask_empty_counts else 0.0,
            "invalid_action_count": float(np.sum(invalid_counts)) if invalid_counts else 0.0,
            "connectivity_violation_rate": float(np.mean(connectivity_violations)) if connectivity_violations else 0.0,
            "coverage_ratio": float(np.mean(coverage_values)) if coverage_values else 0.0,
            "searched_probability_mass": float(np.mean(searched_values)) if searched_values else 0.0,
            "weighted_poi_completion": float(np.mean(poi_values)) if poi_values else 0.0,
            "target_coverage_ratio": float(np.mean(target_values)) if target_values else 0.0,
            "interception_success": float(np.mean(interception_values)) if interception_values else 0.0,
            "mean_speed": float(np.mean(mean_speeds)) if mean_speeds else 0.0,
            "max_speed_observed": float(np.max(max_speeds)) if max_speeds else 0.0,
            "mean_control_norm": float(np.mean(mean_control_norms)) if mean_control_norms else 0.0,
            "mean_distance_to_waypoint_m": float(np.mean(mean_distance_to_waypoint_m)) if mean_distance_to_waypoint_m else 0.0,
            "arrival_rate": float(np.mean(arrival_rates)) if arrival_rates else 0.0,
            "command_update_rate": float(np.mean(command_update_rates)) if command_update_rates else 1.0,
            "command_reject_rate": float(np.mean(command_reject_rates)) if command_reject_rates else 0.0,
            "mean_desired_speed_mps": float(np.mean(mean_desired_speed_mps)) if mean_desired_speed_mps else 0.0,
            "collision_count": float(np.sum(collision_counts)) if collision_counts else 0.0,
            "no_fly_violation_count": float(np.sum(no_fly_counts)) if no_fly_counts else 0.0,
            "mean_no_fly_zone_count": float(np.mean(no_fly_zone_counts)) if no_fly_zone_counts else 0.0,
            "domain_randomization_enabled": float(np.mean(domain_randomization_flags)) if domain_randomization_flags else 0.0,
            "geofence_violation_count": float(np.sum(geofence_counts)) if geofence_counts else 0.0,
            "uses_real_mpe_core": float(np.mean(real_mpe_flags)) if real_mpe_flags else 0.0,
            "mpe_world_step_calls": float(np.mean(mpe_step_calls)) if mpe_step_calls else 0.0,
            "rollout_terminal_success_rate": float(np.mean(terminal_success)) if terminal_success else 0.0,
            "rollout_terminal_path_length": float(np.mean(terminal_path)) if terminal_path else 0.0,
            "rollout_terminal_collision_rate": float(np.mean(terminal_collision)) if terminal_collision else 0.0,
            "env_action_mean": float(np.mean(env_action_means)) if env_action_means else 0.0,
            "env_action_std": float(np.mean(env_action_stds)) if env_action_stds else 0.0,
            "env_action_min": float(np.min(env_action_mins)) if env_action_mins else 0.0,
            "env_action_max": float(np.max(env_action_maxs)) if env_action_maxs else 0.0,
            "train_action_mean": float(np.mean(train_action_means)) if train_action_means else 0.0,
            "train_action_std": float(np.mean(train_action_stds)) if train_action_stds else 0.0,
            "train_action_min": float(np.min(train_action_mins)) if train_action_mins else 0.0,
            "train_action_max": float(np.max(train_action_maxs)) if train_action_maxs else 0.0,
            "policy_std_mean": float(np.mean(policy_std_means)) if policy_std_means else 0.0,
            "log_std_mean": float(np.mean(log_std_means)) if log_std_means else 0.0,
            "log_std_min": float(np.min(log_std_mins)) if log_std_mins else 0.0,
            "log_std_max": float(np.max(log_std_maxs)) if log_std_maxs else 0.0,
            "delta_raw_norm_mean": float(np.mean(delta_raw_norm_means)) if delta_raw_norm_means else 0.0,
            "delta_raw_norm_max": float(np.max(delta_raw_norm_maxs)) if delta_raw_norm_maxs else 0.0,
            "delta_mean_norm_mean": float(np.mean(delta_mean_norm_means)) if delta_mean_norm_means else 0.0,
            "clipped_delta_norm_mean": float(np.mean(clipped_delta_norm_means)) if clipped_delta_norm_means else 0.0,
            "clipped_delta_norm_max": float(np.max(clipped_delta_norm_maxs)) if clipped_delta_norm_maxs else 0.0,
            "delta_clip_fraction": float(np.mean(delta_clip_fractions)) if delta_clip_fractions else 0.0,
            "connectivity_cost": mean_cost,
            "lagrangian_penalty": float(np.mean(lagrangian_penalties)) if lagrangian_penalties else 0.0,
            "episodes_completed": int(np.count_nonzero(np.stack(buffer.done_for_reset))),
            "cumulative_episodes": int(self.completed_episodes),
            "rollout_steps_per_sec": float(horizon * self.env_batch.num_envs / max(elapsed, 1e-6)),
            "rollout_wall_clock_time": float(elapsed),
        }
        for key, values in extra_metrics.items():
            if values:
                rollout_result[key] = float(np.mean(values))
                rollout_result[f"{key}_mean"] = float(np.mean(values))
        actual_raw_reward = float(np.mean(raw_team_rewards)) if raw_team_rewards else 0.0
        actual_penalized_reward = float(np.mean(penalized_team_rewards)) if penalized_team_rewards else 0.0
        actual_lagrangian_penalty = float(np.mean(lagrangian_penalties)) if lagrangian_penalties else 0.0
        rollout_result.update(
            {
                "raw_task_reward": actual_raw_reward,
                "raw_task_reward_mean": actual_raw_reward,
                "penalized_reward": actual_penalized_reward,
                "penalized_reward_mean": actual_penalized_reward,
                "lagrangian_penalty": actual_lagrangian_penalty,
                "lagrangian_penalty_mean": actual_lagrangian_penalty,
            }
        )
        rollout_result.update(lagrangian_stats)
        return buffer.build(advantages, returns), rollout_result

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float()
        return (values * weights).sum() / torch.clamp(weights.sum(), min=1.0)

    def _flatten_info_metric(self, infos: list, key: str, default: float = 0.0) -> np.ndarray:
        values = []
        for timestep_infos in infos:
            for info in timestep_infos:
                values.append(self._info_scalar(info, key, default))
        return np.asarray(values, dtype=np.float32)

    def update(self, batch: MAPPOWaypointBatch) -> dict:
        advantages = batch.advantages.copy()
        if self.train_config.get("advantage_norm", True):
            advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
        flat_obs = self._flatten_obs(batch.observations)
        flat_train_actions = self._flatten_time_env(batch.train_actions)
        flat_old_logprobs = self._flatten_time_env(batch.old_logprobs)
        flat_returns = batch.returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_target_connectivity_violation = self._flatten_info_metric(batch.infos, "connectivity_violation", 0.0)
        flat_target_coverage_gain = self._flatten_info_metric(batch.infos, "coverage_gain", 0.0)
        batch_size = flat_train_actions.shape[0]
        epochs = int(self.train_config["epochs"])
        minibatch_size = int(self.train_config["minibatch_size"])
        clip_coef = float(self.train_config["clip_coef"])
        target_kl = float(self.train_config.get("target_kl", 0.0) or 0.0)
        return_targets = batch.returns.reshape(-1)
        value_predictions = batch.values.reshape(-1)
        return_var = float(np.var(return_targets))
        explained_variance = 0.0 if return_var <= 1e-8 else float(1.0 - np.var(return_targets - value_predictions) / return_var)
        stats = {key: 0.0 for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "ratio_mean", "grad_norm", "aux_connectivity_loss", "aux_coverage_loss", "aux_total_loss", "aux_pred_violation_mean", "aux_target_violation_mean", "aux_pred_coverage_gain_mean", "aux_target_coverage_gain_mean"]}
        step_count = 0
        kl_early_stop = 0.0
        use_aux = bool(self.train_config.get("use_connectivity_auxiliary_loss", False))
        aux_connectivity_coef = float(self.train_config.get("aux_connectivity_coef", 0.05))
        aux_coverage_coef = float(self.train_config.get("aux_coverage_coef", 0.02))
        for _ in range(epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb = indices[start : start + minibatch_size]
                obs_t = {key: torch.as_tensor(value[mb], dtype=torch.float32, device=self.device) for key, value in flat_obs.items()}
                train_action_dtype = torch.long if np.issubdtype(flat_train_actions.dtype, np.integer) else torch.float32
                actions_t = torch.as_tensor(flat_train_actions[mb], dtype=train_action_dtype, device=self.device)
                old_logprob_t = torch.as_tensor(flat_old_logprobs[mb], dtype=torch.float32, device=self.device)
                returns_t = torch.as_tensor(flat_returns[mb], dtype=torch.float32, device=self.device)
                adv_t = torch.as_tensor(flat_advantages[mb], dtype=torch.float32, device=self.device)
                target_violation_t = torch.as_tensor(flat_target_connectivity_violation[mb], dtype=torch.float32, device=self.device)
                target_coverage_t = torch.as_tensor(flat_target_coverage_gain[mb], dtype=torch.float32, device=self.device)
                policy_output = self.policy.get_action_and_value(obs_t, actions_t)
                new_logprob = policy_output.logprob
                entropy = policy_output.entropy
                value = policy_output.value
                agent_mask = obs_t.get("agent_mask", torch.ones_like(new_logprob)).bool()
                logratio = new_logprob - old_logprob_t
                ratio = torch.exp(logratio)
                team_adv = adv_t.unsqueeze(-1).expand_as(ratio)
                pg_loss1 = -team_adv * ratio
                pg_loss2 = -team_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = self._masked_mean(torch.maximum(pg_loss1, pg_loss2), agent_mask)
                value_loss = 0.5 * ((value - returns_t) ** 2).mean()
                entropy_loss = self._masked_mean(entropy, agent_mask)
                aux_connectivity_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                aux_coverage_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                aux_total_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                aux_pred_violation = policy_output.aux.get("aux_pred_connectivity_violation") if policy_output.aux else None
                aux_pred_coverage = policy_output.aux.get("aux_pred_coverage_gain") if policy_output.aux else None
                if use_aux and aux_pred_violation is not None:
                    aux_connectivity_loss = torch.mean((aux_pred_violation.float() - target_violation_t) ** 2)
                    aux_total_loss = aux_total_loss + aux_connectivity_coef * aux_connectivity_loss
                if use_aux and aux_pred_coverage is not None:
                    aux_coverage_loss = torch.mean((aux_pred_coverage.float() - target_coverage_t) ** 2)
                    aux_total_loss = aux_total_loss + aux_coverage_coef * aux_coverage_loss
                loss = policy_loss + float(self.train_config["vf_coef"]) * value_loss - float(self.train_config["ent_coef"]) * entropy_loss + aux_total_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.train_config["max_grad_norm"]))
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = self._masked_mean((ratio - 1.0) - logratio, agent_mask)
                    clip_frac = self._masked_mean(((ratio - 1.0).abs() > clip_coef).float(), agent_mask)
                    ratio_mean = self._masked_mean(ratio, agent_mask)
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_loss.item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["clip_frac"] += float(clip_frac.item())
                stats["ratio_mean"] += float(ratio_mean.item())
                stats["grad_norm"] += float(grad_norm.item())
                stats["aux_connectivity_loss"] += float(aux_connectivity_loss.item())
                stats["aux_coverage_loss"] += float(aux_coverage_loss.item())
                stats["aux_total_loss"] += float(aux_total_loss.item())
                stats["aux_target_violation_mean"] += float(target_violation_t.mean().item())
                stats["aux_target_coverage_gain_mean"] += float(target_coverage_t.mean().item())
                if aux_pred_violation is not None:
                    stats["aux_pred_violation_mean"] += float(aux_pred_violation.detach().float().mean().item())
                if aux_pred_coverage is not None:
                    stats["aux_pred_coverage_gain_mean"] += float(aux_pred_coverage.detach().float().mean().item())
                step_count += 1
                if target_kl > 0.0 and float(approx_kl.item()) > target_kl:
                    kl_early_stop = 1.0
                    break
            if kl_early_stop:
                break
        result = {key: value / max(step_count, 1) for key, value in stats.items()}
        result.update(
            {
                "explained_variance": explained_variance,
                "kl_early_stop": kl_early_stop,
                "advantage_mean": float(batch.advantages.mean()),
                "advantage_std": float(batch.advantages.std()),
                "return_mean": float(batch.returns.mean()),
                "return_std": float(batch.returns.std()),
            }
        )
        return result

    def evaluate(
        self,
        env_config: dict,
        task_names: list[str],
        episodes: int,
        output_dir: Path | None = None,
        update_idx: int = 0,
        headless: bool = True,
        record_episodes: int = 0,
        record_format: str = "gif",
        record_fps: int = 8,
        randomization_mode: str = "inherit",
    ) -> dict:
        from utils.waypoint_evaluation import (
            env_config_for_randomization_mode,
            evaluate_waypoint_policy_per_task,
            resolve_eval_randomization_mode,
        )

        mode = resolve_eval_randomization_mode({"randomization_mode": randomization_mode})
        modes = ["fixed", "random"] if mode == "both" else [mode]
        results: dict[str, dict] = {}
        for current_mode in modes:
            current_config = env_config_for_randomization_mode(env_config, current_mode)
            results[current_mode] = self._evaluate_single_mode(
                current_config,
                task_names,
                episodes,
                output_dir=output_dir,
                update_idx=update_idx,
                headless=headless,
                record_episodes=record_episodes,
                record_format=record_format,
                record_fps=record_fps,
                randomization_mode=current_mode,
                evaluate_fn=evaluate_waypoint_policy_per_task,
            )

        if mode == "inherit":
            return results["inherit"]

        combined: dict[str, float | str] = {"eval_randomization_mode": mode}
        for current_mode, metrics in results.items():
            for key, value in metrics.items():
                if key.startswith("eval_"):
                    combined[f"eval_{current_mode}_{key[5:]}"] = value

        # Keep legacy eval_* fields usable by checkpoint selection and tuning scripts.
        primary_mode = "random" if "random" in results else "fixed"
        combined.update(results[primary_mode])
        if mode == "both":
            fixed = results["fixed"]
            random = results["random"]
            combined["eval_generalization_gap"] = float(
                fixed.get("eval_success_rate", 0.0) - random.get("eval_success_rate", 0.0)
            )
            combined["eval_reward_generalization_gap"] = float(
                fixed.get("eval_reward", 0.0) - random.get("eval_reward", 0.0)
            )
        return combined

    def _evaluate_single_mode(
        self,
        env_config: dict,
        task_names: list[str],
        episodes: int,
        output_dir: Path | None,
        update_idx: int,
        headless: bool,
        record_episodes: int,
        record_format: str,
        record_fps: int,
        randomization_mode: str,
        evaluate_fn,
    ) -> dict:
        record_dir = None
        if output_dir is not None and int(record_episodes) > 0:
            record_dir = output_dir / "media" / f"eval_{randomization_mode}" / f"eval_{update_idx:04d}"
        records, task_summaries, overall = evaluate_fn(
            env_config,
            self.policy,
            task_names,
            episodes,
            self.device,
            headless=headless,
            record_dir=record_dir,
            record_episodes=record_episodes,
            record_format=record_format,
            record_fps=record_fps,
            record_prefix=f"update_{update_idx:04d}_{randomization_mode}",
        )
        if output_dir is not None and records:
            eval_records = [
                dict(record, update=int(update_idx), eval_randomization_mode=randomization_mode)
                for record in records
            ]
            _append_metrics_csv(eval_records, Path(output_dir) / "eval_metrics.csv")
        result = {
            "eval_reward": float(overall.get("return_mean", 0.0)),
            "eval_success_rate": float(overall.get("success_rate_mean", 0.0)),
            "eval_path_length": float(overall.get("path_length_mean", 0.0)),
            "eval_collision_rate": float(overall.get("collision_rate_mean", 0.0)),
            "eval_action_validity_rate": float(overall.get("action_validity_rate_mean", 1.0)),
            "eval_mean_speed": float(overall.get("mean_speed_mean", 0.0)),
            "eval_max_speed_observed": float(overall.get("max_speed_observed_mean", 0.0)),
            "eval_mean_control_norm": float(overall.get("mean_control_norm_mean", 0.0)),
            "eval_mean_distance_to_waypoint_m": float(overall.get("mean_distance_to_waypoint_m_mean", 0.0)),
            "eval_arrival_rate": float(overall.get("arrival_rate_mean", 0.0)),
            "eval_command_update_rate": float(overall.get("command_update_rate_mean", 1.0)),
            "eval_command_reject_rate": float(overall.get("command_reject_rate_mean", 0.0)),
            "eval_mean_desired_speed_mps": float(overall.get("mean_desired_speed_mps_mean", 0.0)),
            "eval_collision_count": float(overall.get("collision_count_mean", 0.0)),
            "eval_no_fly_violation_count": float(overall.get("no_fly_violation_count_mean", 0.0)),
            "eval_geofence_violation_count": float(overall.get("geofence_violation_count_mean", 0.0)),
            "eval_uses_real_mpe_core": float(overall.get("uses_real_mpe_core_mean", 0.0)),
            "eval_mpe_world_step_calls": float(overall.get("mpe_world_step_calls_mean", 0.0)),
        }
        if "mpe_source" in overall:
            result["eval_mpe_source"] = str(overall["mpe_source"])
        for task_name, summary in task_summaries.items():
            safe = task_name.replace("-", "_")
            for key, value in summary.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    result[f"eval_{safe}_{key}"] = float(value)
        if record_dir is not None:
            result["eval_media_dir"] = str(record_dir)
        return result

    def train(
        self,
        output_dir: str | Path,
        env_config: dict,
        task_names: list[str],
        eval_episodes: int,
        headless: bool = True,
        record_eval_episodes: int = 0,
        record_format: str = "gif",
        record_fps: int = 8,
        record_interval: int = 1,
        eval_randomization_mode: str = "inherit",
        log_callback: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        output_dir = Path(output_dir)
        checkpoints = output_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        total_updates = int(self.train_config["total_updates"])
        eval_interval = int(self.train_config.get("eval_interval", total_updates))
        best_success = float("-inf")
        best_reward = float("-inf")
        history = []
        for update_idx in range(1, total_updates + 1):
            self._set_lr(update_idx, total_updates)
            batch, rollout_stats = self.collect_rollout()
            train_stats = self.update(batch)
            record = {"update": int(update_idx), **rollout_stats, **train_stats}
            if update_idx % eval_interval == 0 or update_idx == total_updates:
                media_episodes = int(record_eval_episodes) if (update_idx // max(eval_interval, 1)) % max(record_interval, 1) == 0 else 0
                record.update(
                    self.evaluate(
                        env_config,
                        task_names,
                        eval_episodes,
                        output_dir=output_dir,
                        update_idx=update_idx,
                        headless=headless,
                        record_episodes=media_episodes,
                        record_format=record_format,
                        record_fps=record_fps,
                        randomization_mode=eval_randomization_mode,
                    )
                )
                ckpt = checkpoints / f"checkpoint_{update_idx:04d}.pt"
                torch.save({"model_state_dict": self.policy.state_dict(), "train_config": self.train_config}, ckpt)
                record["checkpoint_path"] = str(ckpt)
                success = float(record.get("eval_success_rate", 0.0))
                reward = float(record.get("eval_reward", 0.0))
                if success > best_success or (success == best_success and reward > best_reward):
                    best_success = success
                    best_reward = reward
                    best = checkpoints / "checkpoint_best_eval.pt"
                    torch.save({"model_state_dict": self.policy.state_dict(), "train_config": self.train_config}, best)
                    record["best_checkpoint_path"] = str(best)
                    (output_dir / "best_eval_summary.json").write_text(
                        json.dumps({"update": update_idx, "eval_success_rate": success, "eval_reward": reward, "checkpoint_path": str(best)}, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
            history.append(record)
            if log_callback is not None:
                log_callback(record)
        return history


def write_metrics_csv(records: list[dict], path: str | Path) -> None:
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for record in records for key in record})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _append_metrics_csv(records: list[dict], path: str | Path) -> None:
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    existing_keys: list[str] = []
    if path.exists():
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_keys = list(reader.fieldnames or [])
            existing_rows = list(reader)
    keys = sorted(set(existing_keys) | {key for record in records for key in record})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        writer.writerows(records)

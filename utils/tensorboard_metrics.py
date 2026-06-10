from __future__ import annotations

from typing import Any

import numpy as np


CORE_EXACT_KEYS = {
    "mean_team_reward",
    "mean_rollout_reward",
    "eval_reward",
    "eval_success_rate",
    "eval_action_validity_rate",
    "eval_generalization_gap",
    "eval_reward_generalization_gap",
    "action_validity_rate",
    "delta_clip_fraction",
    "delta_raw_norm_mean",
    "clipped_delta_norm_mean",
    "policy_std_mean",
    "log_std_mean",
    "approx_kl",
    "clip_frac",
    "entropy",
    "policy_loss",
    "value_loss",
    "grad_norm",
    "explained_variance",
    "advantage_std",
    "mean_speed",
    "mean_control_norm",
    "collision_count",
    "no_fly_violation_count",
    "geofence_violation_count",
    "connectivity_violation_rate",
    "coverage_ratio",
    "new_coverage_ratio",
    "overlap_ratio",
    "uncovered_approach_gain",
    "repeat_footprint_ratio",
    "connected_new_coverage_gain",
    "disconnected_new_coverage_gain",
    "connected_new_coverage_ratio",
    "disconnected_new_coverage_ratio",
    "connected_coverage_reward",
    "disconnected_coverage_penalty",
    "searched_probability_mass",
    "detection_rate",
    "remaining_belief_mass",
    "weighted_poi_completion",
    "visited_poi_ratio",
    "effective_explored_radius",
    "connected_to_base_ratio",
    "connected_angular_spread",
    "relaxed_success",
    "relaxed_success_coverage",
    "relaxed_success_radius",
    "relaxed_max_violation_rate",
    "connectivity_margin_mean",
    "connectivity_margin_min",
    "connectivity_margin_penalty",
    "action_disconnect_risk",
    "action_disconnect_risk_mean",
    "predicted_connected_to_base_ratio",
    "predicted_connectivity_violation",
    "unsafe_action_ratio",
    "connectivity_cost",
    "lambda_connectivity_cost",
    "connectivity_cost_target",
    "connectivity_cost_error",
    "lagrangian_penalty",
    "raw_task_reward",
    "penalized_reward",
    "aux_connectivity_loss",
    "aux_coverage_loss",
    "aux_total_loss",
    "aux_pred_violation_mean",
    "aux_target_violation_mean",
    "aux_pred_coverage_gain_mean",
    "aux_target_coverage_gain_mean",
    "connectivity_action_filter_enabled",
    "connectivity_filter_replacement_count",
    "target_coverage_ratio",
    "interception_success",
    "return",
    "success",
    "success_rate",
    "path_length",
}

MINIMAL_EXACT_KEYS = {
    "mean_team_reward",
    "mean_rollout_reward",
    "eval_reward",
    "eval_success_rate",
    "eval_action_validity_rate",
    "eval_generalization_gap",
    "eval_reward_generalization_gap",
    "action_validity_rate",
    "delta_clip_fraction",
    "delta_raw_norm_mean",
    "clipped_delta_norm_mean",
    "policy_std_mean",
    "log_std_mean",
    "approx_kl",
    "clip_frac",
    "entropy",
    "policy_loss",
    "value_loss",
    "explained_variance",
    "advantage_std",
    "mean_speed",
    "mean_control_norm",
    "collision_count",
    "connectivity_violation_rate",
}

MINIMAL_SUBSTRINGS = (
    "coverage_ratio_mean",
    "searched_probability_mass_mean",
    "weighted_poi_completion_mean",
    "effective_explored_radius_mean",
    "connectivity_violation_rate_mean",
)

CORE_SUBSTRINGS = (
    "coverage_ratio_mean",
    "new_coverage_ratio_mean",
    "overlap_ratio_mean",
    "uncovered_approach_gain_mean",
    "repeat_footprint_ratio_mean",
    "connected_new_coverage_gain_mean",
    "disconnected_new_coverage_gain_mean",
    "connected_new_coverage_ratio_mean",
    "disconnected_new_coverage_ratio_mean",
    "connected_coverage_reward_mean",
    "disconnected_coverage_penalty_mean",
    "searched_probability_mass_mean",
    "detection_rate_mean",
    "remaining_belief_mass_mean",
    "weighted_poi_completion_mean",
    "visited_poi_ratio_mean",
    "effective_explored_radius_mean",
    "connected_to_base_ratio_mean",
    "relaxed_success_mean",
    "relaxed_success_coverage_mean",
    "relaxed_success_radius_mean",
    "relaxed_max_violation_rate_mean",
    "connectivity_violation_rate_mean",
    "connectivity_margin_mean",
    "connectivity_margin_min_mean",
    "connectivity_margin_penalty_mean",
    "action_disconnect_risk_mean",
    "predicted_connected_to_base_ratio_mean",
    "predicted_connectivity_violation_mean",
    "unsafe_action_ratio_mean",
    "connectivity_action_filter_enabled_mean",
    "connectivity_filter_replacement_count_mean",
    "connected_angular_spread_mean",
    "target_coverage_ratio_mean",
    "interception_success_mean",
)


def is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _normalize_eval_randomization_key(key: str) -> str:
    for prefix in ("eval_fixed_", "eval_random_"):
        if key.startswith(prefix):
            return f"eval_{key[len(prefix):]}"
    return key


def should_log_tensorboard_metric(key: str, value: Any, mode: str = "core") -> bool:
    if key == "update" or not is_scalar(value):
        return False
    if mode == "all":
        return True
    normalized = _normalize_eval_randomization_key(key)
    if mode == "minimal":
        return normalized in MINIMAL_EXACT_KEYS or any(fragment in normalized for fragment in MINIMAL_SUBSTRINGS)
    if normalized in CORE_EXACT_KEYS:
        return True
    return any(fragment in normalized for fragment in CORE_SUBSTRINGS)


__all__ = ["should_log_tensorboard_metric"]

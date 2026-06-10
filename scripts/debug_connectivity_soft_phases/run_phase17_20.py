from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.experiment_layout import make_debug_output_dir, make_run_name, validate_phase_name, validate_runtime


TASK = "connectivity_expansion"
BASE_POLICY_CONFIG = "configs/policy/direct_specialists/direct_connectivity_expansion.yaml"
BASE_ENV_CONFIG = "configs/env/waypoint_missions.yaml"


@dataclass(frozen=True)
class PhaseSpec:
    key: str
    name: str
    title: str
    mechanism: str
    policy_overrides: dict[str, Any]
    env_overrides: dict[str, Any]
    solved_text: str
    failure_text: str
    next_text: str


COMMON_POLICY_OVERRIDES: dict[str, Any] = {
    "policy_class": "direct_waypoint",
    "max_delta": 0.14,
    "max_waypoint_distance": 0.14,
    "log_std_init": -2.5,
    "log_std_min": -3.5,
    "log_std_max": -1.1,
    "ent_coef": 0.001,
    "learning_rate": 0.00018,
    "lr_schedule": "constant",
    "target_kl": 0.03,
    "vf_coef": 0.75,
    "use_lagrangian_connectivity_cost": False,
    "connectivity_cost_target": 0.05,
    "lagrangian_lr": 0.01,
    "lambda_cost_init": 1.0,
    "lambda_cost_min": 0.0,
    "lambda_cost_max": 50.0,
    "use_connectivity_auxiliary_loss": False,
    "aux_connectivity_coef": 0.05,
    "aux_coverage_coef": 0.02,
    "aux_targets": ["connectivity_violation", "coverage_gain"],
    "tensorboard_metric_mode": "core",
}

COMMON_ENV_OVERRIDES: dict[str, Any] = {
    "w_expand": 8.0,
    "w_coverage": 360.0,
    "w_disconnect": 12.0,
    "w_uncovered_approach": 1200.0,
    "uncovered_approach_scale": 0.26,
    "w_repeat_footprint": 0.2,
    "w_connected_radius_hold": 0.15,
    "w_connected_spread": 0.0,
    "w_relay": 1.5,
    "w_connectivity_margin": 3.0,
    "connectivity_margin_threshold": 0.05,
    "w_action_disconnect_risk": 0.0,
    "w_connected_new_coverage": 0.0,
    "w_disconnected_new_coverage": 0.0,
    "w_distance": 0.03,
    "w_safety": 2.0,
    "cost_signal": "connectivity_violation_rate",
}

PHASES: dict[str, PhaseSpec] = {
    "phase17": PhaseSpec(
        key="phase17",
        name="phase17_connectivity_margin_penalty",
        title="connectivity disconnect-margin penalty",
        mechanism="Dense reward penalty for current disconnects and pre-disconnect margin.",
        policy_overrides={},
        env_overrides={},
        solved_text="Tests whether dense margin reward can lower disconnects while executing raw direct waypoint actions.",
        failure_text="If violation does not drop, inspect margin geometry, margin threshold, and w_connectivity_margin before adding action risk.",
        next_text="Phase18 is needed if margin improves connectivity but coverage stalls or risky actions remain common.",
    ),
    "phase18": PhaseSpec(
        key="phase18",
        name="phase18_connectivity_action_risk_penalty",
        title="connectivity action-risk soft penalty",
        mechanism="Adds predicted next-step disconnect risk as reward penalty only.",
        policy_overrides={},
        env_overrides={"w_action_disconnect_risk": 4.0, "action_risk_use_predicted_graph": True},
        solved_text="Tests whether the actor reduces waypoint commands that are predicted to disconnect the team.",
        failure_text="If risk stays high, the actor likely lacks a useful connectivity representation rather than needing action replacement.",
        next_text="Phase19 is needed to tune connectivity pressure automatically instead of only hand-setting reward weights.",
    ),
    "phase19": PhaseSpec(
        key="phase19",
        name="phase19_connectivity_lagrangian_cost",
        title="connectivity Lagrangian / CMDP cost",
        mechanism="Adds rollout-level Lagrangian penalty lambda_connectivity_cost * connectivity_cost.",
        policy_overrides={
            "use_lagrangian_connectivity_cost": True,
            "connectivity_cost_target": 0.05,
            "lagrangian_lr": 0.01,
            "lambda_cost_init": 1.0,
            "lambda_cost_min": 0.0,
            "lambda_cost_max": 50.0,
            "cost_signal": "combined",
        },
        env_overrides={
            "w_action_disconnect_risk": 4.0,
            "action_risk_use_predicted_graph": True,
            "cost_signal": "combined",
        },
        solved_text="Tests whether lambda rises when connectivity cost exceeds target and stabilizes violation without killing coverage.",
        failure_text="If lambda saturates, the constraint is too hard for the current curriculum. If lambda stays near zero with high violation, cost logging is broken.",
        next_text="Phase20 is needed if violation is controlled but coverage still does not improve from learned representations.",
    ),
    "phase20": PhaseSpec(
        key="phase20",
        name="phase20_connectivity_auxiliary_prediction",
        title="connectivity auxiliary prediction / representation",
        mechanism="Adds auxiliary predictions for connectivity violation and coverage gain; no action replacement.",
        policy_overrides={
            "use_lagrangian_connectivity_cost": True,
            "connectivity_cost_target": 0.05,
            "lagrangian_lr": 0.01,
            "lambda_cost_init": 1.0,
            "lambda_cost_min": 0.0,
            "lambda_cost_max": 50.0,
            "cost_signal": "combined",
            "use_connectivity_auxiliary_loss": True,
            "aux_connectivity_coef": 0.05,
            "aux_coverage_coef": 0.02,
            "aux_targets": ["connectivity_violation", "coverage_gain"],
        },
        env_overrides={
            "w_action_disconnect_risk": 4.0,
            "action_risk_use_predicted_graph": True,
            "cost_signal": "combined",
        },
        solved_text="Tests whether auxiliary risk/coverage targets make the shared representation more useful for connected coverage.",
        failure_text="If auxiliary losses fall without behavior improvement, the auxiliary signal is not reaching actor-useful features.",
        next_text="A follow-up would connect auxiliary features earlier in the actor/shared encoder or add a curriculum, not a hard filter.",
    ),
    "phase21": PhaseSpec(
        key="phase21",
        name="phase21_connected_coverage_reward",
        title="connectivity connected-coverage reward",
        mechanism="Rewards new coverage primarily when produced by UAVs connected to base and penalizes disconnected new coverage.",
        policy_overrides={
            "max_delta": 0.16,
            "max_waypoint_distance": 0.16,
            "log_std_init": -2.35,
            "log_std_min": -3.4,
            "log_std_max": -1.0,
            "ent_coef": 0.0015,
            "learning_rate": 0.0002,
        },
        env_overrides={
            "w_expand": 12.0,
            "w_coverage": 160.0,
            "w_connected_new_coverage": 760.0,
            "w_disconnected_new_coverage": 420.0,
            "w_disconnect": 8.0,
            "w_uncovered_approach": 1500.0,
            "w_repeat_footprint": 0.6,
            "w_connected_radius_hold": 0.08,
            "w_relay": 2.0,
            "w_connectivity_margin": 1.2,
            "connectivity_margin_threshold": 0.04,
            "w_action_disconnect_risk": 1.5,
            "action_risk_use_predicted_graph": True,
            "w_distance": 0.02,
        },
        solved_text="Tests whether binding coverage reward to base-connected UAVs resolves the coverage-vs-connectivity tradeoff seen in phase17 trial02.",
        failure_text="If coverage remains high but violation stays high, the disconnected coverage penalty is too weak or connectivity is under-observed. If coverage collapses, connected coverage reward is not reachable under current exploration.",
        next_text="If this helps but does not converge, sweep connected/disconnected coverage weights or add nearest-base-connected-component observation features.",
    ),
    "phase22": PhaseSpec(
        key="phase22",
        name="phase22_comm_graph_encoder_aux",
        title="connectivity comm-graph encoder with auxiliary loss",
        mechanism="Uses the existing comm_adjacency observation inside the shared Direct waypoint policy and keeps connected-coverage rewards.",
        policy_overrides={
            "max_delta": 0.16,
            "max_waypoint_distance": 0.16,
            "log_std_init": -2.35,
            "log_std_min": -3.4,
            "log_std_max": -1.0,
            "ent_coef": 0.0015,
            "learning_rate": 0.00018,
            "use_comm_graph_encoder": True,
            "use_connectivity_auxiliary_loss": True,
            "aux_connectivity_coef": 0.04,
            "aux_coverage_coef": 0.015,
        },
        env_overrides={
            "w_expand": 12.0,
            "w_coverage": 220.0,
            "w_connected_new_coverage": 850.0,
            "w_disconnected_new_coverage": 360.0,
            "w_disconnect": 8.5,
            "w_uncovered_approach": 1600.0,
            "w_repeat_footprint": 0.55,
            "w_connected_radius_hold": 0.10,
            "w_connected_spread": 0.25,
            "w_relay": 2.2,
            "w_connectivity_margin": 1.1,
            "connectivity_margin_threshold": 0.04,
            "w_action_disconnect_risk": 1.2,
            "action_risk_use_predicted_graph": True,
            "w_distance": 0.02,
            "relaxed_success_coverage": 0.45,
            "relaxed_success_radius": 0.35,
            "relaxed_max_violation_rate": 0.20,
        },
        solved_text="Tests whether topology-aware encoding over the existing communication graph helps Direct MAPPO learn connected coverage without changing the observation schema.",
        failure_text="If relaxed success improves but target success does not, continue with curriculum. If both stay flat, reward/curriculum rather than topology encoding is the bottleneck.",
        next_text="If phase22 is promising, run a curriculum phase that gradually tightens relaxed thresholds toward target thresholds.",
    ),
    "phase23": PhaseSpec(
        key="phase23",
        name="phase23_relaxed_success_graph_reward",
        title="connectivity relaxed-success graph reward",
        mechanism="Uses the existing comm graph encoder and phase21 connected-coverage reward, but relaxes debug success thresholds by about 0.1.",
        policy_overrides={
            "max_delta": 0.17,
            "max_waypoint_distance": 0.17,
            "log_std_init": -2.25,
            "log_std_min": -3.4,
            "log_std_max": -0.95,
            "ent_coef": 0.002,
            "learning_rate": 0.0002,
            "use_comm_graph_encoder": True,
            "use_connectivity_auxiliary_loss": True,
            "aux_connectivity_coef": 0.04,
            "aux_coverage_coef": 0.015,
        },
        env_overrides={
            "success_coverage": 0.45,
            "success_radius": 0.35,
            "max_violation_rate": 0.20,
            "relaxed_success_coverage": 0.45,
            "relaxed_success_radius": 0.35,
            "relaxed_max_violation_rate": 0.20,
            "w_expand": 14.0,
            "w_coverage": 320.0,
            "w_connected_new_coverage": 950.0,
            "w_disconnected_new_coverage": 300.0,
            "w_disconnect": 7.5,
            "w_uncovered_approach": 1800.0,
            "w_repeat_footprint": 0.50,
            "w_connected_radius_hold": 0.10,
            "w_connected_spread": 0.30,
            "w_relay": 2.2,
            "w_connectivity_margin": 0.90,
            "connectivity_margin_threshold": 0.04,
            "w_action_disconnect_risk": 0.9,
            "action_risk_use_predicted_graph": True,
            "w_distance": 0.02,
        },
        solved_text="Tests whether Direct MAPPO can reliably learn a relaxed connected-expansion policy without hard filters or task-specific observation additions.",
        failure_text="If this phase still cannot reach nonzero success, the bottleneck is not just strict success thresholds.",
        next_text="If it converges, tighten thresholds in a later phase toward coverage 0.55 / radius 0.45 / violation 0.10.",
    ),
    "phase24": PhaseSpec(
        key="phase24",
        name="phase24_connected_only_reward",
        title="connectivity connected-only coverage reward",
        mechanism=(
            "Keeps the phase23 relaxed debug success thresholds, but makes coverage reward mostly depend on "
            "base-connected new coverage instead of total new coverage."
        ),
        policy_overrides={
            "max_delta": 0.165,
            "max_waypoint_distance": 0.165,
            "log_std_init": -2.35,
            "log_std_min": -3.4,
            "log_std_max": -1.0,
            "ent_coef": 0.0015,
            "learning_rate": 0.00018,
            "use_comm_graph_encoder": True,
            "use_connectivity_auxiliary_loss": True,
            "aux_connectivity_coef": 0.04,
            "aux_coverage_coef": 0.015,
        },
        env_overrides={
            "success_coverage": 0.45,
            "success_radius": 0.35,
            "max_violation_rate": 0.20,
            "relaxed_success_coverage": 0.45,
            "relaxed_success_radius": 0.35,
            "relaxed_max_violation_rate": 0.20,
            "w_expand": 10.0,
            "w_coverage": 40.0,
            "w_connected_new_coverage": 1650.0,
            "w_disconnected_new_coverage": 900.0,
            "w_disconnect": 11.0,
            "w_uncovered_approach": 650.0,
            "w_repeat_footprint": 0.35,
            "w_connected_radius_hold": 0.08,
            "w_connected_spread": 0.25,
            "w_relay": 2.5,
            "w_connectivity_margin": 1.6,
            "connectivity_margin_threshold": 0.045,
            "w_action_disconnect_risk": 1.5,
            "action_risk_use_predicted_graph": True,
            "w_distance": 0.018,
        },
        solved_text="Tests whether removing the disconnected-total-coverage shortcut makes Direct MAPPO learn connected coverage.",
        failure_text=(
            "If coverage collapses, the connected coverage reward is not reachable under current exploration. "
            "If violation stays high, disconnected coverage is still too cheap or action risk is underweighted."
        ),
        next_text="If phase24 becomes safe but under-covers, use a curriculum or schedule connected coverage weights rather than adding observation tricks.",
    ),
    "phase25": PhaseSpec(
        key="phase25",
        name="phase25_tight_connected_curriculum",
        title="connectivity tighter connected-coverage curriculum",
        mechanism=(
            "Continues from the phase24 connected-only reward solution and tightens debug success "
            "toward the formal target without using a connectivity action filter."
        ),
        policy_overrides={
            "max_delta": 0.17,
            "max_waypoint_distance": 0.17,
            "log_std_init": -2.45,
            "log_std_min": -3.5,
            "log_std_max": -1.05,
            "ent_coef": 0.0012,
            "learning_rate": 0.00016,
            "target_kl": 0.025,
            "use_comm_graph_encoder": True,
            "use_connectivity_auxiliary_loss": True,
            "aux_connectivity_coef": 0.04,
            "aux_coverage_coef": 0.015,
        },
        env_overrides={
            "success_coverage": 0.50,
            "success_radius": 0.40,
            "max_violation_rate": 0.15,
            "relaxed_success_coverage": 0.50,
            "relaxed_success_radius": 0.40,
            "relaxed_max_violation_rate": 0.15,
            "w_expand": 14.0,
            "w_coverage": 65.0,
            "w_connected_new_coverage": 2300.0,
            "w_disconnected_new_coverage": 1150.0,
            "w_disconnect": 13.5,
            "w_uncovered_approach": 950.0,
            "w_repeat_footprint": 1.2,
            "w_connected_radius_hold": 0.08,
            "w_connected_spread": 0.30,
            "w_relay": 2.8,
            "w_connectivity_margin": 1.9,
            "connectivity_margin_threshold": 0.045,
            "w_action_disconnect_risk": 1.9,
            "action_risk_use_predicted_graph": True,
            "w_distance": 0.018,
        },
        solved_text=(
            "Tests whether the phase24 policy can be pushed from relaxed success to a tighter "
            "connected-coverage target through reward curriculum rather than observation or action postprocess tricks."
        ),
        failure_text=(
            "If violation stays low but coverage remains below 0.50, the next change should increase connected frontier progress "
            "or use a threshold schedule. If violation rises, lower max_delta or increase action-risk/disconnect pressure."
        ),
        next_text="If phase25 converges, phase26 should restore the formal 0.55/0.45/0.10 target and verify multi-seed robustness.",
    ),
    "phase26": PhaseSpec(
        key="phase26",
        name="phase26_formal_connected_target",
        title="connectivity formal connected target",
        mechanism=(
            "Starts from the phase25 tightened checkpoint and restores the formal proxy target "
            "coverage>=0.55, radius>=0.45, violation<=0.10."
        ),
        policy_overrides={
            "max_delta": 0.17,
            "max_waypoint_distance": 0.17,
            "log_std_init": -2.55,
            "log_std_min": -3.6,
            "log_std_max": -1.10,
            "ent_coef": 0.001,
            "learning_rate": 0.00012,
            "target_kl": 0.022,
            "use_comm_graph_encoder": True,
            "use_connectivity_auxiliary_loss": True,
            "aux_connectivity_coef": 0.04,
            "aux_coverage_coef": 0.015,
        },
        env_overrides={
            "success_coverage": 0.55,
            "success_radius": 0.45,
            "max_violation_rate": 0.10,
            "relaxed_success_coverage": 0.55,
            "relaxed_success_radius": 0.45,
            "relaxed_max_violation_rate": 0.10,
            "w_expand": 18.0,
            "w_coverage": 45.0,
            "w_connected_new_coverage": 2750.0,
            "w_disconnected_new_coverage": 1350.0,
            "w_disconnect": 15.0,
            "w_uncovered_approach": 1200.0,
            "w_repeat_footprint": 1.1,
            "w_connected_radius_hold": 0.08,
            "w_connected_spread": 0.30,
            "w_relay": 3.0,
            "w_connectivity_margin": 2.2,
            "connectivity_margin_threshold": 0.045,
            "w_action_disconnect_risk": 2.2,
            "action_risk_use_predicted_graph": True,
            "w_distance": 0.016,
        },
        solved_text="Tests whether the phase25 policy can cross the formal connected-expansion proxy without hard action filters.",
        failure_text=(
            "If coverage remains below 0.55 while violation is safe, the next phase should use a coverage curriculum or increase "
            "connected frontier progress. If violation rises above 0.10, use a stricter safety branch before raising coverage pressure."
        ),
        next_text="If phase26 reaches stable formal success, run repeat seeds. If not, split the failure into coverage-limited vs violation-limited branches.",
    ),
    "phase27": PhaseSpec(
        key="phase27",
        name="phase27_formal_safety_balance",
        title="connectivity formal safety-balanced target",
        mechanism=(
            "Starts from phase26 and keeps the formal target while reducing waypoint aggressiveness and increasing "
            "soft connectivity penalties to move violation below 0.10."
        ),
        policy_overrides={
            "max_delta": 0.16,
            "max_waypoint_distance": 0.16,
            "log_std_init": -2.65,
            "log_std_min": -3.7,
            "log_std_max": -1.20,
            "ent_coef": 0.0008,
            "learning_rate": 0.00010,
            "target_kl": 0.020,
            "use_comm_graph_encoder": True,
            "use_connectivity_auxiliary_loss": True,
            "aux_connectivity_coef": 0.04,
            "aux_coverage_coef": 0.015,
        },
        env_overrides={
            "success_coverage": 0.55,
            "success_radius": 0.45,
            "max_violation_rate": 0.10,
            "relaxed_success_coverage": 0.55,
            "relaxed_success_radius": 0.45,
            "relaxed_max_violation_rate": 0.10,
            "w_expand": 16.0,
            "w_coverage": 40.0,
            "w_connected_new_coverage": 2700.0,
            "w_disconnected_new_coverage": 1500.0,
            "w_disconnect": 19.0,
            "w_uncovered_approach": 1150.0,
            "w_repeat_footprint": 1.2,
            "w_connected_radius_hold": 0.09,
            "w_connected_spread": 0.25,
            "w_relay": 3.2,
            "w_connectivity_margin": 3.0,
            "connectivity_margin_threshold": 0.045,
            "w_action_disconnect_risk": 3.2,
            "action_risk_use_predicted_graph": True,
            "w_distance": 0.016,
        },
        solved_text="Tests whether the formal target can be reached by tightening safety softly after phase26 already crossed coverage/radius.",
        failure_text=(
            "If coverage falls below 0.55 but violation becomes safe, the next branch should re-add coverage pressure gradually. "
            "If violation remains above 0.10, the task likely needs a curriculum schedule rather than a single static reward."
        ),
        next_text="If phase27 passes, keep it running and then repeat seeds. If not, open a scheduled phase28 rather than hard filtering.",
    ),
    "phase28": PhaseSpec(
        key="phase28",
        name="phase28_formal_stability_finetune",
        title="connectivity formal stability fine-tuning",
        mechanism=(
            "Starts from the best phase26 checkpoint and uses lower-learning-rate PPO with reduced exploration "
            "to stabilize the formal-target policy instead of pushing reward weights harder."
        ),
        policy_overrides={
            "max_delta": 0.165,
            "max_waypoint_distance": 0.165,
            "log_std_init": -2.70,
            "log_std_min": -3.8,
            "log_std_max": -1.25,
            "ent_coef": 0.0005,
            "learning_rate": 0.00006,
            "target_kl": 0.015,
            "use_comm_graph_encoder": True,
            "use_connectivity_auxiliary_loss": True,
            "aux_connectivity_coef": 0.04,
            "aux_coverage_coef": 0.015,
        },
        env_overrides={
            "success_coverage": 0.55,
            "success_radius": 0.45,
            "max_violation_rate": 0.10,
            "relaxed_success_coverage": 0.55,
            "relaxed_success_radius": 0.45,
            "relaxed_max_violation_rate": 0.10,
            "w_expand": 15.0,
            "w_coverage": 35.0,
            "w_connected_new_coverage": 2600.0,
            "w_disconnected_new_coverage": 1300.0,
            "w_disconnect": 15.5,
            "w_uncovered_approach": 1100.0,
            "w_repeat_footprint": 1.4,
            "w_connected_radius_hold": 0.08,
            "w_connected_spread": 0.28,
            "w_relay": 3.0,
            "w_connectivity_margin": 2.4,
            "connectivity_margin_threshold": 0.045,
            "w_action_disconnect_risk": 2.4,
            "action_risk_use_predicted_graph": True,
            "w_distance": 0.018,
        },
        solved_text="Tests whether the near-formal phase26 update200 policy can be made stable by reducing PPO update size and exploration.",
        failure_text=(
            "If this still oscillates, the next fix should implement an explicit update/threshold schedule or cost multiplier schedule, "
            "not another static weight increase."
        ),
        next_text="If phase28 stabilizes formal success, launch repeat seeds from its best checkpoint.",
    ),
}


TUNING_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "baseline": {"policy": {}, "env": {}},
    "expand_balanced": {
        "policy": {
            "max_delta": 0.16,
            "max_waypoint_distance": 0.16,
            "log_std_init": -2.35,
            "log_std_min": -3.4,
            "log_std_max": -1.0,
            "ent_coef": 0.0015,
            "learning_rate": 0.0002,
        },
        "env": {
            "w_expand": 12.0,
            "w_coverage": 520.0,
            "w_disconnect": 8.0,
            "w_uncovered_approach": 1600.0,
            "w_repeat_footprint": 0.6,
            "w_connected_radius_hold": 0.08,
            "w_relay": 2.0,
            "w_connectivity_margin": 1.5,
            "connectivity_margin_threshold": 0.04,
            "w_distance": 0.02,
        },
    },
    "connected_expand": {
        "policy": {
            "max_delta": 0.17,
            "max_waypoint_distance": 0.17,
            "log_std_init": -2.25,
            "log_std_min": -3.4,
            "log_std_max": -0.95,
            "ent_coef": 0.0018,
            "learning_rate": 0.0002,
        },
        "env": {
            "w_expand": 14.0,
            "w_coverage": 220.0,
            "w_connected_new_coverage": 1000.0,
            "w_disconnected_new_coverage": 260.0,
            "w_disconnect": 7.0,
            "w_uncovered_approach": 1700.0,
            "w_repeat_footprint": 0.55,
            "w_connected_radius_hold": 0.08,
            "w_relay": 2.0,
            "w_connectivity_margin": 0.9,
            "connectivity_margin_threshold": 0.04,
            "w_action_disconnect_risk": 0.8,
            "w_distance": 0.02,
        },
    },
    "connected_balanced": {
        "policy": {
            "max_delta": 0.165,
            "max_waypoint_distance": 0.165,
            "log_std_init": -2.3,
            "log_std_min": -3.4,
            "log_std_max": -1.0,
            "ent_coef": 0.0016,
            "learning_rate": 0.0002,
        },
        "env": {
            "w_expand": 13.0,
            "w_coverage": 320.0,
            "w_connected_new_coverage": 900.0,
            "w_disconnected_new_coverage": 360.0,
            "w_disconnect": 8.5,
            "w_uncovered_approach": 1800.0,
            "w_repeat_footprint": 0.45,
            "w_connected_radius_hold": 0.12,
            "w_connected_spread": 0.35,
            "w_relay": 2.2,
            "w_connectivity_margin": 1.15,
            "connectivity_margin_threshold": 0.04,
            "w_action_disconnect_risk": 1.2,
            "w_distance": 0.02,
        },
    },
    "graph_expand": {
        "policy": {
            "max_delta": 0.17,
            "max_waypoint_distance": 0.17,
            "log_std_init": -2.25,
            "log_std_min": -3.4,
            "log_std_max": -0.95,
            "ent_coef": 0.002,
            "learning_rate": 0.0002,
            "use_comm_graph_encoder": True,
            "use_connectivity_auxiliary_loss": True,
            "aux_connectivity_coef": 0.04,
            "aux_coverage_coef": 0.015,
        },
        "env": {
            "w_expand": 15.0,
            "w_coverage": 300.0,
            "w_connected_new_coverage": 1050.0,
            "w_disconnected_new_coverage": 280.0,
            "w_disconnect": 7.5,
            "w_uncovered_approach": 1850.0,
            "w_repeat_footprint": 0.50,
            "w_connected_radius_hold": 0.08,
            "w_connected_spread": 0.30,
            "w_relay": 2.0,
            "w_connectivity_margin": 0.85,
            "connectivity_margin_threshold": 0.04,
            "w_action_disconnect_risk": 0.9,
            "w_distance": 0.02,
            "relaxed_success_coverage": 0.45,
            "relaxed_success_radius": 0.35,
            "relaxed_max_violation_rate": 0.20,
        },
    },
    "tight_safety": {
        "policy": {
            "max_delta": 0.16,
            "max_waypoint_distance": 0.16,
            "log_std_init": -2.55,
            "log_std_min": -3.6,
            "log_std_max": -1.15,
            "ent_coef": 0.001,
            "learning_rate": 0.00014,
            "target_kl": 0.022,
        },
        "env": {
            "w_connected_new_coverage": 2200.0,
            "w_disconnected_new_coverage": 1250.0,
            "w_disconnect": 16.0,
            "w_connectivity_margin": 2.4,
            "w_action_disconnect_risk": 2.6,
            "w_expand": 12.0,
            "w_coverage": 55.0,
            "w_repeat_footprint": 1.3,
            "w_distance": 0.018,
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def minute_runtime() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def load_yaml(path: str | Path) -> dict[str, Any]:
    full_path = Path(path)
    if not full_path.is_absolute():
        full_path = ROOT / full_path
    with full_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: dict[str, str], key: str, default: float | None = None) -> float | None:
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        out = float(value)
        return out if out == out else default
    except (TypeError, ValueError):
        return default


def metric(row: dict[str, str], *keys: str, default: float | None = None) -> float | None:
    for key in keys:
        value = finite_float(row, key, None)
        if value is not None:
            return value
    return default


def best_eval_row(rows: list[dict[str, str]]) -> dict[str, str]:
    eval_rows = [row for row in rows if row.get("eval_success_rate") not in ("", None)]
    return max(
        eval_rows,
        key=lambda row: (
            metric(row, "eval_success_rate", default=0.0) or 0.0,
            metric(row, "eval_connectivity_expansion_effective_explored_radius_mean", default=0.0) or 0.0,
            metric(row, "eval_connectivity_expansion_coverage_ratio_mean", default=0.0) or 0.0,
            -(metric(row, "eval_connectivity_expansion_connectivity_violation_rate_mean", default=1.0) or 1.0),
            metric(row, "eval_reward", default=-1.0e9) or -1.0e9,
        ),
        default={},
    )


def trend(rows: list[dict[str, str]], key: str) -> dict[str, float | None]:
    values = [finite_float(row, key, None) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return {"first": None, "last": None, "delta": None, "max": None, "min": None}
    return {
        "first": float(values[0]),
        "last": float(values[-1]),
        "delta": float(values[-1] - values[0]),
        "max": float(max(values)),
        "min": float(min(values)),
    }


def summarize_training(output_dir: Path) -> dict[str, Any]:
    rows = read_csv(output_dir / "training_metrics.csv")
    best = best_eval_row(rows)
    last = rows[-1] if rows else {}
    summary: dict[str, Any] = {
        "rows": len(rows),
        "output_dir": str(output_dir),
        "best_update": metric(best, "update"),
        "best_success": metric(best, "eval_success_rate"),
        "best_radius": metric(best, "eval_connectivity_expansion_effective_explored_radius_mean"),
        "best_coverage": metric(best, "eval_connectivity_expansion_coverage_ratio_mean"),
        "best_violation": metric(best, "eval_connectivity_expansion_connectivity_violation_rate_mean"),
        "best_reward": metric(best, "eval_reward"),
        "best_relaxed_success": metric(best, "eval_connectivity_expansion_relaxed_success_mean"),
        "last_clip": metric(last, "clip_frac", "delta_clip_fraction"),
        "last_valid": metric(last, "action_validity_rate"),
        "last_kl": metric(last, "approx_kl"),
        "last_value_loss": metric(last, "value_loss"),
        "best_connected_to_base_ratio": metric(best, "eval_connectivity_expansion_connected_to_base_ratio_mean"),
        "best_connectivity_margin_min": metric(best, "eval_connectivity_expansion_connectivity_margin_min_mean"),
        "best_connectivity_margin_penalty": metric(best, "eval_connectivity_expansion_connectivity_margin_penalty_mean"),
        "best_action_disconnect_risk": metric(best, "eval_connectivity_expansion_action_disconnect_risk_mean"),
        "best_repeat_footprint": metric(best, "eval_connectivity_expansion_repeat_footprint_ratio_mean"),
        "best_uncovered_approach": metric(best, "eval_connectivity_expansion_uncovered_approach_gain_mean"),
        "best_connected_new_coverage": metric(best, "eval_connectivity_expansion_connected_new_coverage_gain_mean"),
        "best_disconnected_new_coverage": metric(best, "eval_connectivity_expansion_disconnected_new_coverage_gain_mean"),
        "best_connected_new_coverage_ratio": metric(best, "eval_connectivity_expansion_connected_new_coverage_ratio_mean"),
        "best_disconnected_new_coverage_ratio": metric(best, "eval_connectivity_expansion_disconnected_new_coverage_ratio_mean"),
        "connected_new_coverage_trend": trend(rows, "connected_new_coverage_gain_mean"),
        "disconnected_new_coverage_trend": trend(rows, "disconnected_new_coverage_gain_mean"),
        "best_command_reject_rate": metric(best, "eval_command_reject_rate", "command_reject_rate"),
        "best_command_update_rate": metric(best, "eval_command_update_rate", "command_update_rate"),
        "last_filter_enabled": metric(last, "connectivity_action_filter_enabled", "connectivity_action_filter_enabled_mean"),
        "last_filter_replacements": metric(last, "connectivity_filter_replacement_count", "connectivity_filter_replacement_count_mean"),
        "best_eval_filter_enabled": metric(best, "eval_connectivity_expansion_connectivity_action_filter_enabled_mean"),
        "best_eval_filter_replacements": metric(best, "eval_connectivity_expansion_connectivity_filter_replacement_count_mean"),
        "action_risk_trend": trend(rows, "action_disconnect_risk_mean"),
        "violation_trend": trend(rows, "connectivity_violation_rate"),
        "margin_penalty_trend": trend(rows, "connectivity_margin_penalty_mean"),
        "lambda_trend": trend(rows, "lambda_connectivity_cost"),
        "cost_error_trend": trend(rows, "connectivity_cost_error"),
        "aux_loss_trend": trend(rows, "aux_total_loss"),
        "aux_connectivity_loss_trend": trend(rows, "aux_connectivity_loss"),
        "aux_coverage_loss_trend": trend(rows, "aux_coverage_loss"),
        "aux_pred_violation_trend": trend(rows, "aux_pred_violation_mean"),
        "aux_target_violation_trend": trend(rows, "aux_target_violation_mean"),
        "relaxed_success_trend": trend(rows, "relaxed_success_mean"),
    }
    return summary


def make_policy_config(phase: PhaseSpec, args: argparse.Namespace, config_dir: Path) -> Path:
    config = load_yaml(BASE_POLICY_CONFIG)
    config.update(COMMON_POLICY_OVERRIDES)
    config.update(phase.policy_overrides)
    config.update(TUNING_PRESETS[str(args.tuning)]["policy"])
    config.update(
        {
            "num_envs": int(args.num_envs),
            "rollout_steps": int(args.rollout_steps),
            "total_updates": int(args.total_updates),
            "eval_interval": int(args.eval_interval),
            "epochs": int(args.epochs),
            "minibatch_size": int(args.minibatch_size),
        }
    )
    path = config_dir / f"policy_{phase.key}_{TASK}_trial{int(args.trial):02d}.yaml"
    write_yaml(path, config)
    return path


def merged_policy_overrides(phase: PhaseSpec, args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(COMMON_POLICY_OVERRIDES)
    merged.update(phase.policy_overrides)
    merged.update(TUNING_PRESETS[str(args.tuning)]["policy"])
    return merged


def make_env_config(phase: PhaseSpec, args: argparse.Namespace, config_dir: Path) -> Path:
    config = deepcopy(load_yaml(BASE_ENV_CONFIG))
    policy_values = merged_policy_overrides(phase, args)
    action_distance = float(policy_values.get("max_waypoint_distance", policy_values.get("max_delta", 0.14)))
    config["task_name"] = TASK
    config["task_names"] = [TASK]
    config["task_sampling_probs"] = {TASK: 1.0}
    config["connectivity_action_filter"] = False
    config["connectivity_candidate_filter"] = False
    config["strict_action_validation"] = False
    config["max_waypoint_distance"] = action_distance
    config.setdefault("action_adapter", {})["max_command_distance"] = action_distance
    config.setdefault("domain_randomization", {})["enabled"] = True
    task_cfg = config.setdefault("tasks", {}).setdefault(TASK, {})
    task_cfg.update(COMMON_ENV_OVERRIDES)
    task_cfg.update(phase.env_overrides)
    task_cfg.update(TUNING_PRESETS[str(args.tuning)]["env"])
    path = config_dir / f"env_{phase.key}_{TASK}_trial{int(args.trial):02d}.yaml"
    write_yaml(path, config)
    return path


def build_command(
    phase: PhaseSpec,
    policy_config: Path,
    env_config: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/train_mappo_waypoint.py"),
        "--config",
        str(policy_config),
        "--env-config",
        str(env_config),
        "--eval-config",
        "configs/eval/waypoint_eval.yaml",
        "--tasks",
        TASK,
        "--num_agents",
        "4",
        "--num_envs",
        str(args.num_envs),
        "--rollout_steps",
        str(args.rollout_steps),
        "--total_updates",
        str(args.total_updates),
        "--eval_interval",
        str(args.eval_interval),
        "--eval_episodes",
        str(args.eval_episodes),
        "--record_eval_episodes",
        "1",
        "--record_format",
        "gif",
        "--seed",
        "0",
        "--run_name",
        make_run_name(0, int(args.trial)),
        "--output-dir",
        str(output_dir),
        "--train-randomization",
        "random",
        "--eval-randomization",
        "random",
        "--env_backend",
        str(args.env_backend),
        "--headless",
        "--tensorboard",
        "--phase-name",
        phase.name,
    ]
    if args.init_checkpoint:
        command.extend(["--init-checkpoint", str(args.init_checkpoint)])
    return command


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def phase_specific_lines(phase: PhaseSpec, summary: dict[str, Any] | None) -> list[str]:
    if summary is None:
        return ["- Phase-specific checks will be filled after training finishes."]
    if phase.key == "phase17":
        margin = summary.get("margin_penalty_trend", {})
        return [
            f"- best eval connectivity_margin_penalty: `{format_value(summary.get('best_connectivity_margin_penalty'))}`",
            f"- margin penalty first/last: `{format_value(margin.get('first'))}` -> `{format_value(margin.get('last'))}`",
            "- margin penalty is active when this value is positive on near-disconnect or disconnected states.",
        ]
    if phase.key == "phase18":
        risk = summary.get("action_risk_trend", {})
        return [
            f"- best eval action_disconnect_risk: `{format_value(summary.get('best_action_disconnect_risk'))}`",
            f"- action_disconnect_risk first/last: `{format_value(risk.get('first'))}` -> `{format_value(risk.get('last'))}`",
            f"- action_disconnect_risk delta: `{format_value(risk.get('delta'))}`",
        ]
    if phase.key == "phase19":
        lam = summary.get("lambda_trend", {})
        err = summary.get("cost_error_trend", {})
        return [
            f"- lambda_connectivity_cost first/last/max: `{format_value(lam.get('first'))}` / `{format_value(lam.get('last'))}` / `{format_value(lam.get('max'))}`",
            f"- connectivity_cost_error last: `{format_value(err.get('last'))}`",
            "- target is reached when cost error trends toward zero without lambda saturating at max.",
        ]
    if phase.key == "phase21":
        connected = summary.get("connected_new_coverage_trend", {})
        disconnected = summary.get("disconnected_new_coverage_trend", {})
        return [
            f"- best eval connected_new_coverage_gain: `{format_value(summary.get('best_connected_new_coverage'))}`",
            f"- best eval disconnected_new_coverage_gain: `{format_value(summary.get('best_disconnected_new_coverage'))}`",
            f"- best eval connected_new_coverage_ratio: `{format_value(summary.get('best_connected_new_coverage_ratio'))}`",
            f"- connected_new_coverage first/last: `{format_value(connected.get('first'))}` -> `{format_value(connected.get('last'))}`",
            f"- disconnected_new_coverage first/last: `{format_value(disconnected.get('first'))}` -> `{format_value(disconnected.get('last'))}`",
            "- phase21 is promising only if connected coverage grows while violation does not follow phase17/trial02's late blow-up.",
        ]
    if phase.key == "phase22":
        aux = summary.get("aux_loss_trend", {})
        relaxed = summary.get("relaxed_success_trend", {})
        return [
            f"- best eval relaxed_success: `{format_value(summary.get('best_relaxed_success'))}`",
            f"- aux_total_loss first/last: `{format_value(aux.get('first'))}` -> `{format_value(aux.get('last'))}`",
            f"- rollout relaxed_success first/last: `{format_value(relaxed.get('first'))}` -> `{format_value(relaxed.get('last'))}`",
            "- phase22 uses existing `comm_adjacency`; it does not add connectivity-specific observation fields.",
        ]
    if phase.key == "phase26":
        return [
            "- formal proxy target: coverage >= `0.55`, radius >= `0.45`, violation <= `0.10`.",
            f"- best eval coverage: `{format_value(summary.get('best_coverage'))}`",
            f"- best eval radius: `{format_value(summary.get('best_radius'))}`",
            f"- best eval violation: `{format_value(summary.get('best_violation'))}`",
            "- phase26 should be judged against the formal proxy, not the earlier relaxed thresholds.",
        ]
    aux = summary.get("aux_loss_trend", {})
    pred = summary.get("aux_pred_violation_trend", {})
    target = summary.get("aux_target_violation_trend", {})
    return [
        f"- aux_total_loss first/last: `{format_value(aux.get('first'))}` -> `{format_value(aux.get('last'))}`",
        f"- pred violation last vs target last: `{format_value(pred.get('last'))}` vs `{format_value(target.get('last'))}`",
        "- prediction is useful when loss falls and behavior metrics improve without worsening violation.",
    ]


def hard_filter_lines(summary: dict[str, Any] | None) -> list[str]:
    lines = [
        "- Config sets `connectivity_action_filter: false`.",
        "- Direct waypoint action space is unchanged; actor output is not replaced by connectivity code.",
        "- `connectivity_candidate_filter: false` is also set for this debug path; Direct waypoint training does not use candidate replacement.",
    ]
    if summary is None:
        lines.append("- After training, confirm CSV fields `connectivity_action_filter_enabled == 0` and `connectivity_filter_replacement_count == 0`.")
        return lines
    lines.extend(
        [
            f"- last `connectivity_action_filter_enabled`: `{format_value(summary.get('last_filter_enabled'))}`",
            f"- last `connectivity_filter_replacement_count`: `{format_value(summary.get('last_filter_replacements'))}`",
            f"- best eval `connectivity_action_filter_enabled`: `{format_value(summary.get('best_eval_filter_enabled'))}`",
            f"- best eval `connectivity_filter_replacement_count`: `{format_value(summary.get('best_eval_filter_replacements'))}`",
        ]
    )
    return lines


def write_report(
    phase_root: Path,
    phase: PhaseSpec,
    status: str,
    args: argparse.Namespace,
    output_dir: Path,
    policy_config: Path,
    env_config: Path,
    command: list[str],
    summary: dict[str, Any] | None = None,
    return_code: int | None = None,
) -> None:
    lines = [
        f"# {phase.name} report",
        "",
        f"- Status: `{status}`",
        f"- Created at: `{utc_now()}`",
        f"- Phase title: `{phase.title}`",
        f"- Task: `{TASK}`",
        f"- Output dir: `{output_dir}`",
        f"- Policy config: `{policy_config}`",
        f"- Env config: `{env_config}`",
        f"- Return code: `{format_value(return_code)}`",
        f"- trial: `trial{int(args.trial):02d}`",
        f"- tuning preset: `{args.tuning}`",
        f"- total_updates: `{args.total_updates}`",
        f"- num_envs: `{args.num_envs}`",
        f"- rollout_steps: `{args.rollout_steps}`",
        "- train_randomization: `random`",
        "- eval_randomization: `random`",
        "",
        "## Mechanism",
        "",
        phase.mechanism,
        "",
        "## Hard Filter Confirmation",
        "",
        *hard_filter_lines(summary),
        "",
        "## Command",
        "",
        "```bash",
        " ".join(command),
        "```",
        "",
        "## Base Metrics",
        "",
            "| best success | relaxed success | best radius | best coverage | best violation | best update | best reward | last clip | last valid | last KL | last value loss |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if summary is None:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        lines.append(
            "| "
            + " | ".join(
                format_value(summary.get(key))
                for key in [
                    "best_success",
                    "best_relaxed_success",
                    "best_radius",
                    "best_coverage",
                    "best_violation",
                    "best_update",
                    "best_reward",
                    "last_clip",
                    "last_valid",
                    "last_kl",
                    "last_value_loss",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Connectivity Diagnostics",
            "",
            "| connected ratio | margin min | margin penalty | action risk | repeat footprint | uncovered approach | connected new cov | disconnected new cov | command reject | command update |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if summary is None:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        lines.append(
            "| "
            + " | ".join(
                format_value(summary.get(key))
                for key in [
                    "best_connected_to_base_ratio",
                    "best_connectivity_margin_min",
                    "best_connectivity_margin_penalty",
                    "best_action_disconnect_risk",
                    "best_repeat_footprint",
                    "best_uncovered_approach",
                    "best_connected_new_coverage",
                    "best_disconnected_new_coverage",
                    "best_command_reject_rate",
                    "best_command_update_rate",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Phase-Specific Checks",
            "",
            *phase_specific_lines(phase, summary),
            "",
            "## Judgment",
            "",
            f"- Solves: {phase.solved_text}",
            f"- Still fails when: {phase.failure_text}",
            f"- Next phase: {phase.next_text}",
        ]
    )
    (phase_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase17-28 soft connectivity debug phases.")
    parser.add_argument(
        "--phase",
        choices=["all", "soft17_20", *PHASES.keys(), *[phase.name for phase in PHASES.values()]],
        default="all",
    )
    parser.add_argument("--runtime", default=minute_runtime())
    parser.add_argument("--output-root", default="outputs/debug/mappo_direct_specialists")
    parser.add_argument("--total-updates", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--env-backend", choices=["sync", "thread", "process"], default="process")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--tuning", choices=sorted(TUNING_PRESETS), default="baseline")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_phases(value: str) -> list[PhaseSpec]:
    if value == "all":
        return [
            PHASES[key]
            for key in [
                "phase17",
                "phase18",
                "phase19",
                "phase20",
                "phase21",
                "phase22",
                "phase23",
                "phase24",
                "phase25",
                "phase26",
                "phase27",
                "phase28",
            ]
        ]
    if value == "soft17_20":
        return [PHASES[key] for key in ["phase17", "phase18", "phase19", "phase20"]]
    for phase in PHASES.values():
        if value in {phase.key, phase.name}:
            return [phase]
    raise ValueError(f"Unknown phase: {value}")


def validate_long_run_args(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    required = {
        "total_updates": (int(args.total_updates), 2000),
        "num_envs": (int(args.num_envs), 8),
        "rollout_steps": (int(args.rollout_steps), 128),
        "eval_interval": (int(args.eval_interval), 100),
        "eval_episodes": (int(args.eval_episodes), 20),
    }
    too_small = [
        f"{name}={actual} < {minimum}"
        for name, (actual, minimum) in required.items()
        if actual < minimum
    ]
    if too_small:
        raise ValueError(
            "phase17-28 are long-run debug phases; refusing short-run settings: "
            + ", ".join(too_small)
        )


def run_phase(phase: PhaseSpec, args: argparse.Namespace) -> tuple[str, int | None]:
    phase_name = validate_phase_name(phase.name)
    runtime = validate_runtime(args.runtime)
    phase_root = ROOT / args.output_root / phase_name / runtime
    config_dir = phase_root / "configs"
    output_dir = make_debug_output_dir(ROOT / args.output_root, phase_name, runtime, TASK, int(args.trial), 0)
    phase_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_config = make_policy_config(phase, args, config_dir)
    env_config = make_env_config(phase, args, config_dir)
    command = build_command(phase, policy_config, env_config, output_dir, args)
    (output_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    write_report(phase_root, phase, "CONFIGURED" if args.dry_run else "RUNNING", args, output_dir, policy_config, env_config, command)
    if args.dry_run:
        summary = {
            "phase_name": phase.name,
            "runtime": runtime,
            "status": "CONFIGURED",
            "created_at": utc_now(),
            "output_dir": str(output_dir),
            "policy_config": str(policy_config),
            "env_config": str(env_config),
            "command": command,
        }
        (phase_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return phase.name, None

    with (output_dir / "stdout.log").open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
        return_code = int(process.wait())
    metrics_summary = summarize_training(output_dir)
    status = "COMPLETED" if return_code == 0 else "FAILED_RUNTIME"
    summary = {
        "phase_name": phase.name,
        "runtime": runtime,
        "status": status,
        "created_at": utc_now(),
        "return_code": return_code,
        "output_dir": str(output_dir),
        "policy_config": str(policy_config),
        "env_config": str(env_config),
        "metrics": metrics_summary,
    }
    (phase_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(
        phase_root,
        phase,
        status,
        args,
        output_dir,
        policy_config,
        env_config,
        command,
        summary=metrics_summary,
        return_code=return_code,
    )
    return phase.name, return_code


def main() -> int:
    args = parse_args()
    validate_long_run_args(args)
    results: dict[str, int | None] = {}
    for phase in selected_phases(args.phase):
        name, return_code = run_phase(phase, args)
        results[name] = return_code
        if return_code not in (None, 0):
            break
    print(json.dumps({"results": results}, indent=2, sort_keys=True), flush=True)
    return 1 if any(code not in (None, 0) for code in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

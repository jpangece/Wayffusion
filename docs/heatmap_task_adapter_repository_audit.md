# Heatmap Task Adapter Repository Audit

## Purpose and scope

This document records repository-verified architecture relevant to a planned heatmap-guided, conditional task-element adaptation upgrade. It covers the current environment, observations, waypoint actions, candidate systems, policies, MAPPO implementation, rewards, safety, evaluation, configuration, and safe extension points. No object, spatial, or relation adapter currently exists in the repository.

The upgrade must preserve the existing five-channel `task_field`; its channels must not be reinterpreted or overwritten. Task-element heatmaps should be introduced through a new optional observation key, disabled by default.

## Environment and execution flow

The production environment is `envs/waypoint_marl_env.py::WaypointMultiUAVEnv`, re-exported from `envs/__init__.py`. The repository does not register a Gymnasium string environment ID; callers instantiate the class directly.

Primary entry points are:

- `scripts/train_mappo_waypoint.py::build_env_batch` for training.
- `utils/waypoint_evaluation.py::evaluate_waypoint_policy_episodes` for reusable evaluation.
- `utils/waypoint_vector_env.py::{SyncWaypointEnvBatch, ThreadWaypointEnvBatch, ProcessWaypointEnvBatch, make_waypoint_env_batch}` for vectorized execution.
- `scripts/benchmarks/swarm30/evaluate.py` for Swarm-30 checkpoint evaluation.

`WaypointMultiUAVEnv.step` provides the PettingZoo-style dictionary API. `WaypointMultiUAVEnv.step_global` is the centralized trainer fast path. The verified transition sequence is:

```text
policy waypoint
→ WaypointMultiUAVEnv._actions_to_waypoints
→ envs/waypoint/safety.py::SafetyLayer.validate_waypoints
→ action_adapter.adapt
→ dynamics_backend.step
→ current_scenario.compute_rewards
→ observations and info
```

The default execution components are built by `envs/waypoint/control/__init__.py::{build_action_adapter, build_dynamics_backend}`. Configuration selects the waypoint velocity tracker and vendored MPE-core dynamics.

## Observation contracts and tensor shapes

Let `N` be the UAV count, `G` the grid size, `K` the candidate count, `T` the rollout horizon, and `E` the vector-environment count. Spaces are declared by `WaypointMultiUAVEnv._build_spaces`.

Per-agent observations from `WaypointMultiUAVEnv._build_agent_observations` are:

| Key | Shape | Source |
|---|---:|---|
| `self_state` | `[8]` | One row of the normalized swarm state |
| `neighbor_relative_states` | `[N-1,4]` | `WaypointMultiUAVEnv._neighbor_relative_states` |
| `task_field` | `[5,G,G]` | `WaypointScenario.build_task_field` |
| `task_id` | `[6]` | `WaypointScenario.task_one_hot` |
| `mission_goal` | `[3]` | Resolved mission goal |
| `goal_mask` | `[3]` | Active mission-goal components |
| `global_summary` | `[8]` | `MissionWorld.get_global_summary` |
| `all_uav_states` | `[N,8]` | `tasks/waypoint/base.py::normalized_positions` |
| `comm_adjacency` | `[N+1,N+1]` | UAVs plus the base node |

In legacy/debug candidate mode, the per-agent observation additionally contains `candidate_waypoints [K,2]`, `candidate_features [K,6]`, and `candidate_mask [K]`.

`normalized_positions` constructs each UAV row as normalized XY position, normalized XY velocity, battery, normalized role ID, base-connectivity flag, and last-action-valid flag.

The centralized state from `WaypointMultiUAVEnv.get_global_state` contains:

| Key | Environment shape | Rollout-buffer shape |
|---|---:|---:|
| `task_field` | `[5,G,G]` | `[T,E,5,G,G]` |
| `global_task_field` | `[5,G,G]` | `[T,E,5,G,G]` |
| `all_uav_states` | `[N,8]` | `[T,E,N,8]` |
| `task_id` | `[6]` | `[T,E,6]` |
| `mission_goal` | `[3]` | `[T,E,3]` |
| `goal_mask` | `[3]` | `[T,E,3]` |
| `global_info` | `[8]` | `[T,E,8]` |
| `comm_adjacency` | `[N+1,N+1]` | `[T,E,N+1,N+1]` |
| `agent_mask` | `[N]` | `[T,E,N]` |

`WaypointScenario.build_task_field` defines the existing five channels as coverage, clipped visit count, normalized belief, risk, and target/POI occupancy. Task-specific overrides include `PriorityInspectionScenario.build_task_field` and `TargetInterceptionScenario.build_task_field`. These semantics are an established baseline contract and must remain unchanged.

## Action and waypoint flow

`configs/env/waypoint_missions.yaml` defaults to `action_interface: waypoint`, `action_mode: waypoint`, and `waypoint_dim: 2`.

`policies/direct_waypoint_policy.py::DirectWaypointPolicy` produces a Gaussian waypoint delta. Its important tensors are:

- delta mean and sampled training action: `[B,N,2]`;
- environment waypoint: `[B,N,2]`;
- per-agent log probability and entropy: `[B,N]`;
- centralized value: `[B]`.

`DirectWaypointPolicy._clip_delta` bounds delta magnitude by `max_delta`. `DirectWaypointPolicy._delta_to_waypoint` adds the delta to the current normalized XY position scaled by `map_size`, then clamps it to the map.

`WaypointMultiUAVEnv._actions_to_waypoints` validates per-agent shape and finiteness. `WaypointMultiUAVEnv.step` then applies safety validation, the action adapter, and the dynamics backend. Raw and safe waypoints are retained in transition information.

Although the environment can declare a three-dimensional action space when `waypoint_dim == 3`, `_actions_to_waypoints` copies only `raw[:2]` into the simulated world position. `DirectWaypointPolicy` also emits two coordinates. The current physical waypoint flow is therefore effectively two-dimensional, and the first heatmap-adaptation implementation should remain two-dimensional.

The environment action contract must remain final waypoint coordinates `[N,2]`; adding adapters must not change this API.

## Candidate generation systems

Two distinct candidate systems are present.

Scenario-side generation enters through `tasks/waypoint/base.py::WaypointScenario.generate_candidate_waypoints`. Task implementations call functions in `planners/waypoint_candidates.py`, including `coverage_candidates`, `belief_candidates`, `poi_candidates`, `connectivity_boundary_candidates`, `target_prediction_candidates`, and `merge_and_pad_candidates`. Their common contract is candidates `[N,K,2]` plus mask `[N,K]`. `WaypointMultiUAVEnv._refresh_candidates` applies `SafetyLayer.filter_candidates` and pads or truncates to `K`.

Policy-side generation is implemented independently by `policies/candidate_selection_policy.py::CandidateSelectionWaypointPolicy.generate_candidates`. It constructs current-position, radial, high-scoring grid, and optional connectivity candidates. It returns candidates `[B,N,K,2]`, features `[B,N,K,9]`, and mask `[B,N,K]`. `CandidateSelectionWaypointPolicy.forward` produces logits `[B,N,K]`; `get_action_and_value` samples categorical indices `[B,N]` but emits selected waypoint coordinates `[B,N,2]` to the environment.

`action_mode == "candidate_index_legacy"` is a separate diagnostic compatibility path handled by `WaypointMultiUAVEnv._legacy_actions_to_waypoints`. It should not be modified in the initial implementation. The initial heatmap work should choose one candidate-generation path rather than changing the scenario-side and policy-side systems simultaneously.

## Policy architectures

Policy construction is dispatched by `policies/__init__.py::build_policy`.

`DirectWaypointPolicy` contains:

- a stride-two convolutional `field_encoder` over `[B,5,G,G]` and adaptive global pooling;
- `agent_encoder`, mapping `[B,N,8]` to `[B,N,H]`;
- optional `nn.MultiheadAttention` over UAV tokens;
- optional communication-graph processing in `_comm_graph_context`;
- optional local sampling in `_spatial_field_context`;
- optional moment features in `_field_moment_context`;
- a per-UAV actor ending in two delta coordinates;
- a shared learned `log_std [2]`;
- a centralized scalar critic.

The baseline `configs/policy/mappo_waypoint.yaml` uses convolution channels `[16,32,64]`, `agent_hidden_dim: 64`, and `joint_hidden_dim: 192`.

`UVFADirectWaypointPolicy` subclasses `DirectWaypointPolicy` and enables goal conditioning. `DirectWaypointPolicy.encode_goal` consumes `task_id [B,6]`, `mission_goal [B,3]`, and `goal_mask [B,3]`. Its critic combines state embedding, goal embedding, and their elementwise product.

`CandidateSelectionWaypointPolicy` uses a CNN field encoder, shared UAV encoder, optional attention, candidate encoder, candidate logit head, and centralized critic.

Existing baseline classes and behavior must be preserved. Heatmap conditioning should be introduced as a separate policy class registered through `build_policy`, with new functionality disabled by default.

## PPO/MAPPO rollout and update flow

The implementation is `algorithms/mappo_waypoint.py::MAPPOWaypointTrainer` with storage in `MAPPOWaypointBatch` and `WaypointRolloutBuffer`.

Important rollout tensors are:

| Tensor | Shape |
|---|---:|
| observations | `[T,E,...]` |
| direct environment actions | `[T,E,N,2]` |
| direct training actions | `[T,E,N,2]` |
| candidate training actions | `[T,E,N]` |
| old log probabilities | `[T,E,N]` |
| values | `[T,E]` |
| team rewards | `[T,E]` |
| per-agent rewards | `[T,E,N]` |
| advantages and returns | `[T,E]` |

`MAPPOWaypointTrainer.collect_rollout` samples policy actions, steps the vector environment, obtains bootstrap values, applies optional Lagrangian connectivity cost and reward normalization, stores the transition, and calls `compute_gae_returns`.

`compute_gae_returns` bootstraps across time-limit truncation using `1 - terminated`, while `1 - done_for_reset` prevents recursive GAE from crossing an auto-reset boundary.

`MAPPOWaypointTrainer.update` flattens time and environment dimensions to `[T×E,...]`, optionally normalizes advantages, and expands the scalar team advantage across the agent dimension for the per-agent PPO ratios. The loss combines clipped policy loss, centralized value MSE, entropy regularization, and optional connectivity/coverage auxiliary MSE losses. Optimization uses `torch.optim.Adam`, gradient-norm clipping, optional linear learning-rate decay, and optional approximate-KL early stopping.

PPO currently uses the scalar team reward for advantage computation and critic targets. Per-agent rewards are stored as `[T,E,N]`, but they are not independently used by the PPO objective.

## Reward composition and normalization

Each scenario implements `compute_rewards` and returns a scalar `team_reward`, `per_agent_rewards [N]`, component and metric dictionaries, and success/failure state.

- `AreaCoverageScenario.compute_rewards` combines new coverage, uncovered approach, overlap/repeat, path distance, and safety terms.
- `BeliefSearchScenario.compute_rewards` combines searched belief mass, approach, detection bonus, repeat, distance, and safety terms.
- `PriorityInspectionScenario.compute_rewards` combines weighted POI progress, approach, repeat, lateness, distance, and safety terms.
- `ConnectivityExpansionScenario.compute_rewards` combines connected expansion and coverage, disconnected coverage, relay, connectivity cost/margins, action risk, distance, and safety terms.
- `DynamicTargetEscortScenario.compute_rewards` combines target observation, view diversity, distance-band, flight-distance, and safety terms.
- `TargetInterceptionScenario.compute_rewards` combines belief coverage, containment, interception/escape, flight-distance, and safety terms.

Shared reward helpers are `tasks/waypoint/base.py::{per_agent_new_coverage, uncovered_approach_gain, path_length_delta, safety_penalty_count}`.

`algorithms/mappo_waypoint.py::RunningMeanStd` normalizes scalar training rewards without clipping. `MAPPOWaypointTrainer._normalize_rewards` supports one global running statistic or separate statistics per task through `reward_norm_scope`. Advantage normalization is separate and covers the collected `[T,E]` advantage array.

## Collision and safety handling

`envs/waypoint/safety.py::SafetyLayer.validate_waypoints` validates shape/finiteness, maximum command distance, geofence, no-fly zones, minimum UAV separation, and optional predicted connectivity. It returns `SafetyResult` with raw and safe waypoints `[N,2]` and per-agent validity/rejection masks `[N]`.

`SafetyLayer.filter_candidates` applies corresponding checks to candidates `[N,K,2]`. `ensure_fallback_candidate` ensures an available fallback.

Physical collision response belongs to the configured dynamics backend. The default MPE configuration in `configs/env/waypoint_missions.yaml` sets agent size, contact force/margin, and boundary/no-fly projection.

`WaypointMultiUAVEnv._build_shared_info` reports collision, no-fly, geofence, safety, invalid-action, and connectivity-filter diagnostics. Its `collision_rate` field is derived from `safety_violation_count / (step_count × N)`, rather than directly from `collision_count`.

## Evaluation, metrics, checkpoints, and logging

Coverage geometry is implemented by `envs/waypoint/world.py::MissionWorld`, especially `coverage_ratio`, `sensor_masks_for_positions`, `exact_sensor_masks_for_positions`, `coverage_transition_stats`, and `accumulate_sensor_footprints`. `CoverageTransitionStats` includes sensor masks `[N,G,G]`, newly covered mask `[G,G]`, per-agent new masks `[N,G,G]`, per-agent coverage `[N]`, and repeat-footprint ratios `[N]`.

Flight cost is primarily represented by per-step `path_length_delta` and cumulative `UAVState.path_length`. Evaluation also records speed, control norm, desired speed, distance to waypoint, arrival rate, and command update/rejection rates. No reward-integrated battery-energy model was found.

`utils/waypoint_evaluation.py::evaluate_waypoint_policy_episodes` records return, success, path length, safety statistics, goal progress, action validity, dynamics integrity, adapter statistics, and numeric scenario metrics. `_aggregate` computes arithmetic means, while `evaluate_waypoint_policy_per_task` provides per-task and overall summaries. `MAPPOWaypointTrainer.evaluate` supports fixed/random domain randomization, generalization gaps, goal splits, and media recording.

`MAPPOWaypointTrainer.train` saves `checkpoint_XXXX.pt` and `checkpoint_best_eval.pt`. Each checkpoint contains only `model_state_dict` and `train_config`. Optimizer, reward-normalizer, update, Lagrangian, RNG, and environment state are not saved. Current checkpoints are therefore initialization checkpoints, not exact training-resume snapshots.

Training output is organized under `checkpoints/`, `snapshot/`, `tensorboard/`, `training_metrics.csv`, evaluation media, and `best_eval_summary.json`. Relevant symbols are `scripts/train_mappo_waypoint.py::{build_writer, log_tensorboard}`, `algorithms/mappo_waypoint.py::write_metrics_csv`, and `utils/tensorboard_metrics.py::should_log_tensorboard_metric`.

The repository has no Dockerfile. `requirements-server.txt` does not install PyTorch even though neural training and checkpoint evaluation import it.

## Configuration and task registration

Configuration is plain YAML loaded by `scripts/train_mappo_waypoint.py::load_yaml`; no Hydra or typed configuration schema is used. Principal files are `configs/env/waypoint_missions.yaml`, `configs/policy/mappo_waypoint.yaml`, `configs/policy/uvfa/*.yaml`, specialist policy files, `configs/eval/waypoint_eval.yaml`, and `configs/benchmarks/swarm30_v1.yaml`.

`scripts/train_mappo_waypoint.py::sync_policy_env_config` copies map and safety geometry into policy configuration.

Task registration is the static `tasks/waypoint/__init__.py::SCENARIO_REGISTRY`, built by `build_waypoint_scenario`. Its six registered classes correspond to the ordering in `tasks/waypoint/base.py::WAYPOINT_TASKS`; the registry and one-hot ordering must remain synchronized.

## Backward-compatibility constraints

- Preserve the existing `task_field [5,G,G]` and every current channel meaning.
- Add task-element heatmaps through a separate optional observation key.
- Keep task-element observations and behavior disabled by default.
- Preserve existing baseline policy classes, names, parameter layouts, and default behavior.
- Add a separate heatmap-conditioned policy class instead of changing `direct_waypoint` semantics.
- Preserve final waypoint actions `[N,2]` at the environment boundary.
- Keep the first implementation two-dimensional.
- Leave `candidate_index_legacy` unchanged initially.
- Initially extend only one of the two candidate-generation systems.
- Maintain loading of existing baseline initialization checkpoints where possible; strict state-dict loading means new parameters must not be injected into unchanged baseline classes.

## Recommended extension points

Heatmap generation should extend the scenario abstraction around `WaypointScenario.build_task_field` without changing its return contract. A new optional hook can derive task-element data from scenario state, while `WaypointMultiUAVEnv._build_spaces`, `get_global_state`, and `_build_agent_observations` expose fixed-shape heatmap and masked-element tensors only when enabled.

A heatmap-conditioned policy should be added through `policies/__init__.py::build_policy` as a new class adjacent to `DirectWaypointPolicy`. It can reuse the field CNN, `_spatial_field_context`, `_field_moment_context`, UAV attention, and `_comm_graph_context`.

No adapter modules currently exist. Future object, spatial, and relation adapters should have explicit masked tensor contracts:

```text
object features [B,M,D] + object mask [B,M] → object tokens [B,M,H]
heatmap [B,C,G,G] + UAV XY [B,N,2] → spatial context [B,N,H]
UAV/object tokens + adjacency/masks → relation context [B,N,H]
```

Task-element sources must come from verified scenario state in the relevant `reset`, `step_update`, and `compute_rewards` implementations. The communication adjacency includes a base node while `all_uav_states` contains only UAVs, so relation handling must treat the base explicitly.

Metric extensions belong in scenario `get_metrics`/`compute_rewards`, `WaypointMultiUAVEnv._build_shared_info`, `evaluate_waypoint_policy_episodes`, `MAPPOWaypointTrainer.collect_rollout`, and TensorBoard filtering.

## Phased implementation plan

1. Add baseline contract tests for observation keys/shapes, waypoint actions, rollout tensors, default-disabled behavior, and existing checkpoint loading.
2. Add optional fixed-shape task-element tensors and a separate heatmap key behind a configuration flag that defaults to false.
3. Add object, spatial, and relation adapters plus a separate heatmap-conditioned policy class; keep baseline policy code paths unchanged.
4. Fuse adapted context into the actor and centralized critic while preserving policy outputs and the environment waypoint contract.
5. Extend one candidate-generation path with optional heatmap guidance, retaining current-position and safe fallback candidates.
6. Add optional adapter/heatmap auxiliary objectives through `PolicyOutput.aux` and the existing auxiliary-loss pattern, with zero default coefficients.
7. Add evaluation metrics and perform baseline, component-ablation, randomization, goal-split, and swarm-size comparisons.
8. Version future checkpoint payloads for exact resumption while retaining compatibility with existing initialization checkpoints; validate server/container requirements separately.

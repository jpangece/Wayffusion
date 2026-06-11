# Wayffusion

Wayffusion is a waypoint-level multi-UAV MARL mission suite. The main API follows PettingZoo ParallelEnv semantics: each UAV is an agent, and each agent sends a final waypoint coordinate to the environment.

The environment simulates waypoint tracking, real MPE core particle dynamics, safety checks, task updates, rewards, metrics, and visualization. Candidate waypoint generation is not part of the environment action API. It belongs to policy or planner code.

## Main Components

- Environment: `envs/waypoint_marl_env.py::WaypointMultiUAVEnv`
- Training entry: `scripts/train_mappo_waypoint.py`
- Validation entry: `scripts/check/validate_waypoint_mappo.py`
- Candidate-selection policy: `policies/candidate_selection_policy.py::CandidateSelectionWaypointPolicy`
- Direct waypoint policy: `policies/direct_waypoint_policy.py::DirectWaypointPolicy`
- UVFA-conditioned direct policy: `policies/direct_waypoint_policy.py::UVFADirectWaypointPolicy`
- Action adapters: `envs/waypoint/control/action_adapters.py`
- Dynamics backends: `envs/waypoint/control/dynamics_backends.py`
- MAPPO trainer: `algorithms/mappo_waypoint.py::MAPPOWaypointTrainer`

## Action API

Default action mode is `waypoint`:

```python
actions = {
    "uav_0": np.asarray([0.32, 0.18], dtype=np.float32),
    "uav_1": np.asarray([0.40, 0.22], dtype=np.float32),
    "uav_2": np.asarray([0.24, 0.30], dtype=np.float32),
}
```

For 2D worlds, `action_space(agent_id)` is `Box([0, 0], [map_size, map_size], shape=(2,))`. A future 3D mode can extend this to `[x, y, altitude]`.

The environment checks geofence, no-fly zones, maximum waypoint distance, minimum UAV distance, and optional connectivity constraints. Invalid waypoints are corrected to a safe fallback unless `strict_action_validation=true`.

`candidate_index_legacy` exists only for diagnostics. It is not the mainline API.

## Transition Dynamics

The default transition stack is:

```text
Policy env_action [B,N,2]
  -> WaypointMultiUAVEnv.step(dict[str, waypoint])
  -> SafetyLayer.validate_waypoints
  -> WaypointVelocityTracker
  -> MPECoreBackend
  -> scenario reward/metrics
```

Default config:

- `action_interface: waypoint`
- `action_adapter.name: waypoint_velocity_tracker`
- `dynamics_backend.name: mpe_core`

`ActionAdapter` is environment transition dynamics, not policy. It converts safe final waypoints into low-level MPE physical actions. `MPECoreBackend` owns the physical backend call: it synchronizes Wayffusion UAV state into MPE agents, writes `agent.action.u`, calls the real MPE `World.step()`, then synchronizes MPE agent state back into Wayffusion.

Wayffusion no longer uses a self-written MPE-style particle backend. The old handwritten particle backend and kinematic waypoint backend were removed from the runnable mainline. If a config asks for `mpe_particle` or `kinematic_point`, backend construction fails and tells the user to use `mpe_core`.

Available adapters:

- `WaypointVelocityTracker`: tracks a waypoint through desired velocity and acceleration control.
- `WaypointPDTracker`: debug/ablation PD controller.
- `VelocityToForceAdapter` and `DirectAccelerationAdapter`: smoke placeholders for future lower-level action interfaces.

Only supported backend:

- `MPECoreBackend`: calls actual MPE `World.step()` from the explicitly configured source. The default source is the vendored local OpenAI MPE core in `third_party/openai_mpe/core.py`; using installed `mpe2` requires manually setting `dynamics_backend.source: mpe2`.

## Policy Families

`CandidateSelectionWaypointPolicy` internally generates task-aware candidate waypoints, samples a categorical index per UAV, maps that index to a final waypoint, and returns:

- `env_action [B, N, 2]`: final waypoint sent to the environment.
- `train_action [B, N]`: selected candidate index used for PPO log-prob recomputation.
- `logprob [B, N]`, `entropy [B, N]`, `value [B]`.

`DirectWaypointPolicy` directly samples a Gaussian waypoint delta per UAV, clips it by vector norm, converts it into a final waypoint, and uses the sampled delta as `train_action`.

`UVFADirectWaypointPolicy` keeps the same action distribution but encodes a compact episode-level `mission_goal` with a separate goal stream. The centralized critic combines state and goal embeddings as a universal value function \(V(s,g)\). See `docs/uvfa_waypoint_mappo_zh.md`.

The trainer only requires a common `PolicyOutput`; it does not assume the policy is categorical or Gaussian.

## Mission Suite

The waypoint task family is under `tasks/waypoint/`:

- `area_coverage`: uniform area coverage with new-coverage reward and repeat-coverage penalty.
- `belief_search`: probability/belief map weighted search.
- `priority_inspection`: weighted POI inspection with repeat and deadline penalties.
- `connectivity_expansion`: cooperative expansion while maintaining multi-hop connectivity to base.
- `dynamic_target_escort`: waypoint-level dynamic target monitoring/escort.
- `target_interception`: route-branch interception and containment over future target belief.

Connectivity is a standalone mission type, not just a global constraint.

## MAPPO

`MAPPOWaypointTrainer` uses:

- per-agent PPO ratio `exp(new_logprob_i - old_logprob_i)`;
- team advantage broadcast to agents in the first version;
- centralized value loss over `[B]`;
- separate `terminated` and `truncated` handling;
- bootstrap on truncation and no bootstrap on true termination;
- CSV metrics, TensorBoard, checkpoints, eval media, and `best_eval_summary.json`.

## Quick Start

Run tests:

```bash
python -m pytest tests/test_waypoint_env_api.py
python -m pytest tests/test_action_adapters.py
python -m pytest tests/test_mpe_core_backend.py
python -m pytest tests/test_waypoint_scenarios.py
python -m pytest tests/test_waypoint_rewards.py
python -m pytest tests/test_candidate_selection_policy.py
python -m pytest tests/test_direct_waypoint_policy.py
python -m pytest tests/test_mappo_waypoint_trainer.py
python -m pytest tests/test_waypoint_rendering.py
```

Validate candidate-selection MAPPO:

```bash
python scripts/check/validate_waypoint_mappo.py \
  --tasks area_coverage belief_search connectivity_expansion \
  --policy-class candidate_selection_waypoint \
  --dynamics-backend mpe_core \
  --action-adapter waypoint_velocity_tracker \
  --num_agents 3 \
  --total_updates 5 \
  --eval_episodes 2 \
  --record_eval_episodes 1 \
  --record_format gif \
  --headless
```

Validate direct waypoint MAPPO:

```bash
python scripts/check/validate_waypoint_mappo.py \
  --tasks area_coverage belief_search \
  --policy-class direct_waypoint \
  --dynamics-backend mpe_core \
  --action-adapter waypoint_velocity_tracker \
  --num_agents 3 \
  --total_updates 2 \
  --eval_episodes 1 \
  --record_eval_episodes 1 \
  --record_format gif \
  --headless
```

Run debug training:

```bash
python scripts/train_mappo_waypoint.py \
  --config configs/policy/mappo_waypoint_debug.yaml \
  --env-config configs/env/waypoint_missions.yaml \
  --tasks area_coverage belief_search priority_inspection connectivity_expansion \
  --num_agents 4 \
  --num_envs 4 \
  --total_updates 10 \
  --eval_episodes 2 \
  --record_eval_episodes 1 \
  --record_format gif \
  --headless
```

Run the four-task UVFA comparison:

```bash
python scripts/debug/phase32_uvfa_goal_conditioning.py
```

Run longer training:

```bash
python scripts/train_mappo_waypoint.py \
  --config configs/policy/mappo_waypoint.yaml \
  --env-config configs/env/waypoint_missions.yaml \
  --tasks area_coverage belief_search priority_inspection connectivity_expansion \
  --num_agents 4 \
  --num_envs 8 \
  --total_updates 1000 \
  --eval_episodes 20 \
  --record_eval_episodes 2 \
  --record_format gif \
  --headless
```

## Outputs

Waypoint debug validation outputs go under:

```text
outputs/debug/waypoint_mappo_validation/
```

Waypoint training outputs go under:

```text
outputs/training/mappo_waypoint/<timestamp>/<run_name>/
```

Each training run writes `training_metrics.csv`, TensorBoard logs, checkpoints, `best_eval_summary.json`, and optional GIF/MP4 eval media.

View TensorBoard:

```bash
/opt/conda/bin/tensorboard --logdir outputs/training/mappo_waypoint --bind_all --port 6006
```

## Swarm30 Benchmark

The frozen `swarm30_v1` protocol evaluates standardized specialists, tuned
specialists, and a six-task generalist at 4, 10, 20, and 30 UAVs. It scales map
area and task load while keeping sensing, communication, speed, and waypoint
capabilities in fixed physical units.

```bash
bash scripts/benchmarks/swarm30/launch_tmux.sh
bash scripts/benchmarks/swarm30/status.sh
```

The launcher waits for the active Phase31 connectivity reproducibility run
before starting long benchmark training. See
`docs/swarm30_v1_benchmark_zh.md` for the complete protocol.

## Removed Centralized V1

The previous centralized Gymnasium-style benchmark has been removed from the working tree to keep the MARL branch focused on waypoint-level multi-agent training. It remains recoverable from git history if an old experiment needs to be reproduced.

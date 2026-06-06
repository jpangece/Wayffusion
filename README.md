# Wayffusion

Wayffusion is a waypoint-level multi-UAV MARL mission suite. The main API follows PettingZoo ParallelEnv semantics: each UAV is an agent, each agent selects a discrete candidate waypoint index, and the environment simulates bounded waypoint execution rather than low-level flight control.

The suite is designed for CTDE/MAPPO research and future real-drone waypoint experiments. Actor policies consume per-agent candidate waypoint observations; centralized critics can consume the provided global state.

## Main Environment

- Main environment: `envs/waypoint_marl_env.py::WaypointMultiUAVEnv`
- Main training entry: `scripts/train_mappo_waypoint.py`
- Main validation entry: `scripts/check/validate_waypoint_mappo.py`
- Main policy: `policies/mappo_waypoint_policy.py::MAPPOWaypointPolicy`
- Main trainer: `algorithms/mappo_waypoint.py::MAPPOWaypointTrainer`

The action for each UAV is an integer candidate waypoint index:

```python
actions = {
    "uav_0": 3,
    "uav_1": 7,
    "uav_2": 1,
}
```

The selected candidate is a continuous waypoint coordinate. `WaypointExecutionModel` moves each UAV toward that waypoint with bounded speed, geofence/no-fly checks, minimum-distance checks, path-length accounting, and trajectory history.

## Mission Suite

The waypoint task family is under `tasks/waypoint/`:

- `area_coverage`: uniform area coverage with new-coverage reward and repeat-coverage penalty.
- `belief_search`: probability/belief map weighted search.
- `priority_inspection`: weighted POI inspection with repeat and deadline penalties.
- `connectivity_expansion`: cooperative expansion while maintaining multi-hop connectivity to base.
- `dynamic_target_escort`: waypoint-level dynamic target monitoring/escort.
- `target_interception`: route-branch interception and containment over future target belief.

Connectivity is a standalone mission type, not just a global constraint. The reward explicitly prevents "all UAVs stay near base" from becoming a high-score solution.

## Observation And State

Each agent observation includes:

- `self_state`
- `neighbor_relative_states`
- `task_field`
- `candidate_waypoints`
- `candidate_features`
- `candidate_mask`
- `task_id`
- `global_summary`
- `all_uav_states`
- `comm_adjacency`

The centralized critic state is available through:

```python
env.get_global_state()
```

It includes global task field, all UAV states, all candidate waypoints, candidate masks, communication adjacency, task id, global info, and agent mask.

## MAPPO Waypoint

`MAPPOWaypointPolicy` uses a shared per-agent categorical actor:

- encodes the global task field with a CNN;
- encodes all UAV states with a shared agent encoder and optional self-attention;
- encodes each candidate waypoint and candidate feature;
- outputs masked candidate logits `[B, N, K]`;
- samples one categorical waypoint index per UAV;
- returns per-agent `logprob [B, N]` and `entropy [B, N]`.

`MAPPOWaypointTrainer` uses:

- per-agent PPO ratio `exp(new_logprob_i - old_logprob_i)`;
- team advantage broadcast to agents in the first version;
- centralized value loss over `[B]`;
- separate `terminated` and `truncated` handling;
- bootstrap on truncation, no bootstrap on true termination;
- CSV metrics, TensorBoard, checkpoints, eval media, and `best_eval_summary.json`.

## Quick Start

Run waypoint API tests:

```bash
python -m pytest tests/test_waypoint_env_api.py
python -m pytest tests/test_waypoint_scenarios.py
python -m pytest tests/test_waypoint_rewards.py
python -m pytest tests/test_waypoint_mappo_policy.py
python -m pytest tests/test_mappo_waypoint_trainer.py
python -m pytest tests/test_waypoint_rendering.py
```

Run validation with random baseline, greedy heuristic baseline, 5-update MAPPO smoke, and one GIF:

```bash
python scripts/check/validate_waypoint_mappo.py \
  --tasks area_coverage belief_search connectivity_expansion \
  --num_agents 3 \
  --total_updates 5 \
  --eval_episodes 2 \
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

Each training run writes:

- `training_metrics.csv`
- `tensorboard/`
- `checkpoints/checkpoint_XXXX.pt`
- `checkpoints/checkpoint_best_eval.pt`
- `best_eval_summary.json`
- optional `media/eval_XXXX/*.gif` or `*.mp4`

View TensorBoard:

```bash
tensorboard --logdir outputs/training/mappo_waypoint --bind_all --port 6006
```

If `tensorboard` is not on PATH in the conda environment:

```bash
/opt/conda/bin/tensorboard --logdir outputs/training/mappo_waypoint --bind_all --port 6006
```

## Removed Centralized V1

The previous centralized Gymnasium-style benchmark has been removed from the working tree to keep the MARL branch focused on waypoint-level multi-agent training. It remains recoverable from git history if an old experiment needs to be reproduced.

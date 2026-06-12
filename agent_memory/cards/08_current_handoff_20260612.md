# Current Handoff Snapshot - 2026-06-12

This card is the current high-priority handoff for a new agent. It supersedes
old centralized-Gymnasium descriptions in earlier cards when they conflict
with the current code. Earlier cards remain useful for historical context.

## Current Project Identity

Wayffusion is now a waypoint-level multi-UAV MARL mission suite.

Current mainline facts:

- Main environment: `envs/waypoint_marl_env.py::WaypointMultiUAVEnv`.
- API style: PettingZoo-like parallel semantics at the environment level.
- Each UAV is an agent in the environment.
- Policy action interface: final waypoint action, not candidate waypoint index.
- Main learning route: Direct MAPPO with shared per-agent actor and centralized critic.
- Default policy for current specialist work: `policy_class: direct_waypoint`.
- UVFA extension exists as `policy_class: direct_waypoint_uvfa`, but it is not yet a performance conclusion.
- Dynamics backend: real vendored OpenAI MPE core through `MPECoreBackend`.
- Default MPE source: `third_party_openai_mpe`.
- Self-written `MPEParticleBackend` and `KinematicPointBackend` are removed from supported paths.
- BC is not part of the current Direct MAPPO specialist/benchmark route.
- Connectivity hard action filter may exist as an engineering safety option, but it is not the preferred training method for cross-task generalist work.

## Current Task Suite

Primary waypoint mission tasks:

| Task | Current role |
|---|---|
| `area_coverage` | Area coverage specialist/generalist task |
| `belief_search` | Belief/probability-map search task |
| `priority_inspection` | Weighted POI inspection task |
| `connectivity_expansion` | Connected coverage/relay expansion task |
| `dynamic_target_escort` | Multi-target escort/monitoring task |
| `target_interception` | Multi-target route/interception task |

The older `goal_nav`, `coverage`, `formation`, and `risk_nav` centralized
tasks are historical and should not be treated as the current mainline.

## Key Architecture Changes Already Done

### Waypoint Environment

- Rebuilt the project around `WaypointMultiUAVEnv`.
- External action is `dict[str, np.ndarray]`, one final waypoint per UAV.
- `action_space(agent_id)` is a Box waypoint space.
- Candidate waypoint index is not the main environment action anymore.
- Candidate generation moved to policy/planner side.
- `candidate_index_legacy` is only legacy/debug.

### Policy Families

- `DirectWaypointPolicy` directly outputs waypoint delta/final waypoint.
- `CandidateSelectionWaypointPolicy` is preserved as an alternative policy family, not the current specialist route.
- `PolicyOutput` separates:
  - `env_action`: final waypoint sent to env;
  - `train_action`: policy-internal action;
  - `logprob`, `entropy`, `value`, `aux`.
- `UVFADirectWaypointPolicy` adds mission-goal conditioning while preserving direct waypoint action semantics.

### MAPPO Trainer

- `MAPPOWaypointTrainer` supports generic `env_action` plus policy-internal `train_action`.
- PPO ratio/logprob are per-agent, not joint-action logprob.
- Team advantage is broadcast to agents.
- Terminated/truncated handling was corrected:
  - terminated disables value bootstrap;
  - truncated allows value bootstrap;
  - `done_for_reset` prevents GAE from propagating across episodes.
- Direct action clipping diagnostics were added:
  - `delta_raw_norm_mean`
  - `delta_raw_norm_max`
  - `delta_mean_norm_mean`
  - `clipped_delta_norm_mean`
  - `clipped_delta_norm_max`
  - `delta_clip_fraction`.

### MPE Core Backend

- The supported backend is `mpe_core`.
- The backend calls real MPE `World.step()` from the vendored OpenAI MPE core.
- No automatic fallback to `mpe2`.
- If using a different MPE source, it must be selected manually and documented.

### Domain Randomization

Added lightweight domain randomization in the existing `configs/env/waypoint_missions.yaml`.

Randomized:

- `base_position`
- spawn initial positions
- no-fly-zone layout
- belief peaks / target sampling
- POI position/weight/deadline

Not randomized:

- sensor radius
- communication radius
- base communication radius
- max speed
- adapter parameters
- dynamics parameters
- max delta / log std
- number of agents

Eval randomization supports:

- `inherit`
- `fixed`
- `random`
- `both`

When `both` is active, metrics are logged as fixed/random splits plus
generalization gap.

## Output Directory Contract

Current output rules:

| Output type | Required root |
|---|---|
| Debug, tuning, smoke, profiling | `outputs/debug/` |
| Formal training | `outputs/training/` |
| Pure checkpoint evaluation | `outputs/eval/` |

Direct MAPPO specialist debug layout:

```text
outputs/debug/mappo_direct_specialists/<phase_name>/<runtime>/<task_name>/trialXX/s<seed>/
```

Formal specialist layout:

```text
outputs/training/mappo_direct_specialists/<runtime>/<task_name>/s<seed>/
```

Pure evaluation layout:

```text
outputs/eval/mappo_direct_specialists/<runtime>/<task_name>/s<seed>/
```

Run names should be short:

- `s0`
- `s1`
- `trial01_s0`

Full hyperparameters belong in snapshots/reports, not in run names.

Future experiments should save media by default unless the user explicitly
turns it off.

## Major Experiment History

### Direct MAPPO Specialist Debugging

The project moved to pure Direct MAPPO specialists:

- no BC warm start;
- no candidate-selection fallback;
- no environment API rewrite;
- direct policy outputs waypoint action.

Initial observations:

- `belief_search` was the easiest task and reached high short-debug success.
- `area_coverage` showed learning trend but often plateaued around coverage `0.47-0.48`.
- `priority_inspection` improved but plateaued near weighted completion `0.55`.
- `connectivity_expansion` needed the most work; easy curriculum could pass, target config was unstable.
- Very small `approx_kl` and `clip_frac=0` often indicated weak effective updates or clipped/overwide direct action distribution.

### Connectivity Soft-Constraint Phases

Hard action filter in Phase16 reduced violation to zero but was rejected as the main training method because it replaces actor actions after output.

Soft/learnable connectivity route:

| Phase | Purpose |
|---|---|
| Phase17 | connectivity margin penalty |
| Phase18 | action disconnect risk penalty |
| Phase19 | Lagrangian/CMDP-style connectivity cost |
| Phase20 | auxiliary connectivity / coverage prediction |
| Phase21 | connected-only / connected-dominant new coverage reward |
| Phase22 | graph/aux representation experiments |
| Phase23 | relaxed success and graph reward balancing |
| Phase24 | connected-only reward base |
| Phase25 | tight connected curriculum |
| Phase26 | formal connected target |
| Phase27/28 | formal safety/stability finetunes |
| Phase31 | training-seed reproducibility for the Phase24 -> Phase25 -> Phase26 chain |

Important connectivity conclusion:

- Phase26/Phase29 showed seed0 can produce a credible soft-constraint solution.
- Phase31 was designed to test whether the Phase24 -> Phase25 -> Phase26 curriculum is reproducible across training seeds.
- Do not claim connectivity is fully frozen unless Phase31-style multi-seed reproducibility passes.

Connectivity formal success condition:

```text
coverage_ratio >= success_coverage
effective_explored_radius >= success_radius
connectivity_violation_rate <= max_violation_rate
```

Formal target currently:

```text
success_coverage: 0.55
success_radius: 0.45
max_violation_rate: 0.10
```

`connectivity_violation_rate` is the fraction of episode/step mass where UAVs
are not connected back to the base. It is not a hard action filter count.

### Swarm30 v1 Benchmark

Implemented frozen `swarm30_v1` benchmark scaffold.

Tasks:

- area coverage
- belief search
- priority inspection
- connectivity expansion
- dynamic target escort
- target interception

Tracks:

- `standardized_specialists`
- `tuned_specialists`
- `generalist`

Protocol:

- specialist budget: 2000 updates per task/seed;
- generalist budget: 12000 updates per seed;
- seeds: `0,1,2`;
- scale eval: `N=4,10,20,30`;
- deterministic final eval:
  - 100 randomized episodes;
  - 20 fixed episodes;
  - saved media.

Scale profiles:

| UAV | map size | grid | max steps | POI | belief peaks | dynamic targets |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.00 | 32 | 80 | 8 | 3 | 1 |
| 10 | 1.58 | 56 | 128 | 20 | 8 | 2 |
| 20 | 2.24 | 72 | 180 | 40 | 15 | 4 |
| 30 | 2.74 | 88 | 220 | 60 | 23 | 5 |

Sensor radius, comm radius, max speed, waypoint delta, obstacle physical size,
and task physical distances remain absolute map units.

Benchmark scripts:

- `scripts/benchmarks/swarm30/run_benchmark.py`
- `scripts/benchmarks/swarm30/evaluate.py`
- `scripts/benchmarks/swarm30/build_report.py`
- `scripts/benchmarks/swarm30/launch_tmux.sh`
- `scripts/benchmarks/swarm30/status.sh`
- `scripts/benchmarks/swarm30/evaluate_n4_specialists_on_n30.sh`

The formal `swarm30_v1` benchmark was started, but later stopped for runtime
profiling. It should be relaunched only after deciding whether to apply
non-semantic speed optimizations or keep the exact frozen protocol.

### N4 Specialists on N30 Evaluation

Added pure evaluate launcher:

```text
scripts/benchmarks/swarm30/evaluate_n4_specialists_on_n30.sh
```

Default source:

```text
outputs/training/mappo_direct_specialists/20260607_1237/<task>/s0/checkpoints/checkpoint_best_eval.pt
```

Output:

```text
outputs/eval/mappo_direct_specialists/<runtime>/<task>/s0/
```

This is a zero-shot N=4-to-N=30 evaluation, not a trained N=30 result and not
the full six-task benchmark.

The run was started and later stopped during runtime profiling. The
`area_coverage` section completed; later tasks may be partial depending on the
output directory.

### UVFA / Schaul et al. 2015 Work

Implemented a first UVFA adaptation:

- episode-level `mission_goal[3]`;
- `goal_mask[3]`;
- train/interpolation/formal goal sets for four core tasks;
- `direct_waypoint_uvfa` with separate goal encoder;
- critic uses state embedding, goal embedding, and elementwise product.

Added:

- `configs/policy/uvfa/direct_waypoint_uvfa.yaml`
- `configs/policy/uvfa/direct_waypoint_uvfa_debug.yaml`
- `scripts/debug/phase32_uvfa_goal_conditioning.py`
- `docs/uvfa_waypoint_mappo_zh.md`

Phase32 was queued but later stopped during runtime profiling. Do not claim
UVFA improves performance until Phase32 or a later matched comparison finishes.

## Runtime Profiling Conclusions

Profiling date: 2026-06-11.

Outputs:

```text
outputs/debug/perf_profile/swarm30_runtime_breakdown/20260611_1526/
outputs/debug/perf_profile/env_step_cprofile/20260611_1526_connectivity/
```

Diagnostic scripts:

```text
scripts/benchmarks/swarm30/profile_runtime_breakdown.py
scripts/benchmarks/swarm30/profile_env_step_cprofile.py
```

Measured conclusion:

- wall-clock bottleneck is CPU-side environment stepping and render;
- PPO/network update is not the bottleneck.

Approximate 100-update benchmark-period proportions:

| Component | Share |
|---|---:|
| rollout collect / env.step | `86.2%` |
| eval stepping | `9.1%` |
| render | `3.7%` |
| GIF write | `0.2%` |
| PPO backward/update | `0.8%` |

Measured references:

| Metric | Value |
|---|---:|
| N=30 area single-env step | `0.271s` |
| N=30 belief single-env step | `0.221s` |
| N=30 connectivity single-env step | `0.422s` |
| N=30 process batch collect, 8 envs | `0.453s/batch step` |
| process batch policy forward | `0.011s` |
| process batch env step | `0.442s` |
| render frame | `1.14s` |
| GIF write frame | `0.068s` |
| fake PPO minibatch backward | `0.066s` |

`connectivity_expansion, N=30` cProfile hotspots:

- `MPECoreBackend.step`;
- vendored OpenAI MPE `World.step`;
- `get_collision_force`;
- `communication_graph_for_positions`;
- `connectivity_expansion.compute_rewards`;
- repeated `_build_agent_info` / `get_metrics`.

Non-semantic speedup priorities:

1. Cache `task_metrics` once per env step; do not recompute team-level metrics for every agent info.
2. Cache connectivity graph, connected masks, margin stats, and effective radius inside a step.
3. Reduce intermediate media; keep full GIFs for final evaluation.
4. Lower `max_parallel` when CPU is oversubscribed.
5. Do not run heavy evaluate concurrently with heavy training.

Semantic-changing speedups must be separate ablations:

- reducing MPE substeps;
- disabling collision/contact force during training;
- reducing N=30 grid/horizon/eval episodes.

## Current Important Files

Architecture:

- `envs/waypoint_marl_env.py`
- `envs/waypoint/control/dynamics_backends.py`
- `policies/direct_waypoint_policy.py`
- `algorithms/mappo_waypoint.py`
- `utils/waypoint_vector_env.py`
- `utils/waypoint_evaluation.py`

Configs:

- `configs/env/waypoint_missions.yaml`
- `configs/eval/waypoint_eval.yaml`
- `configs/policy/direct_specialists/`
- `configs/policy/direct_specialists_v2/`
- `configs/policy/benchmarks/`
- `configs/policy/uvfa/`
- `configs/benchmarks/swarm30_v1.yaml`

Docs:

- `docs/waypoint_environment_guide_zh.md`
- `docs/swarm30_v1_benchmark_zh.md`
- `docs/swarm30_runtime_profile_zh.md`
- `docs/uvfa_waypoint_mappo_zh.md`
- `docs/mpe_core_backend_zh.md`

Scripts:

- `scripts/train_mappo_waypoint.py`
- `scripts/benchmarks/swarm30/run_benchmark.py`
- `scripts/benchmarks/swarm30/evaluate.py`
- `scripts/benchmarks/swarm30/evaluate_n4_specialists_on_n30.sh`
- `scripts/debug/phase32_uvfa_goal_conditioning.py`

## Completed Work

- Rebuilt current mainline around waypoint-level MARL.
- Implemented Direct MAPPO waypoint training.
- Switched dynamics to real vendored MPE core.
- Added domain randomization and fixed/random/both eval modes.
- Added specialist/debug/formal output layout rules.
- Added pure evaluation output root `outputs/eval/`.
- Added N4-to-N30 evaluation launcher.
- Added Swarm30 benchmark scaffold.
- Added UVFA scaffold.
- Added runtime profiling and diagnosis.
- Updated docs and memory repeatedly to reflect current state.

## Known Incomplete Work

- `swarm30_v1` benchmark was stopped during profiling and does not have a final leaderboard yet.
- N4-to-N30 evaluate was stopped during profiling; `area_coverage` completed, later tasks may be partial.
- Phase32 UVFA comparison was stopped before producing performance conclusions.
- Connectivity reproducibility conclusion should only be trusted if Phase31 aggregate results are complete and verified.
- Area / priority / belief / connectivity Direct specialists still need robust multi-seed final validation if the goal is to freeze them.
- Dynamic escort and target interception need more specialist tuning; current benchmark configs are scaffolds, not final proven specialists.
- Speed optimizations have been diagnosed but not implemented.

## Recommended Next Agent Route

If the user asks for speed:

1. Implement non-semantic metrics/cache optimization in a new speed phase.
2. Re-run runtime profiling.
3. Only then relaunch `swarm30_v1` or N4-to-N30 evaluate.

If the user asks for performance:

1. Do not mix with speed changes unless explicitly allowed.
2. Continue Direct MAPPO specialist tuning, especially area and priority.
3. For connectivity, rely on Phase24 -> Phase25 -> Phase26 curriculum and verify multi-seed reproducibility.
4. Treat hard connectivity filter only as safety fallback, not as training solution.

If the user asks for UVFA:

1. Run Phase32 matched comparison after resource cleanup.
2. Compare seen/interpolation/formal goal splits.
3. Do not claim UVFA benefit without multi-seed metric improvement.

If the user asks for benchmark:

1. Decide whether to preserve exact `swarm30_v1` frozen protocol or create a new version after speed optimizations.
2. Use `outputs/training/benchmarks/swarm30_v1/<runtime>/` for formal benchmark training.
3. Use `outputs/eval/` for evaluate-only checkpoint testing.

## Current Process State at This Handoff

During runtime profiling, the active `swarm30_v1` benchmark, N4-to-N30 evaluate,
and Phase32 queue were stopped to avoid measurement interference. No outputs
were intentionally deleted.

Before launching any long job, check:

```bash
tmux ls
ps -eo pid,ppid,etimes,etime,stat,args | rg 'swarm30|train_mappo_waypoint|evaluate.py|phase32'
```

## Caution About Old Memory Cards

Earlier cards such as `01_system_contract.md` still describe the old
centralized Gymnasium-style contract. That is historical context. Current
work should use this card plus the latest code/docs as source of truth.

# Recent Applied Modifications

## Theme A: Evaluation-time visualization and recording

Implemented:

- `headless` / `no-headless` toggles on training entrypoints
- shared evaluation recording for `PPO`, `SAC`, `TD3`, and `BC`
- unified `gif/mp4` export through `utils/evaluation.py`
- `eval_media_dir` tracking in evaluation outputs

Design choice:

- live rendering and recording are limited to evaluation episodes
- the rollout collector is not rendered frame by frame

## Theme B: Stable render path for media generation

Implemented:

- `rgb_array` rendering uses a NumPy image path
- `human` mode keeps interactive visualization

Reason:

- recording should not depend on a GUI backend
- the previous matplotlib backend path was fragile on some local runs

## Theme C: Timestamped training output roots

Implemented:

- all training outputs now go to `outputs/training/<algorithm>/<timestamp>/<run_name>/`
- the timestamp layer is built through a shared helper

Reason:

- repeated runs no longer overwrite each other
- per-algorithm runs are easier to audit and compare

## Theme D: Sibling directories for checkpoints and snapshots

Implemented:

- model weights are stored under `checkpoints/`
- training-start material is stored under `snapshot/`

Reason:

- start state and trained state are now explicitly separated
- `media/`, `checkpoints/`, `snapshot/`, and `tensorboard/` live at the same level

## Theme E: PPO stop contract

Implemented:

- `scripts/train_ppo.py` supports both `--total_updates` and `--target_episodes`
- the multitask PPO launcher defaults to update-based stopping

Reason:

- supports classic PPO training by update count
- still keeps episode-budget control when needed

## Theme F: Multitask PPO launcher

Implemented:

- `configs/policy/ppo_cnn_deepsets_multitask_20k.yaml`
- `scripts/run_multitask_ppo_20k.ps1`

Current default:

- task interleaving over `goal_nav coverage formation risk_nav`
- centralized PPO
- `CNN + DeepSets`
- update-based stopping

## Theme G: Expanded tests

Added coverage for:

- media export
- non-headless human render path
- timestamped training paths
- snapshot and recursive checkpoint lookup
- PPO target-episode early stop
- agent memory module loading

## Theme H: TensorBoard and live console feedback

Implemented:

- shared TensorBoard event writing for `PPO`, `SAC`, `TD3`, and `BC`
- periodic console progress lines during training
- final evaluation summaries written to TensorBoard
- per-run `tensorboard/` directory under every training output

Design choice:

- TensorBoard is attached to the same run root as checkpoints and media
- console logging is emitted on the trainer's native progress axis: update, step, or epoch

## Theme I: Agent memory synchronization rule

Implemented:

- `agent_memory/README.md` now states a mandatory synchronization rule
- new `maintenance_protocol` card formalizes the process contract

Rule:

- every code, config, doc, CLI, or output-contract change must update `agent_memory/` in the same task

## Theme J: Core credibility repair for scaling, policy math, and ablation semantics

Implemented:

- `waypoint_controller(...)` now clips against `map_size`, not a hard-coded unit square
- `density_preserving` runs can now move into coordinates larger than `1.0`
- `MLP`, `CNN + DeepSets`, and `Attention` policies now use tanh-squashed Gaussian action sampling with corrected log-prob evaluation
- formation radius mismatch is now represented as a positive error with an explicit negative penalty term
- canonical ablation naming is now `no_spatial_field`, while `task_id_only` remains a deprecated alias
- reference normalization now marks unstable anchors and preserves raw-return / anchor metadata more explicitly

Reason:

- these were benchmark-trust issues, not cosmetic cleanups
- cross-`N` evaluation and policy-gradient math are now aligned with the intended centralized benchmark contract

## Trusted conclusion

These changes are no longer isolated patches. They now exist consistently in:

- code
- docs
- tests
- output contract
- operational memory

That combination should be treated as the current engineering source of truth.

## Theme K: Chinese config reference and template config directory

Implemented:

- `docs/config_reference_zh.md` was refreshed into a script-oriented Chinese config guide
- `configs/examples/` now contains ready-to-copy template files for env, policy, and eval categories
- a lightweight parser test now guards the template directory against drift

Reason:

- users now have a single Chinese entrypoint for understanding config categories
- new experiments can start from stable templates instead of mutating old run configs blindly

## Theme L: PPO evaluation now runs per-task instead of only mixed-sampler averages

Implemented:

- `utils/evaluation.py` now provides a fixed-task `evaluate_policy_per_task(...)` helper
- PPO periodic evaluation now records per-task metrics plus an overall summary
- PPO final evaluation now writes per-task rows and an `overall` row into `eval_metrics.csv`
- TensorBoard final-eval namespaces now include per-task paths such as `ppo/final_eval/N4/goal_nav/...`

Reason:

- a multi-task training run should not hide a collapsed task behind one mixed average
- per-task regressions are now visible during training and in saved evaluation tables

## Theme M: Per-task evaluation propagated to eval scripts and off-policy trainers

Implemented:

- `scripts/evaluate_policy.py` now writes per-task rows plus an `overall` row
- `scripts/evaluate_scaling.py` now writes per-task rows plus an `overall` row for every evaluated `N`
- `SAC` and `TD3` periodic evaluation now flatten per-task metrics into training records
- `SAC` and `TD3` final eval now mirror PPO with per-task TensorBoard namespaces and per-task `eval_metrics.csv`
- variable-`N` PPO periodic evaluation now monitors all requested `N`, not just the first one
- restored minimal default configs for `configs/policy/sac_cnn_deepsets.yaml` and `configs/policy/td3_cnn_deepsets.yaml` so the touched training scripts keep a valid default entrypoint

Reason:

- trust-worthy evaluation should expose both task-level failures and cross-`N` failures
- fixed-task breakdowns are now consistent across online training logs, eval scripts, and saved CSV outputs

## Theme N: Ubuntu Docker server training adaptation

Implemented:

- added `requirements-server.txt` for PyTorch Docker images that already provide CUDA-enabled torch
- kept the existing `requirements.txt` as the generic environment file and added the missing `psutil` dependency there
- added `docs/server_training_zh.md` with Docker launch, proxy verification, dependency installation, GPU checks, smoke tests, TensorBoard forwarding, long-run training, and common troubleshooting
- added `scripts/server/check_server_env.py` for offline server environment diagnostics
- added `scripts/server/smoke_train_ppo.sh` for a short headless PPO training and GIF-recording check
- added `scripts/server/start_tensorboard.sh` for the canonical `outputs/training` TensorBoard logdir
- added a README entry pointing server users to the dedicated guide
- extended `.gitignore` so server-side test runs do not leave Python cache files as untracked worktree noise

Reason:

- the target server image is `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`, so server installs must avoid upgrading torch through pip
- server training must remain headless and preserve the existing `outputs/training/<algorithm>/<timestamp>/<run_name>/` artifact layout
- `utils/profiling.py` imports `psutil`, so server and generic dependency files now declare it explicitly

## Theme O: Long all-task PPO training launcher

Implemented:

- added `scripts/run_ppo_all_tasks_long.sh` as the Linux/Ubuntu launcher for long PPO training across `goal_nav coverage formation risk_nav`
- default run uses `configs/policy/ppo_cnn_deepsets_multitask_20k.yaml`, `total_updates=2000`, `target_episodes=0`, `eval_episodes=5`, and `agent_counts=4`
- periodic evaluation and checkpointing follow the config's PPO `eval_interval` of 25 updates
- GIF recording is enabled with `record_eval_episodes=1` and `record_interval=4`, so media is saved every 100 updates plus final evaluation media
- script defaults to headless training, TensorBoard enabled, `MPLBACKEND=Agg`, and `CUDA_VISIBLE_DEVICES=0`, while allowing shell environment overrides
- script has a top-level editable `DEFAULT_*` configuration block so normal parameter changes can be made inside the `.sh` file
- README now points Linux users to the long all-task launcher

Reason:

- server users need one direct command for long all-task PPO training that preserves the existing artifact contract
- checkpoints, TensorBoard logs, evaluation CSVs, and GIFs continue to live under `outputs/training/ppo/<timestamp>/<run_name>/`

## Theme S: goal_nav remaining-goal observability and conservative PPO diagnostics

Implemented:

- `tasks/goal_nav.py` and `tasks/risk_nav.py` now rebuild the `goal_reward` field from only unreached goals each step instead of keeping a static reset-time goal map
- `tests/test_rewards_basic.py` now checks that goal-progress shaping ignores goals already marked as reached
- `algorithms/ppo.py` now clamps `log_std` both at trainer init and after checkpoint load, and logs rollout reward stats, terminal metrics, KL/clip diagnostics, ratio mean, gradient norm, explained variance, and policy std summaries
- `utils/evaluation.py` now accumulates episode-level reward components into eval records for deeper reward audits

Reason:

- goal-nav specialists were repeatedly re-attracted to already completed goals
- conservative PPO fine-tunes need explicit `log_std` enforcement after loading BC checkpoints
- long-horizon debug work needed richer diagnostics than `eval_reward` / `eval_success_rate` alone

## Theme T: goal_nav specialist repair via success-trajectory BC, DAgger BC, and ultra-strict PPO

Implemented:

- added goal-nav debug configs:
  - `configs/policy/debug_bc_goal_nav.yaml`
  - `configs/policy/debug_bc_goal_nav_success.yaml`
  - `configs/policy/debug_ppo_goal_nav_controlled.yaml`
  - `configs/policy/debug_ppo_goal_nav_bc_finetune.yaml`
  - `configs/policy/debug_ppo_goal_nav_bc_finetune_strict.yaml`
  - `configs/policy/debug_ppo_goal_nav_dagger_finetune_ultra_strict.yaml`
- `scripts/train_bc.py` now accepts `--init_checkpoint` so BC can be warm-started from previous BC or PPO checkpoints during DAgger-style repair loops
- `outputs/debug_long/20260528_035433/` now records the success-only BC run, BC+PPO fine-tune, DAgger dataset collection, and DAgger BC retrain

Current best evidence:

- success-only DAgger BC run:
  - `outputs/training/bc/20260528_051955/debug_bc_goal_nav_success_goal_nav_N4_multi_channel_field_plus_task_id/`
  - final `success_rate_mean=0.8`
  - final `goal_coverage_ratio_mean=0.9125`
  - final `collision_rate_mean=0.0646`
- ultra-strict PPO fine-tune from the DAgger BC checkpoint:
  - `outputs/training/bc_ppo/20260528_goalnav_dagger_finetune_ultra_strict/goalnav_dagger_finetune_ultra_strict/`
  - final `success_rate_mean=0.8`

## Theme AC: coverage centralized per-agent waypoint allocation experiments

Implemented:

- `CNNDeepSetsPolicy` now has an optional `coverage_utility_slot_head`, disabled by default.
- The new head reads the centralized multi-channel field and adds a small per-agent waypoint bias from a pooled coverage utility grid.
- The head keeps the same action contract: output is still `[B, N, 2]`, log-prob remains one joint log-prob per env, and `agent_mask` still zeros padded agents.
- `policies/__init__.py` forwards the new config fields.
- `tests/test_variable_policies.py` now checks the utility slot head and cross-task compatibility.
- New configs:
  - `configs/policy/debug_bc_coverage_utilityslot.yaml`
  - `configs/policy/debug_ppo_coverage_utilityslot_ultra_strict.yaml`

Diagnostic result:

- Direct untrained utility-slot behavior was bad (`coverage_ratio_mean≈0.314`, high collision), so the head is not a standalone heuristic.
- BC on `coverage_success_from_phase24_best_round4.npz` produced:
  - run: `outputs/training/bc/20260530_025202/debug_bc_coverage_utilityslot_coverage_N4_multi_channel_field_plus_task_id/`
  - `success_rate_mean=0.10`
  - `coverage_ratio_mean≈0.666`
  - `collision_rate_mean≈0.000375`
- PPO fine-tune `phase27_coverage_utilityslot` was stopped at update 40 because eval stayed at `success_rate=0.0`.

Conclusion:

- A fixed coverage utility bias can preserve low collision after BC, but it destabilizes PPO fine-tuning.
- Do not continue this exact fixed-bias branch unless the bias is made learnable or much weaker.

## Theme AD: coverage expert-v3 local frontier teacher

Implemented:

- `scripts/debug_long/generate_coverage_expert_v2_dataset.py` now also exposes `CoverageExpertV3` and `--expert v3`.
- `CoverageExpertV3` uses local remaining-demand frontier selection plus short-range collision-avoidance repulsion.
- `scripts/debug_long/generate_success_expert_dataset.py` now supports:
  - `--teacher coverage_expert_v2`
  - `--teacher coverage_expert_v3`

Diagnostic result:

- `CoverageExpertV2` diagnostic: `success=0.0`, `coverage_ratio_mean≈0.538`, high collision.
- local frontier without repulsion: `success≈0.0625`, `coverage_ratio_mean≈0.645`, but collision was too high.
- `CoverageExpertV3` with repulsion: `success≈0.07`, `coverage_ratio_mean≈0.622`, `collision≈0.0041`.

Conclusion:

- `CoverageExpertV3` is not a solved expert, but it is a better source of successful coverage trajectories than `CoverageExpertV2`.
- Next coverage branch should collect success-only v3 trajectories, then train BC/PPO from that dataset.

## Theme AE: coverage permutation-invariant BC breakthrough

Implemented:

- `algorithms/bc.py` now supports optional `permutation_invariant_loss`.
- The loss works in waypoint space, not raw action space:
  - predicted waypoint = current position + predicted action * `bc_waypoint_step`
  - expert waypoint = current position + expert action * `bc_waypoint_step`
  - all target waypoint assignments are enumerated for small N, and the minimum assignment loss is used
- `scripts/train_bc.py` injects `bc_waypoint_step` from the env config into BC config.
- Added `tests/test_bc_permutation_loss.py`.
- Added `configs/policy/debug_bc_coverage_successonly_perm.yaml`.
- Added `scripts/debug_long/merge_expert_datasets.py` for atomic NPZ dataset merges.

Data generated:

- `outputs/debug_long/20260530_coverage_expert_v3/coverage_expert_v3_success_N4_e40.npz`
  - 40 successful episodes
  - 7354 samples
  - 665 attempts
  - mean successful collision rate: `0.0`
- `outputs/debug_long/20260530_coverage_expert_v3/coverage_success_round4_plus_v3_e40.npz`
  - merged samples: 17162
  - inputs:
    - phase24 round4 success data: 9808 samples
    - v3 success data: 7354 samples

Key result:

- permutation-invariant BC run:
  - `outputs/training/bc/20260530_032314/debug_bc_coverage_successonly_perm_coverage_N4_multi_channel_field_plus_task_id/`
  - `success_rate_mean=0.25`
  - `coverage_ratio_mean≈0.717`
  - `collision_rate_mean≈0.0079`
  - `return_mean≈10.61`

Conclusion:

- Coverage BC was materially harmed by fixed-index expert action matching.
- The current best coverage specialist checkpoint is now the permutation-invariant BC checkpoint, pending PPO fine-tune.
  - final `goal_coverage_ratio_mean=0.9425`
  - final `collision_rate_mean=0.0275`
  - final `return_mean=10.1225`

Reason:

- naive PPO from scratch on goal_nav showed nonzero intermediate improvements but repeatedly fell back to `success_rate≈0`
- BC from generic heuristic data improved alignment but not enough robustness
- DAgger-style relabeling on learner states produced the first specialist line that stayed in a high-success region under PPO

## Theme U: coverage specialist controlled-PPO branch and BC probe

Implemented:

- added coverage debug configs:
  - `configs/policy/debug_ppo_coverage_controlled.yaml`
  - `configs/policy/debug_ppo_coverage_bc_finetune.yaml`
  - `configs/policy/debug_bc_coverage.yaml`
  - `configs/policy/debug_ppo_coverage_long_horizon.yaml`
- generated heuristic expert dataset:
  - `outputs/datasets/expert_coverage_N4_phase3_e80.npz`
- trained BC coverage probe:
  - `outputs/training/bc/20260528_072926/debug_bc_coverage_coverage_N4_multi_channel_field_plus_task_id/`

Current status:

- best PPO controlled coverage line so far comes from:
  - `outputs/training/ppo/20260528_phase2_coverage_controlled/phase2_coverage_controlled/`
  - `eval_coverage_coverage_ratio≈0.447`
  - `eval_reward≈-1.29`
  - `eval_success_rate=0.0`
- resumed controlled PPO under:
  - `outputs/training/bc_ppo/20260528_phase3_coverage_controlled_resume/phase3_coverage_controlled_resume/`
  - resumed improving after an early dip, but still had `eval_success_rate=0.0`
- BC coverage probe underperformed the PPO controlled line:
  - final `coverage_ratio_mean≈0.406`
  - final `success_rate_mean=0.0`
  - final `collision_rate_mean≈0.211`

Reason:

- coverage is not “randomly failing”; PPO can learn meaningful partial coverage behavior
- however, heuristic BC alone does not currently beat the best PPO controlled checkpoint
- next coverage work should prioritize long-horizon PPO tuning over generic heuristic BC

## Theme V: goal_nav convergence confirmation and coverage credit-assignment branch

Implemented:

- confirmed the DAgger BC -> ultra-strict PPO chain all the way through final eval:
  - `outputs/training/bc_ppo/20260528_goalnav_dagger_finetune_ultra_strict/goalnav_dagger_finetune_ultra_strict/eval_metrics.csv`
- added a faster coverage long-horizon / credit-assignment PPO config:
  - `configs/policy/debug_ppo_coverage_creditassign.yaml`
- launched a resumed coverage run from the best phase-3 checkpoint:
  - `outputs/training/bc_ppo/20260528_phase4_coverage_creditassign/phase4_coverage_creditassign/`

Current best confirmed goal_nav result:

- final `return_mean=10.1225`
- final `normalized_score_mean=1.0496`
- final `success_rate_mean=0.8`
- final `collision_rate_mean=0.0275`
- final `goal_coverage_ratio_mean=0.9425`

Current coverage status:

- heuristic BC probe stays weaker than PPO:
  - `coverage_ratio_mean≈0.406`
  - `success_rate_mean=0.0`
  - `collision_rate_mean≈0.211`
- coverage credit-assignment PPO currently has early eval points:
  - update 20: `eval_reward≈-0.551`, `coverage_ratio≈0.450`, `collision≈0.0899`, `success=0.0`
  - update 40: `eval_reward≈-1.008`, `coverage_ratio≈0.449`, `collision≈0.0944`, `success=0.0`

Reason:

- goal_nav is now strong enough to be treated as the current canonical specialist PPO success path
- coverage still needs iterative tuning, but the credit-assignment branch is now the best active mainline to continue from

## Theme W: coverage expert-v2 dataset and BC warm-start branch

Implemented:

- added `scripts/debug_long/generate_coverage_expert_v2_dataset.py`
- added `configs/policy/debug_ppo_coverage_expertv2_finetune_strict.yaml`
- generated stronger coverage expert dataset:
  - `outputs/debug_long/20260528_coverage_expert_v2/coverage_expert_v2_N4_e40.npz`
- trained BC on that dataset:
  - `outputs/training/bc/20260528_081939/debug_bc_coverage_coverage_N4_multi_channel_field_plus_task_id/`
- launched strict PPO from the new BC checkpoint:
  - `outputs/training/bc_ppo/20260528_phase6_coverage_expertv2_finetune/phase6_coverage_expertv2_finetune/`

Current evidence:

- quick expert-v2 probe over 5 episodes beat the canonical heuristic:
  - heuristic: `coverage_ratio≈0.420`, `collision≈0.130`, `intrinsic≈0.389`
  - expert-v2: `coverage_ratio≈0.579`, `collision≈0.0865`, `intrinsic≈0.555`
- BC trained on expert-v2 data also beat both the canonical heuristic BC and the PPO plateau:
  - final `coverage_ratio_mean≈0.512`
  - final `collision_rate_mean≈0.150`
  - final `return_mean≈-0.685`
  - final `normalized_score_mean≈1.532`

Reason:

- coverage heuristic BC from the canonical heuristic was not strong enough
- a stronger spread-aware expert policy produced the first clear coverage BC warm-start that surpassed the previous PPO coverage-ratio plateau

## Theme X: coverage coordination-repulsion PPO branch

Implemented:

- added optional `coordination_repulsion_strength` support to `policies/cnn_deepsets_policy.py`
- threaded the new config field through `policies/__init__.py`
- added coverage experiment configs:
  - `configs/policy/debug_ppo_coverage_expertv2_finetune_ultra_strict.yaml`
  - `configs/policy/debug_ppo_coverage_expertv2_repulsion.yaml`
- added a minimal regression test:
  - `tests/test_variable_policies.py::test_cnn_deepsets_policy_supports_optional_coordination_repulsion`

Current evidence:

- the plain ultra-strict PPO fine-tune from the expert-v2 BC checkpoint still degraded to about `coverage_ratio≈0.486`
- enabling repulsion bias produced the first clear post-BC PPO jump:
  - `outputs/training/bc_ppo/20260528_phase8_coverage_repulsion/phase8_coverage_repulsion/`
  - update 20: `coverage_ratio≈0.578`
  - update 20: `collision_rate≈0.0015`
  - update 20: `eval_reward≈5.397`
  - success is still `0.0`, so this is not yet full task convergence

Reason:

- coverage PPO had repeatedly plateaued near the heuristic structure ceiling
- a small explicit anti-crowding bias appears to help the factorized per-agent actor preserve spatial spread instead of collapsing back into redundant overlap

## Theme Y: coverage repulsion branch follow-up with reward-focus resume

Implemented:

- added `configs/env/debug_coverage_reward_focus.yaml`
- resumed a repulsion PPO checkpoint under the reward-focus coverage env:
  - `outputs/training/bc_ppo/20260528_phase9_coverage_repulsion_rewardfocus/phase9_coverage_repulsion_rewardfocus/`

Current evidence:

- the plain repulsion PPO branch remains the strongest PPO coverage line so far:
  - peak `coverage_ratio≈0.664`
  - `collision≈0.001`
  - one eval point reached `success_rate=0.05`
- adding reward-focus on top of the repulsion checkpoint did not further increase the coverage ratio:
  - update 20 under phase 9: `coverage_ratio≈0.665`
  - update 20 under phase 9: `success=0.0`

Reason:

- coverage reward scaling alone is not the main remaining blocker once anti-crowding coordination is in place
- the remaining gap now looks more like “last-mile completion behavior” than generic reward sparsity

## Theme Z: coverage spatial-action-head specialist branch

Implemented:

- extended `CNNDeepSetsPolicy` with optional:
  - `use_spatial_action_head`
  - `spatial_action_strength`
- added `configs/policy/debug_bc_coverage_spatialhead.yaml`
- added `configs/policy/debug_ppo_coverage_spatialhead_ultra_strict.yaml`
- trained a stronger coverage BC warm-start with expert-v2 data and the new spatial action head:
  - `outputs/training/bc/20260528_092949/debug_bc_coverage_spatialhead_coverage_N4_multi_channel_field_plus_task_id/`
- launched PPO from that checkpoint:
  - `outputs/training/bc_ppo/20260528_phase10_coverage_spatialhead_ultra/phase10_coverage_spatialhead_ultra/`

Current evidence:

- spatial-head BC outperformed the prior coverage BC:
  - `coverage_ratio_mean≈0.661`
  - `success_rate_mean≈0.05`
  - `collision_rate_mean≈0.00094`
  - `return_mean≈8.659`
- spatial-head ultra-strict PPO then preserved and slightly improved that structure:
  - update 20: `success≈0.10`, `coverage_ratio≈0.667`, `collision≈0.000625`
  - update 40: `success≈0.10`, `coverage_ratio≈0.670`, `collision≈0.0003125`
  - update 60: `success≈0.10`, `coverage_ratio≈0.682`, `collision≈0.00025`
  - final eval:
    - `return_mean≈8.083`
    - `normalized_score_mean≈2.615`
    - `coverage_ratio_mean≈0.667`
    - `success_rate_mean=0.0`
    - `collision_rate_mean≈0.003`

Reason:

- coverage needed a stronger geometric inductive bias than plain per-agent decoder plus repulsion alone
- an explicit spatial action head helps the actor map each agent to a spatial target instead of only producing raw waypoint deltas

## Theme AA: coverage spatial-target suppression probe

Implemented:

- added non-parametric `spatial_target_suppression_strength` and `spatial_target_suppression_sigma` support inside the spatial action head
- added `configs/policy/debug_ppo_coverage_spatialhead_suppression.yaml`
- verified the new branch with `tests/test_variable_policies.py`

Current evidence:

- suppression branch did not beat the plain spatial-head PPO branch:
  - update 20: `success≈0.10`, `coverage_ratio≈0.655`, `collision≈0.00044`
  - update 40: `success≈0.05`, `coverage_ratio` and reward both slipped relative to the no-suppression branch

Reason:

- explicit soft suppression was a reasonable next guess for target diversity
- but in current form it underperforms the simpler spatial-head + repulsion branch, so it should not replace the current mainline

## Theme AB: interruption-safe debug artifact writing

Implemented:

- `scripts/debug_long/collect_success_policy_dataset.py`
- `scripts/debug_long/collect_dagger_dataset.py`
- `scripts/debug_long/generate_success_expert_dataset.py`
- `scripts/debug_long/generate_coverage_expert_v2_dataset.py`

now write datasets and JSON sidecars via temporary files plus atomic rename.

Reason:

- long-running debug jobs were being interrupted by connection drops or accidental thread closure
- partial `.npz` writes were leaving corrupted datasets such as `BadZipFile`
- atomic write semantics keep the final path either fully valid or absent

## Theme AC: risk_nav and formation specialist repair

Implemented:

- added risk-nav configs:
  - `configs/policy/debug_bc_risk_nav_success.yaml`
  - `configs/policy/debug_ppo_risk_nav_success_ultra_strict.yaml`
- added formation configs:
  - `configs/policy/debug_bc_formation_success.yaml`
  - `configs/policy/debug_ppo_formation_success_ultra_strict.yaml`
  - `configs/policy/debug_bc_formation_dagger.yaml`
- generated success-only datasets:
  - `outputs/debug_long/20260528_risk_nav_success/risk_nav_success_N4_stride2.npz`
  - `outputs/debug_long/20260528_formation_success/formation_success_N4_stride2.npz`
- generated formation success-policy dataset:
  - `outputs/debug_long/20260528_formation_success_policy/formation_success_from_phase14_u40_plus_fullheuristic.npz`

Current evidence:

- risk_nav:
  - BC on full heuristic-state dataset reached `success_rate_mean≈0.55`
  - ultra-strict PPO from that BC checkpoint stabilized around `success≈0.55~0.60`
  - final line:
    - `outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra/`
    - final `success_rate_mean=0.6`
    - final `goal_coverage_ratio_mean≈0.829`
    - final `collision_rate_mean≈0.049`
- formation:
  - success-only BC remained below heuristic
  - full-state BC improved but still lagged heuristic
  - ultra-strict PPO from the full-state BC checkpoint reached the first stable heuristic-level line:
    - update 40/60/80 held `success≈0.4`
    - final eval fell back to `success=0.3`
  - formation DAgger BC and formation success-policy BC did not beat that PPO branch

Reason:

- risk_nav was close enough to goal_nav that the BC -> strict PPO recipe transferred well
- formation responds to the current slot-aware actor and PPO warm-start, but its best result is still a best-checkpoint result rather than a final stabilized result

## Theme AD: human-readable summary doc and formation success-policy PPO branch

Implemented:

- added human-readable summary doc:
  - `docs/specialist_training_repair_summary_zh.md`
- launched formation success-policy PPO branch:
  - `outputs/training/bc_ppo/20260528_phase16_formation_successpolicy_ultra/phase16_formation_successpolicy_ultra/`

Current evidence:

- the doc now records:
  - what changed
  - why each code/config change was made
  - which task is currently repaired vs partially repaired
  - why conservative PPO / BC / DAgger / structure bias were technically justified
- formation success-policy PPO first eval currently holds:
  - update 20: `success≈0.4`
  - `formation_error≈0.091`
  - `collision≈0.0045`

Reason:

- the user explicitly requested a human-facing written summary under `docs/`
- frequent connection drops mean partial-but-important experimental state should be written to memory before runs finish

## Theme AE: coverage success-trajectory reinforcement follow-up

Implemented:

- collected additional coverage success-policy datasets:
  - deterministic from `phase10` update 60
  - deterministic from `phase10` update 40
  - noisy (`action_noise_std=0.05`) from `phase10` update 60
- tested several follow-up branches:
  - `phase15_coverage_successbc_ultra`
  - `phase17_coverage_sectorbias_ultra`
  - `phase18_coverage_stronger_repulsion`
- added optional `sector_target_bias_strength` and optional `use_global_slot_head` / `global_slot_strength`
- trained `debug_bc_coverage_globalslot`

Current evidence:

- success-policy dataset collected from `phase10` policy remained sparse:
  - deterministic `u60`: only `2` successful episodes over `240` attempts
  - deterministic `u40`: `0` successful episodes over `240` attempts
  - noisy `u60 + noise 0.05`: still only `2` successful episodes over `240` attempts
- `phase15_coverage_successbc_ultra` underperformed the plain `phase10` branch
- `phase17_coverage_sectorbias_ultra` underperformed the plain spatial-head branch
- `debug_bc_coverage_globalslot` underperformed the best `spatialhead BC`

Reason:

- the current coverage PPO line can enter the success region, but the success basin is still narrow and brittle
- adding weak success-only data is not enough by itself
- current lightweight group-level approximations have not yet surpassed the simpler spatial-head PPO branch

## Theme AF: risk_nav repaired and formation pushed to heuristic-level PPO

Implemented:

- generated success-only heuristic dataset for `risk_nav`
- trained full-state BC and ultra-strict PPO:
  - `outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra/`
- generated success-only and full-state datasets for `formation`
- trained formation BC / PPO / DAgger BC / success-policy BC branches

Current evidence:

- `risk_nav` final line is now stable and reusable:
  - final `success_rate_mean=0.6`
  - final `goal_coverage_ratio_mean≈0.829`
  - final `collision_rate_mean≈0.049`
- `formation` best PPO line remains:
  - `outputs/training/bc_ppo/20260528_phase14_formation_ultra/phase14_formation_ultra/`
  - stable mid-run `success≈0.4`
  - final `success≈0.3`
- later formation success-policy / DAgger BC branches did not exceed that line

Reason:

- `risk_nav` transferred well from the `goal_nav` repair recipe
- `formation` benefits from structure-aware actor heads, but still does not hold its best success level all the way to final eval

## Theme AG: optional global-slot actor branch with cross-task compatibility

Implemented:

- extended `CNNDeepSetsPolicy` with optional:
  - `use_global_slot_head`
  - `global_slot_strength`
- added a cross-task compatibility test:
  - `tests/test_variable_policies.py::test_cnn_deepsets_global_slot_head_is_compatible_across_tasks`
- verified the expanded policy test suite passes with `9 passed`

Reason:

- the user explicitly required that any new task-allocation architecture remain compatible with the old centralized policy path
- the new group-level allocation branch therefore stays optional, default-off, and is now minimally validated across `goal_nav`, `coverage`, `formation`, and `risk_nav`

## Theme AH: PPO best-eval checkpoint preservation

Implemented:

- `algorithms/ppo.py` now saves:
  - `checkpoints/checkpoint_best_eval.pt`
  - `best_eval_summary.json`
- `scripts/train_ppo.py` variable-`N` loop mirrors the same behavior

Selection rule:

- prefer higher `eval_success_rate`
- break ties with higher `eval_reward`

Reason:

- `coverage` and `formation` often hit their best success mid-run and then drift downward by final eval
- preserving the best eval checkpoint avoids losing the most useful specialist snapshot just because the final checkpoint is worse

## Theme AI: final eval now prefers best PPO checkpoint

Implemented:

- `scripts/train_ppo.py` now accepts `--final_eval_source best|last`
- default is `best`
- if `checkpoints/checkpoint_best_eval.pt` exists, final per-task evaluation loads that checkpoint before writing `eval_metrics.csv`
- final eval rows now include `final_eval_source`

Reason:

- coverage and formation repeatedly showed “mid-run best, final worse”
- selecting the best checkpoint for final reporting is a pragmatic engineering fix that turns an unstable tail into a usable specialist artifact without changing the training update rule itself

## Theme AJ: coverage success-only reinforcement deepening

Implemented:

- collected a stronger success-heavy coverage dataset from `phase10` update 80 over a better seed window:
  - `outputs/debug_long/20260529_coverage_success_policy/coverage_success_from_phase10_u80_seed5000_plus_expertv2.npz`
  - `successful_episodes=15`
- extracted / iterated success-only mixes:
  - `coverage_success_only_from_phase10_u80_seed5000.npz`
  - `coverage_success_from_successBC_e40.npz`
  - `coverage_success_from_successBC_e40_round2.npz`
  - `coverage_success_from_successBC_e40_round3.npz`
- trained successive success-heavy BC refinements:
  - `outputs/training/bc/20260529_015503/...`
  - `outputs/training/bc/20260529_021428/...`
  - `outputs/training/bc/20260529_051145/...`
  - `outputs/training/bc/20260529_155058/...`

Current evidence:

- the first success-heavy BC pushed coverage BC success from `0.05` to `0.10`
- the next strengthened success-heavy BC pushed it further to:
  - `return_mean≈9.697`
  - `normalized_score_mean≈2.814`
  - `success_rate_mean≈0.15`
  - `collision_rate_mean≈0.0`
- PPO launched from these stronger success-only BC checkpoints (`phase20`, `phase23`) still did not exceed the `phase21 best` coverage PPO line

Reason:

- coverage success behavior can be partially reinforced through data, but the gain saturates before PPO turns that into a clearly better final specialist
- this suggests the remaining bottleneck is no longer “missing success examples” alone

## Theme AK: true group-level coverage actor candidates

Implemented:

- added optional `use_global_spatial_slot_head` / `global_spatial_slot_strength`
- added optional `actor_mean_residual_weight`
- added configs and probes for:
  - `global-slot BC`
  - `global-slot PPO`
  - `sector-bias PPO`
  - `slot-dominant BC`
- expanded `tests/test_variable_policies.py` to cover:
  - global slot head
  - slot-dominant actor
  - global-slot compatibility across tasks

Current evidence:

- `global-slot BC` and `slot-dominant BC` did not beat the best `spatial-head BC`
- `phase26_coverage_gspatialslot_bestfinal` first eval is still only `success≈0.05`
- the best coverage line remains the simpler `spatial-head PPO` branch with best-final checkpoint selection

Reason:

- lightweight group-level extensions are now present and compatible, but they have not yet beaten the current best coverage mainline
- further progress on coverage likely needs a stronger, more explicit group-level coordination design rather than another small bias term

## Theme P: Sequential PPO task-combination queue launcher

Implemented:

- added `scripts/run_ppo_task_queue.sh` for sequential PPO training over editable task-combination rows
- each queue row has its own label, task list, agent counts, update budget, eval episodes, GIF recording cadence, GPU selection, and extra CLI args
- queue runs remain headless, use TensorBoard, preserve `outputs/training/ppo/<timestamp>/<run_name>/`, and write queue logs under `outputs/training/task_queue/<timestamp>/`
- completion notification targets `muadib@foxmail.com` by default and supports SMTP env vars or local `mail` / `mailx` / `sendmail`
- `NOTIFY_ONLY=1` can now be used to test the notification path without starting training
- README now includes the queue launcher entrypoint

Reason:

- task-combination sweeps need repeatable sequential execution without manually starting every PPO run
- per-row parameters make it practical to debug different task mixes and hyperparameter variants while keeping the existing PPO training contract

## Theme Q: Core code explanatory comments

Implemented:

- added explanatory docstrings and comments to `envs/centralized_env.py` around centralized environment setup, observation/action contracts, runtime scaling, state initialization, task-field composition, transition stepping, collision handling, reward composition, and info metrics
- added comments to `algorithms/ppo.py` explaining rollout storage, reward normalization, critic bootstrap, GAE, minibatch flattening, clipped PPO ratios, and loss components
- added neural-network comments to `policies/cnn_deepsets_policy.py`, `policies/attention_policy.py`, and `policies/mlp_policy.py` describing architecture roles, variable-N handling, token/pooling logic, and centralized value/action heads
- added distribution comments to `policies/action_distribution.py` explaining tanh-squashed Gaussian actions and log-prob change-of-variables correction

Reason:

- future readers need the environment, algorithm, and policy framework to be understandable without reverse-engineering every tensor transformation
- comments document intent and contracts only; no algorithm semantics or training outputs were changed

## Theme R: Threaded environment batch backend

Implemented:

- added `ThreadEnvBatch` beside `SyncEnvBatch` in `utils/vector_env.py`, using `ThreadPoolExecutor` while preserving the same `envs`, `num_envs`, `reset(...)`, and `step(actions)` interface
- kept `SyncEnvBatch` behavior unchanged as the serial debug baseline
- extended `make_env_batch(...)` with `backend="sync|thread"` and optional `max_workers`
- added `make_task_balanced_env_batch(...)` to create a fixed number of environments per task with deterministic per-env seeds
- exported `ThreadEnvBatch` and `make_task_balanced_env_batch` through `utils/__init__.py`
- added `--env_backend`, `--envs_per_task`, and `--env_workers` to PPO, SAC, and TD3 training entrypoints
- added tests for threaded reset/step shape, ordered result collection, done-time auto reset, task-balanced env counts, forced task assignment, and task-balanced thread stepping

Reason:

- rollout collection was previously serialized over environment instances, which limits throughput when environment work dominates policy inference
- task-balanced batches make multi-task rollout sampling explicit by creating equal fixed-task environment groups instead of relying only on stochastic task sampling
- default backend remains `sync`, so all existing training commands preserve their old behavior unless a caller opts into `thread`

## Theme S: PPO multi-task suite launcher with specialist policies

Implemented:

- added `scripts/run_ppo_multitask_suite.sh` as a dedicated PPO experiment suite launcher
- the suite explicitly trains four independent specialist policies: `goal_nav`, `coverage`, `formation`, and `risk_nav`
- the suite also trains selected multi-task policies for pairs, triples, and all four tasks
- important tunables are listed at the top of the script, including config paths, scaling mode, observation variant, total updates, eval episodes, GIF cadence, env backend, task-balanced env count, thread workers, GPU id, and email settings
- each queue row carries its own task mix and training/evaluation/backend/GPU parameters
- queue logs and summary CSVs are written under `outputs/training/ppo_multitask_suite/<timestamp>/`, while training artifacts preserve the standard `outputs/training/ppo/<timestamp>/<run_name>/` layout
- `NOTIFY_ONLY=1` can now be used to test the suite notification path without running the queue

Reason:

- comparing specialist single-task policies with multi-task policies requires running both families under one repeatable queue
- keeping all important parameters visible in the script makes server-side experiment editing practical without changing trainer code

## Theme T: Shorter PPO run directory names

Implemented:

- shortened `scripts/train_ppo.py` run names from `<config>_<full_task_names>_N<agents>_<obs_variant>` to `<config>_<compact_task_tag>_N<agents>`
- compact task tags use `goal`, `cov`, `form`, `risk`, and `all4` for the canonical four-task set
- removed observation variant from the PPO run directory name because it is already saved in `snapshot/cli_args.yaml`

Reason:

- long PPO run directory names were cumbersome on server runs and nested output paths
- the removed fields are still recoverable from run snapshots and CSV metadata, so shortening the directory name does not remove experiment provenance

## Theme U: Queue-level PPO output grouping

Implemented:

- added optional `--run_timestamp` and `--run_name` arguments to `scripts/train_ppo.py`
- `scripts/run_ppo_task_queue.sh` and `scripts/run_ppo_multitask_suite.sh` now pass their launch timestamp to every child PPO run
- queue row labels are passed as child PPO run names, producing output paths such as `outputs/training/ppo/<queue_timestamp>/<run_label>/...`

Reason:

- a scripted training suite should group all child runs under the script launch time instead of scattering them across many per-run timestamps
- queue labels are clearer than long auto-generated names when comparing many independent runs from one launch

## Theme V: QQ SMTP notification configuration

Implemented:

- queue scripts now support a QQ SMTP preset with defaults for `smtp.qq.com:465` over SSL
- sensitive mail values are loaded from environment variables or an ignored local file at `.secrets/wayffusion_mail.env`
- added `configs/examples/wayffusion_mail.env.example` as the safe template for local SMTP configuration
- `.gitignore` now excludes `.secrets/` and `wayffusion_mail.env`
- notification self-test remains available with `NOTIFY_ONLY=1`

Reason:

- email authorization codes must not be committed into scripts
- QQ/Foxmail SMTP requires an authorization code, so scripts now explicitly warn when `SMTP_USER` or `SMTP_PASSWORD` is missing

Follow-up validation:

- fixed SMTP variables loaded from `.secrets/wayffusion_mail.env` so they are explicitly passed into the Python mail sender subprocess
- verified `NOTIFY_ONLY=1` succeeds for both `scripts/run_ppo_multitask_suite.sh` and `scripts/run_ppo_task_queue.sh`

## Theme W: Queue script GPU selection inheritance

Implemented:

- updated `scripts/run_ppo_multitask_suite.sh` so `DEFAULT_CUDA_VISIBLE_DEVICES` inherits an externally supplied `CUDA_VISIBLE_DEVICES`
- changed the default suite rows to leave their per-row GPU field empty, allowing commands such as `CUDA_VISIBLE_DEVICES=5 bash scripts/run_ppo_multitask_suite.sh` to run on physical GPU 5
- kept per-row GPU override support intact; filling the row GPU field still takes precedence over the inherited default

Reason:

- hard-coded per-row `cuda_visible_devices=0` caused external GPU selection to be ignored
- server users need a simple launch-time way to choose a GPU without editing every queue row

## Theme X: Goal-nav reward alignment repair

Implemented:

- `tasks/goal_nav.py` and `tasks/risk_nav.py` now rebuild `goal_reward` from unreached goals only
- `goal_progress` and distance shaping now use remaining goals instead of all goals
- `tests/test_rewards_basic.py` now guards the remaining-goal shaping contract

Reason:

- the previous static field and all-goal shaping combination could keep already reached goals attractive and distort progress rewards
- this change aligns reward shaping with the intended completion objective without lowering the success threshold

## Theme Y: Goal-nav debug-long diagnostics and DAgger workflow

Implemented:

- added `outputs/debug_long/20260528_035433/` experiment records for takeover, PPO baselines, BC, diagnostics, and summaries
- added `scripts/debug_long/analyze_ppo_run.py` to summarize PPO CSVs
- added `scripts/debug_long/diagnose_goal_nav_policy.py` to measure deterministic action alignment against remaining goal peaks
- added `scripts/debug_long/generate_success_expert_dataset.py` to build a success-only goal_nav BC dataset
- added `scripts/debug_long/collect_dagger_dataset.py` to collect learner states and relabel them with heuristic expert actions
- added `configs/policy/debug_bc_goal_nav_success.yaml` for the success-only BC run

Reason:

- PPO from random initialization produced zero success and a near-zero deterministic mean action
- successful-trajectory BC improved action direction but still had brittle collisions, so DAgger-style learner-state relabeling became the next repair path

## Theme Z: BC warm-start and PPO fine-tune hooks for goal_nav

Implemented:

- `scripts/train_bc.py` now accepts `--init_checkpoint`
- `algorithms/ppo.py` now clamps `log_std` immediately after checkpoint load and at trainer init
- `configs/policy/debug_ppo_goal_nav_bc_finetune_strict.yaml` is used for conservative BC-to-PPO fine-tuning

Reason:

- BC and PPO now share a reproducible warm-start path
- keeping the exploration scale fixed during fine-tune avoids reintroducing the BC checkpoint's looser action variance

## Theme AF: coverage completion-focused reward diagnostic

Implemented on 2026-05-30:

- `tasks/coverage.py` now supports two default-off reward components:
  - `coverage_shortfall`: per-step shaping based on the gap between current `coverage_ratio` and the unchanged success threshold
  - `failure_penalty`: one terminal penalty when the episode reaches `max_steps` without coverage success
- the default/base coverage task behavior is unchanged unless those weights are present in the env config
- added `configs/env/debug_coverage_completion_focus.yaml`
- added `configs/policy/debug_ppo_coverage_perm_ref_completion.yaml`
- added `tests/test_rewards_basic.py::test_coverage_failure_penalty_only_on_unsuccessful_timeout`

Reason:

- latest coverage runs show the policy often plateaus around `coverage_ratio≈0.70` with low collision but below the fixed `0.82` success threshold
- old reward settings paid large cumulative `coverage_level_reward` to non-successful long episodes, so PPO could prefer stable near-threshold failure over early successful termination
- the diagnostic branch keeps the success threshold unchanged, reduces the incentive to idle at partial coverage, raises the success terminal signal, and explicitly penalizes timed-out non-success

Validation:

- `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_bc_permutation_loss.py tests/test_variable_policies.py tests/test_ppo_episode_budget.py tests/test_expert_dataset.py`
- result: `22 passed`

Current coverage evidence before phase31:

- phase29 reference-regularized PPO best checkpoint:
  - `outputs/training/bc_ppo/20260530_phase29_coverage_permref/phase29_coverage_permref/checkpoints/checkpoint_best_eval.pt`
  - 50-episode eval: `success_rate_mean=0.16`, `coverage_ratio_mean≈0.702`, `collision_rate_mean≈0.00035`
- phase29 self-success BC dataset and BC replay did not improve:
  - dataset: `outputs/debug_long/20260530_coverage_expert_v3/coverage_success_round4_v3_phase29.npz`
  - 50-episode eval of self-success BC: `success_rate_mean=0.08`, `coverage_ratio_mean≈0.708`, `collision_rate_mean≈0.000875`
  - conclusion: simply cloning phase29 successful episodes regresses toward average partial-coverage behavior

Next experiment:

- phase31 should warm-start from phase29 best checkpoint and use the completion-focused env config
- this is a labeled reward-shaping diagnostic, not a lowered-threshold success claim

Phase31 result:

- run root: `outputs/training/bc_ppo/20260530_phase31_coverage_completion/phase31_coverage_completion/`
- best checkpoint update: `90`
- 20-episode final/best eval: `success_rate_mean=0.30`, `coverage_ratio_mean≈0.720`, `collision_rate_mean≈0.00025`
- 50-episode best eval: `success_rate_mean=0.16`, `coverage_ratio_mean≈0.707`, `collision_rate_mean≈0.000226`
- conclusion: completion-focused reward fixes the objective semantics and maintains the 20-episode peak, but does not improve 50-episode robustness over phase29

Additional coverage expert diagnostics:

- added `CoverageExpertV4` in `scripts/debug_long/generate_coverage_expert_v2_dataset.py`
  - stateful sector assignment
  - persistent per-agent targets
  - short-range collision repulsion
- extended `scripts/debug_long/generate_success_expert_dataset.py --teacher` with `coverage_expert_v4`
- validation:
  - `/opt/conda/bin/python -m pytest -q tests/test_expert_dataset.py tests/test_rewards_basic.py tests/test_bc_permutation_loss.py tests/test_variable_policies.py`
  - result: `21 passed`
- direct 80-episode V4 eval:
  - output: `outputs/debug_long/20260530_coverage_expert_v4/coverage_expert_v4_eval_e80.summary.json`
  - `success_rate=0.0`, `coverage_ratio_mean≈0.567`, `collision_rate_mean≈0.0161`
  - conclusion: V4 is worse than phase29 policy and should not be used for BC/PPO
- direct 80-episode band-sweep diagnostic:
  - output: `outputs/debug_long/20260530_coverage_band_sweep/band_sweep_eval_e80.summary.json`
  - `success_rate=0.0`, `coverage_ratio_mean≈0.596`, `collision_rate_mean≈0.0127`
  - conclusion: simple hand-coded lane/band sweep is not a viable teacher

Phase32 result:

- added `configs/policy/debug_ppo_coverage_perm_ref_completion_explore.yaml`
- run root: `outputs/training/bc_ppo/20260530_phase32_coverage_completion_explore/phase32_coverage_completion_explore/`
- warm-start: phase29 best checkpoint
- key change: higher exploration (`log_std≈-1.2`), lower reference regularization, 2 PPO epochs, small entropy bonus
- stopped manually after drift because best checkpoint was already saved:
  - best checkpoint update: `20`
  - 20-episode best eval: `success_rate_mean=0.35`
  - later eval drifted to `0.15` at update 50 and `0.10` at update 60
- 50-episode best eval:
  - `success_rate_mean=0.20`
  - `coverage_ratio_mean≈0.703`
  - `collision_rate_mean≈0.000125`
- conclusion: exploration can find slightly better coverage specialists than phase29/31, but the improvement is still weak and drifts without stronger stabilization

Phase33 result:

- added `configs/policy/debug_ppo_coverage_perm_ref_completion_stabilize.yaml`
- run root: `outputs/training/bc_ppo/20260530_phase33_coverage_completion_stabilize/phase33_coverage_completion_stabilize/`
- warm-start: phase32 best checkpoint
- key change: lower LR (`3e-6`), stronger reference regularization (`0.35`), lower fixed exploration than phase32
- training behavior:
  - update20 eval `success_rate=0.35`
  - update40 eval `success_rate=0.35`
  - no phase32-style collapse through update80
- 50-episode h200 best eval:
  - `success_rate_mean=0.20`
  - `coverage_ratio_mean≈0.707`
  - `collision_rate_mean≈0.000125`
- conclusion: phase33 stabilizes the 20-episode `0.35` peak but still only generalizes to `0.20` on h200

Horizon feasibility diagnostic:

- added `configs/env/debug_coverage_completion_focus_h300.yaml`
- this config keeps `coverage.success_ratio=0.82` unchanged but increases `max_steps` from 200 to 300
- same phase33 checkpoint evaluated under h300:
  - output: `outputs/training/bc_ppo/20260530_phase33_coverage_completion_stabilize/phase33_coverage_completion_stabilize/eval_best_50_h300_diagnostic/`
  - `success_rate_mean=0.48`
  - `coverage_ratio_mean≈0.769`
  - `collision_rate_mean≈0.000198`
- interpretation:
  - the h200 policy already knows a useful coverage behavior
  - many h200 failures are time/path-budget failures, because allowing more steps without lowering the threshold nearly doubles the 50-episode success rate
  - h300 is a diagnostic environment change and must not be mixed with h200 canonical results

Phase34 h300 PPO result:

- run root: `outputs/training/bc_ppo/20260530_phase34_coverage_h300_stabilize/phase34_coverage_h300_stabilize/`
- warm-start: phase33 best checkpoint
- env: `configs/env/debug_coverage_completion_focus_h300.yaml`
- best checkpoint update: `60`
- 20-episode best/final-source eval:
  - `success_rate_mean=0.65`
  - `collision_rate_mean=0.0`
- 50-episode best eval:
  - `success_rate_mean=0.56`
  - `coverage_ratio_mean≈0.770`
  - `collision_rate_mean≈0.000135`
  - `path_length_mean≈1.410`
- conclusion:
  - coverage PPO can train a substantially stronger expert when the horizon allows enough path budget
  - current h200 coverage remains unresolved; current h300 coverage is the strongest diagnostic specialist

## Theme AG: factorized group actor policy

Implemented on 2026-05-30:

- added `policies/factorized_group_policy.py`
- added `policy_class: factorized_group` to `policies/__init__.py`
- added configs:
  - `configs/policy/ppo_factorized_group.yaml`
  - `configs/policy/debug_bc_goal_nav_factorized_group.yaml`
  - `configs/policy/debug_ppo_goal_nav_factorized_group_finetune.yaml`
  - `configs/policy/debug_bc_coverage_factorized_group_perm.yaml`
  - `configs/policy/debug_ppo_coverage_factorized_group_h300.yaml`
- added tests in `tests/test_variable_policies.py`

Architecture contract:

- critic remains centralized and still consumes one global state feature
- actor remains shared across UAVs and still outputs `[B, N, 2]`
- a small bank of learned group tokens is conditioned on the global state
- each agent soft-assigns to those group tokens
- group tokens optionally select spatial slots from the field and feed both:
  - a per-agent group context into the actor decoder
  - a group-coordinate waypoint bias
- `agent_mask` and variable `N` stay supported
- PPO joint log-prob aggregation remains unchanged because the action head still emits one factorized `[N,2]` action tensor per env

Reason:

- the old `CNNDeepSetsPolicy` already had a shared per-agent decoder, but all agents received the same pooled swarm context
- this new policy inserts an explicit coordination layer between pooled state and per-agent action decoding, matching the requested “single-agent / small-group waypoint allocation under centralized information sharing” direction

Validation:

- `/opt/conda/bin/python -m pytest -q tests/test_variable_policies.py tests/test_rewards_basic.py tests/test_bc_permutation_loss.py tests/test_expert_dataset.py`
- result after policy integration: `23 passed`
- final recheck after all edits:
  - `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_expert_dataset.py tests/test_bc_permutation_loss.py tests/test_variable_policies.py tests/test_ppo_episode_budget.py`
  - result: `22 passed`

Goal-nav factorized-group specialist:

- BC run:
  - `outputs/training/bc/20260530_082030/debug_bc_goal_nav_factorized_group_goal_nav_N4_multi_channel_field_plus_task_id/`
  - dataset: `outputs/debug_long/20260528_035433/datasets/goal_nav_dagger_from_bcppo060_plus_success.npz`
  - 20-episode eval: `success_rate_mean=0.75`, `goal_coverage_ratio_mean≈0.855`
- PPO fine-tune:
  - `outputs/training/bc_ppo/20260530_phase36_goalnav_factorized_group_ppo/phase36_goalnav_factorized_group_ppo/`
  - 20-episode best/final-source eval: `success_rate_mean=0.80`, `goal_coverage_ratio_mean≈0.872`
  - 50-episode best eval:
    - `success_rate_mean=0.78`
    - `goal_coverage_ratio_mean≈0.877`
    - `collision_rate_mean≈0.056`
- conclusion:
  - the new factorized-group policy can train a real PPO specialist expert on `goal_nav`
  - this de-risks the new decision architecture itself

Coverage factorized-group specialist:

- BC run:
  - `outputs/training/bc/20260530_083743/debug_bc_coverage_factorized_group_perm_coverage_N4_multi_channel_field_plus_task_id/`
  - dataset: `outputs/debug_long/20260530_coverage_expert_v3/coverage_success_round4_plus_v3_e40.npz`
  - 20-episode eval: `success_rate_mean=0.20`, `coverage_ratio_mean≈0.700+`, `collision_rate_mean=0.0`
- h300 PPO fine-tune:
  - `outputs/training/bc_ppo/20260530_phase37_coverage_factorized_group_h300/phase37_coverage_factorized_group_h300/`
  - 20-episode final-source eval: `success_rate_mean=0.50`, `coverage_ratio_mean≈0.768`, `collision_rate_mean≈0.0`
  - 50-episode best eval:
    - `success_rate_mean=0.40`
    - `coverage_ratio_mean≈0.768`
    - `collision_rate_mean≈0.00138`
- comparison against old h300 specialist:
  - old phase34 h300 50-episode eval: `success_rate_mean=0.56`, `coverage_ratio_mean≈0.770`, `collision_rate_mean≈0.000135`
- conclusion:
  - the new architecture is viable for coverage but is not yet the best-performing coverage policy
  - current factorized-group coverage issue is not “cannot train at all”; it is “path efficiency / stability still worse than old spatial-head line”

## Theme AH: factorized-group risk_nav and formation validation

Implemented on 2026-05-30:

- added new-architecture configs:
  - `configs/policy/debug_bc_risk_nav_factorized_group.yaml`
  - `configs/policy/debug_ppo_risk_nav_factorized_group_ultra_strict.yaml`
  - `configs/policy/debug_bc_formation_factorized_group.yaml`
  - `configs/policy/debug_ppo_formation_factorized_group_ultra_strict.yaml`
  - `configs/policy/debug_ppo_risk_nav_factorized_group_safe_ref.yaml`
- generated a risk low-collision diagnostic dataset:
  - `outputs/debug_long/20260530_risk_nav_factorized_group/risk_nav_success_lowcollision_N4_stride2.npz`
  - source: `outputs/debug_long/20260528_risk_nav_success/risk_nav_success_N4_stride2.npz`
  - kept `30 / 40` successful episodes with `collision_rate <= 0.02`
  - samples: `903`

Risk-nav factorized-group results:

- BC from original success dataset:
  - run root: `outputs/training/bc/20260530_095258/debug_bc_risk_nav_factorized_group_risk_nav_N4_multi_channel_field_plus_task_id/`
  - 20-episode eval: `success_rate_mean=0.45`, `goal_coverage_ratio_mean≈0.633`, `collision_rate_mean≈0.139`
- PPO from that BC:
  - run root: `outputs/training/bc_ppo/20260530_phase38_risknav_factorized_group_ppo/phase38_risknav_factorized_group_ppo/`
  - 20-episode best/final-source eval: `success_rate_mean=0.45`, `collision_rate_mean≈0.129`
  - 50-episode best eval: `success_rate_mean=0.24`, `goal_coverage_ratio_mean≈0.570`, `collision_rate_mean≈0.177`
- safe/reference PPO diagnostic:
  - run root: `outputs/training/bc_ppo/20260530_phase40_risknav_factorized_group_safe_ref/phase40_risknav_factorized_group_safe_ref/`
  - stopped early after update 35 because eval stayed poor
  - best observed eval: `success_rate=0.15`
  - reason for failure: stronger repulsion and lower group bias degraded goal-reaching before it solved collisions
- low-collision-only BC:
  - run root: `outputs/training/bc/20260530_115519/debug_bc_risk_nav_factorized_group_risk_nav_N4_multi_channel_field_plus_task_id/`
  - 20-episode eval: `success_rate_mean=0.05`, `collision_rate_mean≈0.240`
  - reason for failure: filtering removed too much state coverage; the model overfit a narrower success subset and generalized worse
- conclusion:
  - risk_nav is not repaired under `factorized_group`
  - current blocker is collision-heavy path behavior; imitation-only and simple repulsion/reference fixes were not enough
  - old CNNDeepSets risk specialist remains stronger (`50/20 eval success around 0.60`)

Formation factorized-group results:

- BC from DAgger/full-heuristic dataset:
  - run root: `outputs/training/bc/20260530_095515/debug_bc_formation_factorized_group_formation_N4_multi_channel_field_plus_task_id/`
  - 20-episode eval: `success_rate_mean=0.50`, `formation_error_mean≈0.058`, `collision_rate_mean=0.0`
- PPO from that BC:
  - run root: `outputs/training/bc_ppo/20260530_phase39_formation_factorized_group_ppo/phase39_formation_factorized_group_ppo/`
  - 20-episode best/final-source eval: `success_rate_mean=0.50`, `collision_rate_mean=0.0`
  - 50-episode best eval: `success_rate_mean=0.44`, `formation_error_mean≈0.063`, `collision_rate_mean≈0.000825`
- comparison:
  - old formation best 20/final eval was around `success_rate_mean=0.35`
- conclusion:
  - formation is now provisionally repaired under `factorized_group`
  - PPO did not improve BC, but best-checkpoint preservation keeps a useful specialist

Validation:

- `/opt/conda/bin/python -m pytest -q tests/test_variable_policies.py tests/test_rewards_basic.py tests/test_bc_permutation_loss.py tests/test_expert_dataset.py tests/test_ppo_episode_budget.py`
- result: `24 passed`

## Theme AI: factorized_group continuation after connection loss

Implemented / recorded:

- Added `outputs/debug_long/20260530_factorized_group_continuation/00_takeover.md` as the recovery record for the current continuation.
- Added `configs/env/debug_risk_nav_safety_completion.yaml`, a reward-only risk-nav diagnostic/training config. It changes reward weights only and preserves dynamics, observations, task success thresholds, max steps, and task definitions.
- Added `configs/policy/debug_ppo_risk_nav_factorized_group_dagger_safe.yaml`, a conservative factorized-group PPO fine-tune config intended for DAgger BC checkpoints.

Reason:

- Under the new architecture, `goal_nav` is already repaired and `formation` is provisionally usable; `risk_nav` and `coverage` remain the blockers.
- Risk-nav failure is currently best explained by unsafe/collision-heavy successful demonstrations plus distribution mismatch. Low-collision-only filtering previously reduced state coverage and generalized worse.
- The next risk-nav repair attempt therefore uses learner-state DAgger plus conservative PPO rather than another strong hand-coded repulsion/reference branch.

## Theme AJ: risk_nav factorized_group DAgger BC result

Experiment:

- Collected learner-state DAgger data from phase38 factorized-group risk learner and relabeled with `HeuristicPolicy`.
- Dataset path: `outputs/debug_long/20260530_factorized_group_continuation/risk_nav_dagger_from_phase38_plus_success.npz`.
- Dataset includes original success data plus 6971 new learner-state samples, 8394 total samples.
- Trained BC with `configs/policy/debug_bc_risk_nav_factorized_group.yaml`.
- BC run: `outputs/training/bc/20260530_121227/debug_bc_risk_nav_factorized_group_risk_nav_N4_multi_channel_field_plus_task_id/`.

Result:

- 30-episode BC eval: `success_rate_mean=0.533`, `collision_rate_mean=0.114`, `return_mean=-4.408`, `normalized_score_mean=0.659`.

Reasoning:

- This supports the hypothesis that risk-nav was not failing because the new architecture cannot represent useful actions; adding learner-state coverage improves success over the previous factorized-group risk plateau.
- Remaining blocker is safety/collision, so the follow-up PPO should be conservative and safety-shaped rather than high-entropy exploration.

## Theme AK: risk_nav factorized_group provisionally repaired

Experiment:

- PPO phase41 from the DAgger BC checkpoint using `configs/policy/debug_ppo_risk_nav_factorized_group_dagger_safe.yaml` and reward-only env config `configs/env/debug_risk_nav_safety_completion.yaml`.
- Run: `outputs/training/bc_ppo/20260530_phase41_risknav_factorized_group_dagger_safe/phase41_risknav_factorized_group_dagger_safe/`.
- Independent 100-episode eval: `outputs/debug_long/20260530_factorized_group_continuation/eval_phase41_risknav_100ep/risk_nav_N4_multi_channel_field_plus_task_id.csv`.

Result:

- Best-source 30-episode final eval: `success_rate_mean=0.70`, `collision_rate_mean=0.017`.
- Independent 100-episode eval: `success_rate_mean=0.65`, `goal_coverage_ratio_mean=0.85`, `collision_rate_mean=0.0208`, `path_length_mean=0.699`.

Reasoning:

- DAgger fixed learner-state distribution mismatch; conservative PPO then improved safety/completion without overwriting BC behavior.
- Risk-nav should now be treated as provisionally repaired under the new `factorized_group` architecture, pending seed-repeat validation.

## Theme AL: coverage factorized_group h200 utility-slot PPO config

Implemented:

- Added `configs/policy/debug_ppo_coverage_factorized_group_completion_h200.yaml`.

Reason:

- Coverage under factorized-group is not mainly a collision/safety failure; it reaches reasonable coverage but misses the success threshold within the canonical 200-step budget.
- h300 success is diagnostic only, not a canonical fix.
- The new config stays on `factorized_group` and enables the existing coverage utility slot bias to improve spatial partitioning while using conservative PPO and reference regularization.

## Theme AM: coverage phase42 direct utility PPO failed

Experiment:

- Started `phase42_coverage_factorized_group_h200_utility` from the factorized-group coverage BC checkpoint using `configs/policy/debug_ppo_coverage_factorized_group_completion_h200.yaml` and h200 completion reward config.

Result:

- Eval success was `0.0` at updates 20, 40, 60, and 80.
- The run was stopped early after update 90.

Reasoning:

- Directly enabling the coverage utility slot bias at PPO time changed the action prior too much and destroyed the BC behavior.
- Next coverage attempt should first train BC with the utility-enabled policy config, then PPO fine-tune from the adapted BC checkpoint.

## Theme AN: coverage utility-head BC adaptation configs

Implemented:

- Added `configs/policy/debug_bc_coverage_factorized_group_utility_perm.yaml`.
- Added `configs/policy/debug_ppo_coverage_factorized_group_utility_bcfit.yaml`.

Reason:

- phase42 showed that enabling a strong utility-slot bias only at PPO time destroys the BC action distribution.
- The adapted route lowers utility/repulsion strength and first trains BC with the same action prior, so PPO starts from a policy already fitted to the modified architecture.

## Theme AO: coverage utility-head BC failed

Experiment:

- Trained `configs/policy/debug_bc_coverage_factorized_group_utility_perm.yaml` on `outputs/debug_long/20260530_coverage_expert_v3/coverage_success_round4_plus_v3_e40.npz`.
- Run: `outputs/training/bc/20260530_135952/debug_bc_coverage_factorized_group_utility_perm_coverage_N4_multi_channel_field_plus_task_id/`.

Result:

- BC loss reached about `0.002`.
- 30-episode eval: `success_rate_mean=0.0`, `collision_rate_mean=0.022`, `return_mean=5.010`.

Reasoning:

- This branch shows that supervised action fit is not enough for closed-loop coverage success when the utility prior changes the trajectory distribution.
- Do not PPO this checkpoint. Next coverage attempt should use learner-state DAgger with the original factorized-group policy first.

## Theme AP: coverage h200 teacher diagnostics

Diagnostics:

- Evaluated existing coverage teachers in canonical h200 env. Results saved to `outputs/debug_long/20260530_factorized_group_continuation/coverage_teacher_h200_eval.json`.
- `coverage_expert_v2` success `0.0`, coverage ratio `0.536`.
- `coverage_expert_v3` success `0.025`, coverage ratio `0.598`.
- `coverage_expert_v4` success `0.0`, coverage ratio `0.557`.
- `heuristic_greedy_coverage` success `0.0`, coverage ratio `0.456`.
- A temporary in-shell sweep prototype was tested and discarded because it had success `0.0`, coverage about `0.373`, and high collision.

Reasoning:

- Current coverage BC/DAgger data sources are not true h200 experts, so imitation alone is not expected to solve canonical h200 coverage.
- Added `configs/policy/debug_ppo_coverage_factorized_group_h200_stable.yaml` to run conservative PPO from the original factorized-group BC without utility-head changes.

## Theme AQ: coverage phase43 h200 stable PPO did not break through

Experiment:

- Ran `phase43_coverage_factorized_group_h200_stable` from the original factorized-group coverage BC checkpoint.
- Config: `configs/policy/debug_ppo_coverage_factorized_group_h200_stable.yaml`.
- Env config: `configs/env/debug_coverage_completion_focus.yaml`.
- Run: `outputs/training/bc_ppo/20260530_phase43_coverage_factorized_group_h200_stable/phase43_coverage_factorized_group_h200_stable/`.

Result:

- Best periodic 30-episode eval: `success_rate=0.20` at update 60.
- Later evals stayed in `0.133-0.167`; run stopped early after update 120.

Reasoning:

- Stable PPO preserved but did not improve the original BC behavior.
- Coverage h200 remains unresolved under `factorized_group`.
- The blocker is now identified as missing high-success h200 supervision/curriculum, not direct architecture incompatibility or PPO math instability.

## Theme AR: coverage ratio-curriculum reward config

Implemented:

- Added `configs/env/debug_coverage_ratio_curriculum.yaml`.
- Added `configs/policy/debug_ppo_coverage_factorized_group_h200_ratio_curriculum.yaml`.

Reason:

- Coverage h200 policies are near but below threshold; teacher/DAgger and utility-head routes failed.
- This route changes reward weights only, preserving threshold, h200 budget, dynamics, and task definition.
- It increases dense coverage-level and terminal shortfall/failure pressure so PPO has stronger gradient near the success threshold.

## Theme AS: coverage phase44 ratio-curriculum failed

Experiment:

- Ran `phase44_coverage_factorized_group_ratio_curriculum` using reward-only config `configs/env/debug_coverage_ratio_curriculum.yaml` and PPO config `configs/policy/debug_ppo_coverage_factorized_group_h200_ratio_curriculum.yaml`.
- Run: `outputs/training/bc_ppo/20260530_phase44_coverage_factorized_group_ratio_curriculum/phase44_coverage_factorized_group_ratio_curriculum/`.

Result:

- Best periodic 30-episode eval: `success_rate=0.1667` at update 20.
- Update 100 eval: `success_rate=0.1667`, `coverage_ratio=0.696`, `collision_rate=0.0090`.

Reasoning:

- Stronger dense coverage-ratio reward did not exceed the original BC baseline and made value fitting noisier.
- Coverage h200 remains unresolved. Current evidence points to missing high-success h200 trajectory/curriculum signal rather than PPO math or the factorized-group architecture alone.

## Theme AT: coverage optional milestone reward

Implemented:

- `tasks/coverage.py` now supports optional one-time coverage milestone bonuses configured by `reward_weights.coverage.milestone_thresholds` and `milestone_bonuses`.
- Added `configs/env/debug_coverage_milestone_reward.yaml`.
- Added `configs/policy/debug_ppo_coverage_factorized_group_h200_milestone.yaml`.
- Added `test_coverage_milestone_reward_pays_once` in `tests/test_rewards_basic.py`.

Reason:

- Coverage h200 is stuck near-but-below the canonical success threshold. Existing dense ratio shaping did not provide useful PPO improvement.
- Milestones provide clearer intermediate credit without changing success criteria, max steps, dynamics, observations, or task definition.

Verification:

- `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_variable_policies.py`
- Result: `22 passed`.

## Theme AU: coverage phase45 milestone PPO did not transfer to canonical task

Experiment:

- Ran `phase45_coverage_factorized_group_milestone` with optional milestone reward.
- Run: `outputs/training/bc_ppo/20260530_phase45_coverage_factorized_group_milestone/phase45_coverage_factorized_group_milestone/`.
- Canonical 100-episode eval: `outputs/debug_long/20260530_factorized_group_continuation/eval_phase45_coverage_canonical_100ep/coverage_N4_multi_channel_field_plus_task_id.csv`.

Result:

- Milestone-env best-source 30-episode eval: `success_rate_mean=0.233`, `collision_rate_mean=0.0`.
- Canonical h200 100-episode eval: `success_rate_mean=0.17`, `coverage_ratio_mean=0.715`, `collision_rate_mean=0.00236`, `repeated_coverage_ratio_mean=0.991`.

Reasoning:

- Milestone reward did not transfer to the canonical task. Coverage remains unresolved.
- Very high repeated coverage ratio shows the remaining failure mode is inefficient revisit/poor spatial allocation, not safety or action saturation.

## Theme AV: coverage terminal anti-revisit reward

Implemented:

- `tasks/coverage.py` now supports optional `reward_weights.coverage.terminal_repeated_coverage`.
- Added `configs/env/debug_coverage_antirevisit_reward.yaml`.
- Added `configs/policy/debug_ppo_coverage_factorized_group_h200_antirevisit.yaml`.
- Added `test_coverage_terminal_repeated_penalty_only_on_timeout`.

Reason:

- Phase45 canonical eval showed `repeated_coverage_ratio_mean=0.991`; the dominant failure mode is repeated revisiting, not collision.
- The new reward term penalizes final repeated coverage at timeout without changing success threshold, max steps, dynamics, or observations.

Verification:

- `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_variable_policies.py`
- Result: `23 passed`.

## Theme AW: coverage phase46 anti-revisit PPO did not transfer

Experiment:

- Ran `phase46_coverage_factorized_group_antirevisit` with optional terminal anti-revisit reward.
- Run: `outputs/training/bc_ppo/20260530_phase46_coverage_factorized_group_antirevisit/phase46_coverage_factorized_group_antirevisit/`.
- Canonical 100-episode eval: `outputs/debug_long/20260530_factorized_group_continuation/eval_phase46_coverage_canonical_100ep/coverage_N4_multi_channel_field_plus_task_id.csv`.

Result:

- Anti-revisit training config best-source 30-episode eval: `success_rate_mean=0.267`, `collision_rate_mean=0.0`.
- Canonical h200 100-episode eval: `success_rate_mean=0.20`, `coverage_ratio_mean=0.733`, `collision_rate_mean=0.00228`, `repeated_coverage_ratio_mean=0.99125`.

Reasoning:

- The reward term did not reduce repeated coverage under canonical validation. Coverage h200 remains unresolved.
- The remaining issue requires persistent frontier assignment or a stronger h200 teacher/curriculum, not more scalar reward-only tweaks of the same form.

## Theme AX: coverage canonical success-only dataset from phase46

Experiment:

- Collected successful canonical h200 coverage trajectories from phase46 best checkpoint.
- Dataset: `outputs/debug_long/20260530_factorized_group_continuation/coverage_success_from_phase46_canonical_plus_v3.npz`.
- Summary: `outputs/debug_long/20260530_factorized_group_continuation/coverage_success_from_phase46_canonical_plus_v3.summary.json`.

Result:

- 30 successful episodes from 142 attempts, success rate over attempts `0.2113`.
- 22643 total samples including base dataset.
- Mean success collision rate `0.0`, mean success path length `1.072`.

Reasoning:

- These are actual canonical h200 successes from the new architecture, so they are a better BC target than weak coverage heuristic teachers.

## Theme AY: coverage success-heavy BC failed

Experiment:

- Trained original factorized-group BC config on `outputs/debug_long/20260530_factorized_group_continuation/coverage_success_from_phase46_canonical_plus_v3.npz`.
- Run: `outputs/training/bc/20260530_163520/debug_bc_coverage_factorized_group_perm_coverage_N4_multi_channel_field_plus_task_id/`.

Result:

- Canonical 50-episode eval: `success_rate_mean=0.140`, `collision_rate_mean=0.0`, `return_mean=9.522`.

Reasoning:

- Rare success trajectory cloning overfits and lowers closed-loop robustness. Coverage remains unresolved.
- The next credible route is persistent frontier/group target assignment or a true high-success h200 teacher, not more BC/PPO from current weak data.

## Theme AZ: post-continuation regression test

Verification:

- Ran `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_variable_policies.py tests/test_bc_permutation_loss.py tests/test_expert_dataset.py tests/test_ppo_episode_budget.py`.
- Result: `26 passed`.

Current task status under factorized_group after this continuation:

- `goal_nav`: repaired from earlier phase36 evidence, 50-episode best eval `success_rate_mean=0.78`.
- `risk_nav`: provisionally repaired in this continuation. Phase41 canonical/safety 100-episode eval `success_rate_mean=0.65`, `collision_rate_mean=0.0208`.
- `formation`: provisionally repaired from earlier phase39 evidence, 50-episode best eval `success_rate_mean=0.44`, near-zero collision.
- `coverage`: still unresolved under canonical h200. Best reliable canonical 100-episode result remains about `success_rate_mean=0.20`, coverage ratio about `0.733`; repeated coverage remains about `0.991`.

## Theme BA: coverage frontier-slot policy for factorized_group

Implemented:

- Added optional `use_coverage_frontier_slot_head` to `policies/cnn_deepsets_policy.py`.
- Wired the same frontier slot bias into `policies/factorized_group_policy.py`.
- Added tests for CNNDeepSets and factorized-group frontier slot compatibility.
- Added configs:
  - `configs/policy/debug_bc_coverage_factorized_group_frontier_perm.yaml`
  - `configs/policy/debug_ppo_coverage_factorized_group_frontier.yaml`

Reason:

- Coverage canonical h200 remained stuck around `0.20` success with repeated coverage ratio around `0.991`.
- Reward-only tweaks and weak-teacher BC did not reduce repeated coverage.
- The new frontier slot explicitly assigns agents to remaining unvisited demand using stable sectors and suppression, while preserving centralized critic and per-agent actor output shape.

Verification:

- `/opt/conda/bin/python -m pytest -q tests/test_variable_policies.py tests/test_rewards_basic.py`
- Result: `25 passed`.

## Theme BB: coverage frontier-slot BC from scratch failed

Experiment:

- Trained `configs/policy/debug_bc_coverage_factorized_group_frontier_perm.yaml` from scratch on phase46 success-heavy dataset.
- Run: `outputs/training/bc/20260531_015126/debug_bc_coverage_factorized_group_frontier_perm_coverage_N4_multi_channel_field_plus_task_id/`.

Result:

- Canonical 50-episode eval: `success_rate_mean=0.10`, `collision_rate_mean=0.004`, `return_mean=8.239`.

Reasoning:

- Supervised loss is low but closed-loop behavior degraded. Warm-start from the original coverage BC checkpoint is the next safer route.

## Theme BC: coverage frontier-slot warm-start BC still below baseline

Experiment:

- Warm-started frontier BC from the original coverage BC checkpoint.
- Run: `outputs/training/bc/20260531_015932/debug_bc_coverage_factorized_group_frontier_perm_coverage_N4_multi_channel_field_plus_task_id/`.

Result:

- Canonical 50-episode eval: `success_rate_mean=0.16`, `collision_rate_mean=0.003`, `return_mean=8.434`.

Reasoning:

- Warm-start helped compared with frontier BC from scratch but did not beat the original baseline. Next attempt is conservative PPO from this checkpoint.

## Theme BD: coverage phase47 frontier PPO failed

Experiment:

- Ran `phase47_coverage_factorized_group_frontier` from warm-start frontier BC.
- Run: `outputs/training/bc_ppo/20260531_phase47_coverage_factorized_group_frontier/phase47_coverage_factorized_group_frontier/`.

Result:

- Best/final-source 30-episode eval: `success_rate_mean=0.167`, `collision_rate_mean=0.012`.

Reasoning:

- Frontier strength `0.22` remains too disruptive. Next attempt should use lower frontier strength and original BC checkpoint.

## Theme BE: low-strength coverage frontier PPO config

Implemented:

- Added `configs/policy/debug_ppo_coverage_factorized_group_frontier_low.yaml`.

Reason:

- Phase47 showed frontier strength `0.22` is too disruptive.
- The low-strength config uses frontier strength `0.08`, stronger reference regularization, lower target KL, and starts directly from the original coverage BC checkpoint.

## Theme BF: low-strength coverage frontier PPO failed

Experiment:

- Ran `phase48_coverage_factorized_group_frontier_low` from the original coverage BC checkpoint.
- Run: `outputs/training/bc_ppo/20260531_phase48_coverage_factorized_group_frontier_low/phase48_coverage_factorized_group_frontier_low/`.

Result:

- Update 20 success `0.033`, update 40 success `0.033`, update 60 success `0.067`.
- Stopped early after update 70.

Reasoning:

- Even low-strength immediate frontier bias damages the existing closed-loop coverage behavior. Do not continue this branch as-is.

## Theme BG: coverage h200 teacher prototypes failed

Diagnostics:

- Tested an in-shell one-step local greedy coverage teacher: success `0.0`, coverage `≈0.379`, collision `0.0`.
- Tested an in-shell waypoint-lookahead greedy coverage teacher: success `0.05`, coverage `≈0.608`, collision `≈0.0217`.

Reasoning:

- Simple greedy h200 teacher generation is insufficient. A useful coverage teacher likely needs explicit route planning/sweeping, not local scoring.
- No official teacher code was added for these failed prototypes.

## Theme BH: frontier continuation regression test

Verification:

- Ran `/opt/conda/bin/python -m pytest -q tests/test_variable_policies.py tests/test_rewards_basic.py tests/test_bc_permutation_loss.py tests/test_expert_dataset.py tests/test_ppo_episode_budget.py`.
- Result: `28 passed`.
- No training/evaluation process remained running after phase48 and teacher prototype diagnostics.

## Theme BI: specialist PPO reliability documentation

Implemented:

- Added `docs/specialist_ppo_reliability_zh.md`.

Content:

- Explains why pure PPO is unreliable for this benchmark: sparse success, high-dimensional joint continuous actions, long horizon, multi-objective reward conflict, credit assignment, and on-policy drift.
- Records why BC/DAgger warm-start is currently necessary rather than cosmetic.
- Summarizes current per-task training recipes and status under `factorized_group`.
- Documents why coverage remains unresolved and why the next route should be h200 route-planning/sweep teacher or stateful persistent target memory.

Reason:

- The user asked to preserve the recent analysis in both memory and docs so future agents can reuse the rationale instead of repeating failed PPO-only or reward-only attempts.

## Theme BJ: coverage training curriculum v1

Implemented:

- `tasks/coverage.py` now supports optional `coverage.demand_quantile`; default remains `0.55`.
- Added `configs/env/debug_coverage_train_curriculum_v1.yaml` with coverage-specific training environment/reward curriculum.
- Added `configs/policy/debug_ppo_coverage_factorized_group_train_curriculum_v1.yaml`.
- Added `test_coverage_demand_quantile_controls_demand_area`.

Reason:

- User now allows coverage-specific training environment and reward design changes.
- Canonical h200 coverage has not converged with reward-only scalar tweaks or frontier bias.
- Curriculum v1 narrows demand area, extends training horizon to 260, and strengthens completion/anti-repeat reward to bootstrap successful behavior.

Verification:

- `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_variable_policies.py`
- Result: `26 passed`.

## Theme BK: coverage phase49 curriculum v1 converged in modified training env

Experiment:

- Ran `phase49_coverage_train_curriculum_v1` from original factorized-group coverage BC checkpoint.
- Run: `outputs/training/bc_ppo/20260531_phase49_coverage_train_curriculum_v1/phase49_coverage_train_curriculum_v1/`.

Result:

- Training-env success rose from `0.60` at update 20 to `0.733` at update 60/160.
- Independent training-env 100-episode eval: `success_rate_mean=0.73`, `coverage_ratio_mean=0.742`, `return_mean=116.729`.
- Canonical 100-episode eval: `success_rate_mean=0.14`, `coverage_ratio_mean=0.712`.

Reasoning:

- Coverage PPO can converge when the coverage-specific training env is made curriculum-friendly.
- This is not yet a canonical repair; it should be used as a curriculum stage before a closer-to-canonical bridge.

## Theme BL: coverage curriculum v2 bridge config

Implemented:

- Added `configs/env/debug_coverage_train_curriculum_v2_bridge.yaml`.
- Added `configs/policy/debug_ppo_coverage_factorized_group_train_curriculum_v2_bridge.yaml`.

Reason:

- Phase49 converged in the easier curriculum env but did not transfer to canonical.
- V2 bridge moves closer to canonical by using demand quantile `0.60`, success ratio `0.80`, and max steps `240`.

## Theme BM: coverage phase50 bridge curriculum result

Experiment:

- Ran `phase50_coverage_train_curriculum_v2_bridge` from phase49 best checkpoint.
- Run: `outputs/training/bc_ppo/20260531_phase50_coverage_train_curriculum_v2_bridge/phase50_coverage_train_curriculum_v2_bridge/`.

Result:

- Bridge 100-episode eval: `success_rate_mean=0.56`, `coverage_ratio_mean=0.744`, `return_mean=59.100`, `collision_rate_mean=0.00323`.
- Canonical 100-episode eval: `success_rate_mean=0.13`, `coverage_ratio_mean=0.716`, `return_mean=9.777`.

Reasoning:

- Modified coverage training env can produce rising reward/success, but transfer to canonical h200 remains weak.
- As the curriculum approaches canonical, success falls, confirming that canonical coverage still needs stronger environment/reward redesign or route-planning supervision.

## Theme BN: coverage curriculum regression test

Verification:

- Ran `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_variable_policies.py tests/test_bc_permutation_loss.py tests/test_expert_dataset.py tests/test_ppo_episode_budget.py`.
- Result: `29 passed`.
- No active training/evaluation process remained after phase50 and evaluations.

## Theme BO: coverage anti-repeat reward and sequential group actor

Implemented:

- `tasks/coverage.py` now exposes two optional anti-revisit reward terms:
  - `reward_weights.coverage.repeated_demand_coverage`: per-step penalty for covering already-fulfilled demand cells, normalized by demand area.
  - `reward_weights.coverage.terminal_revisit_excess`: terminal penalty for cumulative excess demand visits, reported by new metric `demand_revisit_excess`.
- `policies/factorized_group_policy.py` now supports optional sequential group conditioning:
  - `use_sequential_group_context: true`
  - `sequential_group_context_strength`
  - Later group tokens are conditioned on previous group target coordinates, so group decisions are no longer independent when this option is enabled.
- Fixed policy factory propagation for `coverage_frontier_*` options. Before this fix, frontier-slot configs constructed a valid policy but did not actually enable the frontier head.
- Added `configs/env/debug_coverage_train_curriculum_v3_antirepeat.yaml`.
- Added `configs/policy/debug_ppo_coverage_factorized_group_seq_frontier_antirepeat.yaml`.
- Added tests for configurable demand revisit penalties and sequential group context.

Reasoning:

- Previous canonical/bridge coverage runs had `repeated_coverage_ratio≈0.99`, so policies learned to sit in or revisit already covered areas while still receiving partial coverage reward.
- The new per-step demand revisit term gives dense negative feedback exactly when a policy wastes coverage radius on already-fulfilled demand cells.
- The new terminal revisit-excess term separates “same final coverage with efficient sweep” from “same final coverage after excessive revisits”.
- Sequential group conditioning implements the user-requested grouped decision mode where later group decisions can see earlier group outputs while preserving centralized critic, shared per-agent decoder, `[B, N, 2]` output, `agent_mask`, and variable-N support.
- The frontier factory fix means future frontier experiments are now real; earlier frontier phase47/48 results should be interpreted cautiously because the config knobs were not actually wired through.

Verification:

- `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_variable_policies.py tests/test_policy_action_distribution.py`
- Result: `31 passed`.
- Checkpoint compatibility probe from phase50 best into the new sequential policy succeeded with only the 4 new sequential projection parameters missing.
- 2-update PPO smoke:
  - Run: `outputs/training/bc_ppo/20260601_phase51_smoke/phase51_smoke_seq_frontier_antirepeat/`
  - Init: phase50 best checkpoint.
  - Result: training/eval path completed; 2-episode eval `success_rate=0.50`, `collision_rate=0.0`, return negative because anti-repeat penalties are intentionally much stronger.

Long experiment launched:

- Phase51 run command started with `setsid`.
- PID file: `outputs/debug_long/20260601_phase51_coverage_seq_frontier_antirepeat/train.pid`.
- Log file: `outputs/debug_long/20260601_phase51_coverage_seq_frontier_antirepeat/train.log`.
- Expected run dir: `outputs/training/bc_ppo/20260601_phase51_coverage_seq_frontier_antirepeat/phase51_coverage_seq_frontier_antirepeat/`.
- Init checkpoint: phase50 bridge best.
- Configs:
  - `configs/env/debug_coverage_train_curriculum_v3_antirepeat.yaml`
  - `configs/policy/debug_ppo_coverage_factorized_group_seq_frontier_antirepeat.yaml`

Phase51 early stop:

- Stopped at update 45 after update 40 eval showed no recovery.
- Update 20 eval: `success_rate=0.133`, `coverage_ratio=0.696479`, `repeated_coverage_ratio=0.991442`, `demand_revisit_excess=31.645258`.
- Update 40 eval: `success_rate=0.133`, `coverage_ratio=0.696357`, `repeated_coverage_ratio=0.991393`, `demand_revisit_excess=31.673179`.
- Interpretation: turning on real frontier bias plus new sequential parameters plus very large terminal revisit-excess penalty perturbed the phase50 checkpoint too much and did not reduce repeated coverage.

Implemented phase52:

- Added `configs/env/debug_coverage_train_curriculum_v3_moderate_antirepeat.yaml`.
- Added `configs/policy/debug_ppo_coverage_factorized_group_seq_moderate_antirepeat.yaml`.
- Differences from phase51:
  - frontier slot disabled to preserve the phase50 behavior distribution;
  - sequential group context kept but weakened to `0.15`;
  - anti-repeat reward remains active but terminal dominance is reduced:
    - `repeated_coverage=-0.35`
    - `repeated_demand_coverage=-2.0`
    - `terminal_repeated_coverage=-18.0`
    - `terminal_revisit_excess=-3.0`
- Reason: phase51 proved that huge terminal penalties create value noise and do not teach mid-episode retargeting. Phase52 tests whether weak sequential conditioning plus moderate dense revisit cost can improve without destroying the phase50 bridge policy.

Phase52 launched:

- PID file: `outputs/debug_long/20260601_phase52_coverage_seq_moderate_antirepeat/train.pid`.
- Log file: `outputs/debug_long/20260601_phase52_coverage_seq_moderate_antirepeat/train.log`.
- Expected run dir: `outputs/training/bc_ppo/20260601_phase52_coverage_seq_moderate_antirepeat/phase52_coverage_seq_moderate_antirepeat/`.

Phase52 early stop and control eval:

- Phase52 was stopped after update 20 because eval stayed poor:
  - `success_rate=0.1667`
  - `coverage_ratio=0.718132`
  - `repeated_coverage_ratio=0.991589`
  - `demand_revisit_excess=29.152401`
- Control eval of the unchanged phase50 policy on the same `v3_moderate_antirepeat` env:
  - CSV: `outputs/debug_long/20260601_phase52_coverage_seq_moderate_antirepeat/eval_phase50_on_v3_moderate_30ep/coverage_N4_multi_channel_field_plus_task_id.csv`
  - `success_rate=0.50`
  - `coverage_ratio=0.747009`
  - `repeated_coverage_ratio=0.991306`
  - `demand_revisit_excess=28.143183`
  - `return=-64.43454`
- Interpretation: the moderate anti-repeat environment itself preserves phase50 behavior; the weak sequential group context still perturbs the checkpoint enough to damage success.

Implemented phase53:

- Added `configs/policy/debug_ppo_coverage_factorized_group_moderate_antirepeat.yaml`.
- This keeps the original factorized-group architecture from phase50 and trains only under the moderate anti-repeat reward.
- Reason: first establish whether the reward change can improve phase50 without architecture perturbation. If yes, sequential group context should be added later via BC/refit or a gated zero-init path, not by direct PPO warm-start with random new parameters.

Phase53 launched:

- PID file: `outputs/debug_long/20260601_phase53_coverage_moderate_antirepeat/train.pid`.
- Log file: `outputs/debug_long/20260601_phase53_coverage_moderate_antirepeat/train.log`.
- Expected run dir: `outputs/training/bc_ppo/20260601_phase53_coverage_moderate_antirepeat/phase53_coverage_moderate_antirepeat/`.

Phase53 early stop:

- Stopped after update 40 eval failed to recover to the phase50 control.
- Update 20: `success_rate=0.3667`, `coverage_ratio=0.738746`, `repeated_coverage_ratio=0.990699`, `demand_revisit_excess=28.518539`.
- Update 40: `success_rate=0.3667`, `coverage_ratio=0.729345`, `repeated_coverage_ratio=0.990817`, `demand_revisit_excess=28.495747`.
- Interpretation: moderate anti-repeat PPO without architecture changes is less damaging than phase51/52 but still drifts below the phase50 control (`success=0.50`) and does not reduce revisit metrics enough.

Implemented phase54:

- Added `configs/policy/debug_ppo_coverage_factorized_group_moderate_antirepeat_safe.yaml`.
- Same architecture and env as phase53, but more conservative PPO:
  - `learning_rate=2e-6`
  - `reference_policy_coef=1.0`
  - `clip_coef=0.01`
  - `target_kl=0.0008`
  - `max_grad_norm=0.35`
  - `log_std_max=-1.25`
- Reason: phase53 shows PPO is moving the policy out of the success basin faster than anti-repeat reward can improve routing. Phase54 tests whether tighter behavior regularization can preserve phase50 success while allowing slow reward adaptation.

Phase54 launched:

- PID file: `outputs/debug_long/20260601_phase54_coverage_moderate_antirepeat_safe/train.pid`.
- Log file: `outputs/debug_long/20260601_phase54_coverage_moderate_antirepeat_safe/train.log`.
- Expected run dir: `outputs/training/bc_ppo/20260601_phase54_coverage_moderate_antirepeat_safe/phase54_coverage_moderate_antirepeat_safe/`.

Phase54 interim result:

- Update 20: `success_rate=0.4667`, `coverage_ratio=0.745157`, `repeated_coverage_ratio=0.990504`, `demand_revisit_excess=27.586895`, `return=-75.093002`.
- Update 40: `success_rate=0.5000`, `coverage_ratio=0.746011`, `repeated_coverage_ratio=0.990377`, `demand_revisit_excess=27.777411`, `return=-64.570771`.
- Update 60: `success_rate=0.5333`, `coverage_ratio=0.743630`, `repeated_coverage_ratio=0.990713`, `demand_revisit_excess=27.829508`, `return=-53.686137`.
- Interpretation:
  - Conservative PPO finally preserves and slightly improves phase50 bridge success (`0.50 -> 0.5333`).
  - Return improves under the anti-repeat reward.
  - Revisit metrics are only slightly better than the phase50 control (`demand_revisit_excess≈28.14`) and remain high, so this is not a complete coverage fix yet.
  - Continue phase54 to completion, then run independent 100-episode eval on `v3_moderate_antirepeat` and canonical `configs/env/coverage.yaml`.

Phase54 stopped and independently evaluated:

- Stopped after update 100 drifted down (`success_rate=0.4333`) from the best update 60.
- Best checkpoint remained update 60:
  - `outputs/training/bc_ppo/20260601_phase54_coverage_moderate_antirepeat_safe/phase54_coverage_moderate_antirepeat_safe/checkpoints/checkpoint_best_eval.pt`
- Independent v3 moderate 100-episode eval:
  - CSV: `outputs/debug_long/20260601_phase54_coverage_moderate_antirepeat_safe/eval_phase54_best_v3_moderate_100ep/coverage_N4_multi_channel_field_plus_task_id.csv`
  - `success_rate=0.56`
  - `coverage_ratio=0.745690`
  - `repeated_coverage_ratio=0.990990`
  - `demand_revisit_excess=27.358407`
  - `return=-44.033868`
  - `collision_rate=0.003417`
  - `path_length=1.166319`
- Phase50 control on the same v3 moderate env was:
  - `success_rate=0.50`
  - `coverage_ratio=0.747009`
  - `repeated_coverage_ratio=0.991306`
  - `demand_revisit_excess=28.143183`
  - `return=-64.434540`
- Independent canonical 100-episode eval:
  - CSV: `outputs/debug_long/20260601_phase54_coverage_moderate_antirepeat_safe/eval_phase54_best_canonical_100ep/coverage_N4_multi_channel_field_plus_task_id.csv`
  - `success_rate=0.12`
  - `coverage_ratio=0.711085`
  - `repeated_coverage_ratio=0.991357`
  - `demand_revisit_excess=25.194444`
  - `return=9.568191`
  - `collision_rate=0.002713`
  - `path_length=1.099539`
- Phase50 canonical transfer was `success_rate=0.13`, `coverage_ratio=0.716001`, so phase54 improves the modified training env but does not improve canonical transfer.

Conclusion:

- The safe anti-repeat PPO recipe is the best modified-env continuation so far for coverage (`v3_moderate` 100-episode success `0.56`).
- It is not a canonical repair. Canonical h200 remains around `0.12-0.13` success.
- Sequential group context cannot be introduced by direct PPO warm-start with random new parameters; it needs BC/refit or a zero-impact/gated initialization before PPO.

Implemented phase55:

- Added `configs/env/debug_coverage_train_curriculum_v4_canonical_bridge.yaml`.
- Added `configs/policy/debug_ppo_coverage_factorized_group_canonical_bridge_safe.yaml`.
- Purpose: continue from phase54 best toward canonical coverage:
  - `max_steps=220`
  - `success_ratio=0.81`
  - `demand_quantile=0.55`
  - moderate anti-repeat rewards retained but weaker than v3
  - same conservative PPO settings as phase54.
- Verification:
  - `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_variable_policies.py tests/test_policy_action_distribution.py`
  - Result: `31 passed`.

Phase55 launched:

- PID file: `outputs/debug_long/20260601_phase55_coverage_canonical_bridge_safe/train.pid`.
- Log file: `outputs/debug_long/20260601_phase55_coverage_canonical_bridge_safe/train.log`.
- Expected run dir: `outputs/training/bc_ppo/20260601_phase55_coverage_canonical_bridge_safe/phase55_coverage_canonical_bridge_safe/`.
- Init checkpoint: phase54 best.

Phase55 result:

- Run completed:
  - `outputs/training/bc_ppo/20260601_phase55_coverage_canonical_bridge_safe/phase55_coverage_canonical_bridge_safe/`
- Best checkpoint:
  - `outputs/training/bc_ppo/20260601_phase55_coverage_canonical_bridge_safe/phase55_coverage_canonical_bridge_safe/checkpoints/checkpoint_best_eval.pt`
  - best update `100`
- Best/final 30-episode bridge eval:
  - `success_rate=0.30`
  - `coverage_ratio=0.725249`
  - `repeated_coverage_ratio=0.991341`
  - `demand_revisit_excess=27.166739`
  - `return=-99.392137`
  - `collision_rate=0.002576`
- Interpretation:
  - Phase55 is not an improvement over phase54's v3-moderate 100-episode result (`success_rate=0.56`).
  - It does show partial bridge transfer above canonical (`0.30` vs canonical `0.12`), but the closer-to-canonical demand/horizon still breaks most successful completion behavior.
  - Revisit metrics remain high. The remaining issue is not just scalar reward weight; the policy lacks a robust route/sweep behavior under wider demand.

Next implication:

- Stop pure PPO continuation as the main coverage path.
- Move to data-side repair: collect successful and preferably lower-revisit trajectories from phase54/phase55 environments, then BC/refit before any further PPO.

Phase56 data-side repair attempt:

- Enhanced `scripts/debug_long/collect_success_policy_dataset.py`:
  - added `--max_demand_revisit_excess`
  - added `--max_repeated_coverage_ratio`
  - summary now records mean successful `repeated_coverage_ratio` and `demand_revisit_excess`.
- Collected phase54 v3-moderate low-revisit success data:
  - dataset: `outputs/debug_long/20260601_phase56_coverage_low_revisit_data/phase54_v3_success_lowrevisit.npz`
  - summary: `outputs/debug_long/20260601_phase56_coverage_low_revisit_data/phase54_v3_success_lowrevisit.summary.json`
  - `30` successful episodes from `140` attempts
  - `4848` samples
  - mean successful `demand_revisit_excess=27.160990`
  - mean successful `repeated_coverage_ratio=0.990324`
  - mean successful collision `0.0000406`
- Collected phase55 v4-bridge low-revisit success data:
  - dataset: `outputs/debug_long/20260601_phase56_coverage_low_revisit_data/phase55_v4_success_lowrevisit.npz`
  - summary: `outputs/debug_long/20260601_phase56_coverage_low_revisit_data/phase55_v4_success_lowrevisit.summary.json`
  - `20` successful episodes from `112` attempts
  - `3453` samples
  - mean successful `demand_revisit_excess=25.644311`
  - mean successful `repeated_coverage_ratio=0.990960`
  - mean successful collision `0.000100`
- Merged data:
  - dataset: `outputs/debug_long/20260601_phase56_coverage_low_revisit_data/coverage_phase54_phase55_lowrevisit_success_merged.npz`
  - `8301` samples.
- Warm-start BC/refit:
  - init checkpoint: phase54 best
  - config: `configs/policy/debug_bc_coverage_factorized_group_perm.yaml`
  - dataset: merged low-revisit success data above
  - run: `outputs/training/bc/20260601_021316/debug_bc_coverage_factorized_group_perm_coverage_N4_multi_channel_field_plus_task_id/`
  - v4 bridge final eval: `success_rate=0.30`, `coverage_ratio=0.725430`, `demand_revisit_excess=27.419515`, `return=-100.167997`.

Conclusion:

- Low-revisit success-only BC from current policy did not improve over phase55.
- It mostly reproduces the same partial bridge behavior.
- Next coverage route needs a stronger teacher/route-planning data source rather than more current-policy success filtering.

Phase57 sweep teacher diagnostic:

- Added `CoverageExpertV5` in `scripts/debug_long/generate_coverage_expert_v2_dataset.py`.
- V5 is a diagnostic stateful lawnmower/stripe-sweep teacher:
  - partitions demand cells into x-stripes;
  - assigns one persistent route per agent;
  - follows the route until local demand is fulfilled;
  - includes simple repulsion for collision avoidance.
- 50-episode diagnostic rollout:
  - v4 bridge env:
    - `success=0.82`
    - `coverage_ratio=0.802366`
    - `repeated_coverage_ratio=0.985556`
    - `demand_revisit_excess=23.714672`
    - `collision_rate=0.0`
    - `path_length=0.876563`
  - canonical env:
    - `success=0.66`
    - `coverage_ratio=0.804135`
    - `repeated_coverage_ratio=0.985571`
    - `demand_revisit_excess=23.708844`
    - `collision_rate=0.0`
    - `path_length=0.880012`
- Generated formal V5 datasets:
  - bridge: `outputs/debug_long/20260601_phase57_coverage_sweep_teacher/v5_bridge_e100.npz`
    - terminal success `0.80`
    - `17160` samples
  - canonical: `outputs/debug_long/20260601_phase57_coverage_sweep_teacher/v5_canonical_e100.npz`
    - terminal success `0.63`
    - `16979` samples
  - merged: `outputs/debug_long/20260601_phase57_coverage_sweep_teacher/v5_bridge_canonical_e200_merged.npz`
    - `34139` samples
    - terminal success `0.715`
- BC on V5 merged data:
  - init: phase54 best
  - config: `configs/policy/debug_bc_coverage_factorized_group_perm.yaml`
  - run: `outputs/training/bc/20260601_022614/debug_bc_coverage_factorized_group_perm_coverage_N4_multi_channel_field_plus_task_id/`
  - canonical 50-episode final eval:
    - `success_rate=0.30`
    - `coverage_ratio=0.758937`
    - `demand_revisit_excess=23.682952`
    - `repeated_coverage_ratio=0.991763`
    - `collision_rate=0.0`
    - `return=13.441922`

Interpretation:

- V5 proves that persistent route/sweep behavior solves much of coverage: teacher canonical success `0.66`.
- Plain factorized-group BC does not fully inherit this behavior because the actor is memoryless; it sees visited map but does not maintain a route pointer or persistent assigned target.
- Static BC improves coverage ratio and demand revisit excess, but not success enough.
- Next implementation should put persistence into the policy/decision mechanism, not only into the teacher.

Phase58 stateless lawnmower route-head attempt:

- Implemented optional policy route bias:
  - `use_coverage_lawnmower_route_head`
  - `coverage_lawnmower_route_strength`
  - `coverage_lawnmower_*` weights
- Added factory wiring in `policies/__init__.py`.
- Added factorized-group forward support.
- Added tests for factorized-group lawnmower route head.
- Added configs:
  - `configs/policy/debug_bc_coverage_factorized_group_lawnmower_perm.yaml`
  - `configs/policy/debug_ppo_coverage_factorized_group_lawnmower_safe.yaml`
- Verification:
  - `/opt/conda/bin/python -m pytest -q tests/test_variable_policies.py tests/test_policy_action_distribution.py`
  - Result: `22 passed`.

Diagnostic results:

- Direct phase54 best checkpoint evaluated with route-head config on canonical:
  - CSV: `outputs/debug_long/20260601_phase58_lawnmower_head_eval/eval_phase54_plus_lawnmower_canonical_30ep/coverage_N4_multi_channel_field_plus_task_id.csv`
  - `success_rate=0.0333`
  - `coverage_ratio=0.672943`
  - `demand_revisit_excess=27.246735`
  - `collision_rate=0.014792`
- Route-head BC run was interrupted/ended after epoch 4:
  - run: `outputs/training/bc/20260601_025152/debug_bc_coverage_factorized_group_lawnmower_perm_coverage_N4_multi_channel_field_plus_task_id/`
  - epoch4 checkpoint canonical eval:
    - CSV: `outputs/debug_long/20260601_phase58_lawnmower_head_eval/eval_lawnmower_bc_epoch4_canonical_30ep/coverage_N4_multi_channel_field_plus_task_id.csv`
    - `success_rate=0.1333`
    - `coverage_ratio=0.719913`
    - `demand_revisit_excess=24.180358`
    - `collision_rate=0.009542`
- For comparison, V5 BC without route head was better:
  - `success_rate=0.30`
  - `coverage_ratio=0.758937`
  - `demand_revisit_excess=23.682952`
  - collision `0`.

Interpretation:

- The stateless route-head is not a valid replacement for V5 persistence.
- It recomputes stripe targets each step and can conflict with the learned residual actor, causing collisions and lower success.
- Do not continue this exact lawnmower-head config.
- The strong result remains the V5 teacher itself; if using it for learning, the model needs genuine temporal persistence/recurrent state or route pointer information, not just a per-step deterministic bias.

Phase59 route-hint observation implementation:

- Stopped the still-running phase58 route-head BC process because phase58 was already empirically worse than V5 BC and should not consume more compute.
- Added optional coverage env/task persistence without changing observation shape:
  - `coverage.route_hint_enabled`
  - `coverage.route_hint_stride`
  - `coverage.route_hint_sigma`
  - `coverage.route_hint_value_quantile`
- Implementation location:
  - `tasks/coverage.py`
- Design:
  - CoverageTask now maintains persistent per-agent stripe/lawnmower routes in `task_state`.
  - The current route target map is exposed through the existing `formation_template` channel only when `coverage.route_hint_enabled=true`.
  - This avoids changing `CHANNEL_NAMES` / task field shape, so old CNN checkpoints remain loadable.
  - Persistence is in env/task state, not in `policy.forward()`, so PPO minibatch logprob recomputation remains valid.
- Added optional policy-side route-hint bias:
  - `use_coverage_route_hint_head`
  - `coverage_route_hint_strength`
  - `coverage_route_hint_temperature`
  - `coverage_route_hint_distance_weight`
- Implementation location:
  - `policies/cnn_deepsets_policy.py`
  - `policies/factorized_group_policy.py`
  - `policies/__init__.py`
- Added phase59 configs:
  - `configs/env/debug_coverage_route_hint_canonical.yaml`
  - `configs/policy/debug_bc_coverage_factorized_group_route_hint_perm.yaml`
  - `configs/policy/debug_ppo_coverage_factorized_group_route_hint_safe.yaml`
- Rationale:
  - V5 teacher succeeded because it has persistent route memory.
  - Stateless route-head failed because it recomputed targets every step and fought the learned residual actor.
  - Route-hint observation makes persistence part of the Markov observation instead of hidden policy state, which is the safe path for PPO.

Phase59 direct route-hint checkpoint diagnostics:

- Evaluated phase54 best checkpoint with route-hint env/policy:
  - CSV: `outputs/debug_long/20260601_phase59_coverage_route_hint/eval_phase54_route_hint_50ep/coverage_N4_multi_channel_field_plus_task_id.csv`
  - `success_rate=0.0`
  - `coverage_ratio=0.531557`
  - `collision_rate=0.008700`
  - `path_length=1.096755`
  - `repeated_coverage_ratio=0.988878`
  - `demand_revisit_excess=24.384818`
  - `return=-219.611562`
- Evaluated V5-BC checkpoint with route-hint env/policy before sequential suppression:
  - CSV: `outputs/debug_long/20260601_phase59_coverage_route_hint/eval_v5bc_route_hint_50ep/coverage_N4_multi_channel_field_plus_task_id.csv`
  - `success_rate=0.0`
  - `coverage_ratio=0.594574`
  - `collision_rate=0.014675`
  - `path_length=1.078980`
  - `repeated_coverage_ratio=0.988729`
  - `demand_revisit_excess=26.474563`
  - `return=-213.821676`
- Added route-hint sequential suppression in the policy head:
  - `coverage_route_hint_suppression_strength`
  - `coverage_route_hint_suppression_sigma`
  - intent: later agents should avoid already selected route-hint peaks, approximating sequential group assignment while keeping `forward()` stateless and replay-safe.
- Tests:
  - `/opt/conda/bin/python -m pytest -q tests/test_task_fields.py tests/test_variable_policies.py tests/test_policy_action_distribution.py`
  - Result: `29 passed`.
- Evaluated V5-BC checkpoint with suppression:
  - CSV: `outputs/debug_long/20260601_phase59_coverage_route_hint/eval_v5bc_route_hint_suppression_30ep/coverage_N4_multi_channel_field_plus_task_id.csv`
  - `success_rate=0.0`
  - `coverage_ratio=0.615120`
  - `collision_rate=0.013708`
  - `path_length=1.099243`
  - `repeated_coverage_ratio=0.990041`
  - `demand_revisit_excess=27.814795`
  - `return=-213.428082`
- Interpretation:
  - Directly adding route-hint bias to old checkpoints is not viable.
  - The CNN/actor was not trained with this channel and the route-hint map alone lacks a supervised association to the residual behavior.
  - Continue with route-hint-aware V5 data generation and BC, not more direct checkpoint+bias evaluation.

Phase59 route-hint BC data:

- Generated V5 teacher data under route-hint env:
  - dataset: `outputs/debug_long/20260601_phase59_coverage_route_hint/v5_route_hint_canonical_e120.npz`
  - samples: `20210`
  - episodes: `120`
  - terminal success: `0.641667`
  - route-hint channel nonzero fraction: `1.0`
- First BC attempt failed before training due route-hint suppression OOM:
  - reason: suppression built a full 64x64 token pair matrix for batch 256, requiring about 32GB CUDA memory.
- Fix:
  - added `coverage_route_hint_pool_size` with default `16`.
  - route-hint head now adaptive-max-pools the hint channel before sequential suppression, matching the existing utility/frontier head pattern.

Phase59 route-hint BC result:

- Route-hint-head BC completed:
  - run: `outputs/training/bc/20260601_032154/debug_bc_coverage_factorized_group_route_hint_perm_coverage_N4_multi_channel_field_plus_task_id/`
  - dataset: `outputs/debug_long/20260601_phase59_coverage_route_hint/v5_route_hint_canonical_e120.npz`
  - env: `configs/env/debug_coverage_route_hint_canonical.yaml`
  - final eval: `success_rate=0.26`, `coverage_ratio=0.744232`, `collision_rate=0.006125`, `path_length=1.077288`, `repeated_coverage_ratio=0.990756`, `demand_revisit_excess=25.625035`, `return=-103.843146`.
- Interpretation:
  - This is not better than V5 BC without route head (`success_rate=0.30`).
  - The route-hint action bias still interferes with the learned residual actor.
  - Next attempt should keep the route-hint channel in observation but disable the explicit route-hint action head, so the CNN can learn whether/how to use the channel.
- Added:
  - `configs/policy/debug_bc_coverage_factorized_group_route_hint_obs_perm.yaml`

Phase60 per-agent route-target observation implementation:

- Added optional agent observation extension:
  - config key: `include_route_targets_in_agents`
  - location: `envs/centralized_env.py`
  - default: `false`, so old configs/checkpoints keep the original 6-D agent token.
- When enabled, each agent token appends:
  - route target delta normalized by map size: `[dx, dy]`
  - route target position normalized by map size: `[tx, ty]`
  - resulting agent observation shape: `[N, 10]`.
- The route targets come from `CoverageTask`'s persistent route state:
  - `task_state["route_hint_targets"]`
  - updated when the route-hint field is built.
- Added BC warm-start compatibility filtering in `scripts/train_bc.py`:
  - only loads checkpoint tensors whose keys and shapes match the current model;
  - skips mismatched tensors such as the first agent encoder layer when agent observation size changes.
- Added configs:
  - `configs/env/debug_coverage_route_target_agents_canonical.yaml`
  - `configs/policy/debug_bc_coverage_factorized_group_route_target_agents_perm.yaml`
  - `configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_safe.yaml`
- Reason:
  - route-hint shared heatmap did not give a reliable per-agent route pointer.
  - V5 teacher is strong because each agent has its own persistent route index; appending target coordinates gives the per-agent actor this missing information without hidden mutable policy state.

Phase60 route-target-agent BC result:

- Fixed `CentralizedMultiUAVEnv` observation space shape for optional agent extensions:
  - `spaces.Box(shape=agent_low.shape)` instead of hard-coded `(num_agents, 6)`.
- Tests:
  - `/opt/conda/bin/python -m pytest -q tests/test_task_fields.py tests/test_variable_policies.py tests/test_policy_action_distribution.py`
  - Result: `31 passed`.
- Generated V5 teacher data with per-agent route target features:
  - dataset: `outputs/debug_long/20260601_phase60_coverage_route_target_agents/v5_route_target_agents_canonical_e120.npz`
  - samples: `20210`
  - episodes: `120`
  - terminal success: `0.641667`
  - agent tensor shape: `(20210, 4, 10)`
  - mean absolute route delta features: `0.118764`
- BC run:
  - run: `outputs/training/bc/20260601_034310/debug_bc_coverage_factorized_group_route_target_agents_perm_coverage_N4_multi_channel_field_plus_task_id/`
  - config: `configs/policy/debug_bc_coverage_factorized_group_route_target_agents_perm.yaml`
  - env: `configs/env/debug_coverage_route_target_agents_canonical.yaml`
  - init: V5-BC no-route checkpoint, with only `agent_encoder.0.weight` skipped due shape mismatch.
- Final 50-episode eval:
  - `success_rate=0.66`
  - `coverage_ratio=0.797840`
  - `collision_rate=0.005549`
  - `path_length=0.882761`
  - `repeated_coverage_ratio=0.986634`
  - `demand_revisit_excess=23.928779`
  - `return=19.242915`
- Interpretation:
  - This is the first coverage learning result that matches the V5 teacher success scale.
  - The decisive fix is not a scalar reward tweak; it is exposing per-agent persistent route targets to the actor while keeping centralized global context.
  - This supports the user's requested decision model: each agent receives its own waypoint-relevant target, while policy/critic still share full state.

Phase60 route-target-agent PPO result:

- PPO run:
  - run: `outputs/training/bc_ppo/20260601_phase60_coverage_route_target_agents_ppo/phase60_coverage_route_target_agents_ppo/`
  - init checkpoint: `outputs/training/bc/20260601_034310/debug_bc_coverage_factorized_group_route_target_agents_perm_coverage_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0032.pt`
  - config: `configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_safe.yaml`
  - env: `configs/env/debug_coverage_route_target_agents_canonical.yaml`
- PPO periodic eval:
  - update 20: `success_rate=0.667`
  - update 40: `success_rate=0.667`
  - update 60: `success_rate=0.667`
  - update 80: `success_rate=0.667`
  - update 100: `success_rate=0.667`
  - interpretation: conservative PPO preserved the BC expert; no PPO collapse.
- PPO final 30-episode eval:
  - `success_rate=0.666667`
  - `coverage_ratio=0.803509`
  - `collision_rate=0.001875`
  - `path_length=0.900458`
  - `repeated_coverage_ratio=0.986835`
  - `demand_revisit_excess=24.186381`
  - `return=23.276407`
- Independent best-checkpoint 100-episode eval:
  - CSV: `outputs/debug_long/20260601_phase60_coverage_route_target_agents/eval_phase60_best_route_target_100ep/coverage_N4_multi_channel_field_plus_task_id.csv`
  - checkpoint: `outputs/training/bc_ppo/20260601_phase60_coverage_route_target_agents_ppo/phase60_coverage_route_target_agents_ppo/checkpoints/checkpoint_best_eval.pt`
  - `success_rate=0.72`
  - `coverage_ratio=0.801508`
  - `collision_rate=0.002905`
  - `path_length=0.881318`
  - `repeated_coverage_ratio=0.986430`
  - `demand_revisit_excess=23.424259`
  - `return=35.787137`
- Current coverage conclusion:
  - Under the new per-agent route-target decision mode, coverage is now functioning and PPO fine-tuning is stable.
  - This is not the original canonical `configs/env/coverage.yaml`; it is a coverage environment/observation design change explicitly allowed by the user for coverage debugging.

## Theme AM: formation template-aware success repair and 100-episode validation

Implemented on 2026-06-01:

- Fixed `FormationTask.get_metrics(...)` success logic in `tasks/formation.py`.
- Previous issue:
  - formation success required low slot error, low radius error, and high angular uniformity for every template.
  - This is valid for radial templates such as `circle` and `diamond`.
  - It is structurally wrong for `line` and too strict for `arc`, because those templates are not angularly uniform around the center even when UAVs exactly match their assigned slots.
- New success logic:
  - all templates require `formation_error <= formation_tolerance`;
  - `circle` and `diamond` additionally require radius error and angular uniformity;
  - `arc` additionally requires radius error;
  - `line` uses slot matching error as the success condition.
- Added regression test:
  - `tests/test_rewards_basic.py::test_formation_line_template_success_uses_slot_error_not_angular_uniformity`
  - It places UAVs exactly on a line template, confirms angular uniformity is below the old radial threshold, and confirms success is now true.
- Test command:
  - `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_variable_policies.py tests/test_policy_action_distribution.py`
  - Result: `35 passed`.

Formation phase61 validation:

- Baseline before success-definition repair:
  - checkpoint: `outputs/training/bc_ppo/20260530_phase39_formation_factorized_group_ppo/phase39_formation_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt`
  - CSV: `outputs/debug_long/20260601_phase61_formation_validation/eval_phase39_best_100ep/formation_N4_multi_channel_field_plus_task_id.csv`
  - 100 episodes, seed 7
  - `success_rate=0.50`
  - `formation_error=0.067085`
  - `radius_error=0.055980`
  - `angular_coverage_uniformity=0.601650`
  - `collision_rate=0.002550`
  - `path_length=0.540066`
  - `return=28.816939`
- After template-aware success repair, same checkpoint:
  - CSV: `outputs/debug_long/20260601_phase61_formation_validation/eval_phase39_best_template_success_100ep/formation_N4_multi_channel_field_plus_task_id.csv`
  - 100 episodes, seed 7
  - `success_rate=0.77`
  - `formation_error=0.072942`
  - `radius_error=0.055500`
  - `angular_coverage_uniformity=0.633523`
  - `collision_rate=0.002313`
  - `path_length=0.446007`
  - `return=29.534165`
- Repeat seed validation:
  - added `configs/env/debug_formation_seed23.yaml`
  - CSV: `outputs/debug_long/20260601_phase61_formation_validation/eval_phase39_best_template_success_seed23_100ep/formation_N4_multi_channel_field_plus_task_id.csv`
  - 100 episodes, seed 23
  - `success_rate=0.78`
  - `formation_error=0.073723`
  - `radius_error=0.055759`
  - `angular_coverage_uniformity=0.639514`
  - `collision_rate=0.002313`
  - `path_length=0.439148`
  - `return=29.182813`

Current formation conclusion:

- Formation is now repaired under the `factorized_group` specialist path.
- The repair is primarily a task-metric correctness fix, not a PPO hyperparameter fix.
- The same phase39 checkpoint now has coverage-level verification strength: two independent 100-episode evaluations with `success_rate≈0.77-0.78`, low collision, and stable formation error.

## Theme AN: risk_nav repeat-seed validation after formation phase61

Implemented on 2026-06-01:

- Added repeat-seed eval config:
  - `configs/env/debug_risk_nav_seed23.yaml`
  - It keeps the phase41 risk-nav task, obstacle/risk settings, and reward weights unchanged.
  - Only the seed is changed to `23`; this is a validation config, not an environment/reward tuning change.
- Evaluated phase41 best `factorized_group` checkpoint:
  - checkpoint: `outputs/training/bc_ppo/20260530_phase41_risknav_factorized_group_dagger_safe/phase41_risknav_factorized_group_dagger_safe/checkpoints/checkpoint_best_eval.pt`
  - policy config: `configs/policy/debug_ppo_risk_nav_factorized_group_dagger_safe.yaml`
  - eval env: `configs/env/debug_risk_nav_seed23.yaml`
  - output CSV: `outputs/debug_long/20260601_phase62_risknav_validation/eval_phase41_best_seed23_100ep/risk_nav_N4_multi_channel_field_plus_task_id.csv`
  - command: `/opt/conda/bin/python scripts/evaluate_policy.py --checkpoint outputs/training/bc_ppo/20260530_phase41_risknav_factorized_group_dagger_safe/phase41_risknav_factorized_group_dagger_safe/checkpoints/checkpoint_best_eval.pt --policy-config configs/policy/debug_ppo_risk_nav_factorized_group_dagger_safe.yaml --env-config configs/env/debug_risk_nav_seed23.yaml --tasks risk_nav --agent_counts 4 --scaling_mode fixed_map --obs_variant multi_channel_field+task_id --episodes 100 --output-dir outputs/debug_long/20260601_phase62_risknav_validation/eval_phase41_best_seed23_100ep`

Phase62 result:

- `success_rate=0.65`
- `goal_coverage_ratio=0.841667`
- `collision_rate=0.018796`
- `path_length=0.712840`
- `cumulative_risk_exposure=34.320727`
- `safety_violation_count=39.08`
- `return=3.544033`

Comparison with previous seed 7 100-episode eval:

- seed 7 CSV: `outputs/debug_long/20260530_factorized_group_continuation/eval_phase41_risknav_100ep/risk_nav_N4_multi_channel_field_plus_task_id.csv`
- seed 7 metrics: `success_rate=0.65`, `goal_coverage_ratio=0.85`, `collision_rate=0.020797`, `path_length=0.698778`, `cumulative_risk_exposure=33.393036`.
- seed 23 reproduces the same success level and similar safety/path metrics.

Current risk_nav conclusion:

- `risk_nav` should now be treated as repaired/reproducible under the phase41 `factorized_group` specialist path.
- It is not as clean as `goal_nav` or `formation`: success is stable at about `0.65`, and cumulative risk/safety violation metrics remain non-trivial.
- The effective recipe remains learner-state DAgger BC plus conservative PPO with reference regularization, not pure PPO from scratch.

Post-phase62 regression tests:

- Command: `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_variable_policies.py tests/test_policy_action_distribution.py`
- Result: `35 passed in 4.88s`.

## Theme AO: coverage route-target repeat-seed validation

Implemented on 2026-06-01:

- Added repeat-seed coverage eval config:
  - `configs/env/debug_coverage_route_target_agents_seed23.yaml`
  - It keeps the phase60 route-target-agent coverage task design, reward weights, success ratio, demand quantile, and route hint settings unchanged.
  - Only the seed is changed to `23`; this is a validation config, not a new coverage tuning change.
- Evaluated phase60 best `factorized_group` checkpoint:
  - checkpoint: `outputs/training/bc_ppo/20260601_phase60_coverage_route_target_agents_ppo/phase60_coverage_route_target_agents_ppo/checkpoints/checkpoint_best_eval.pt`
  - policy config: `configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_safe.yaml`
  - eval env: `configs/env/debug_coverage_route_target_agents_seed23.yaml`
  - output CSV: `outputs/debug_long/20260601_phase63_coverage_route_target_validation/eval_phase60_best_route_target_seed23_100ep/coverage_N4_multi_channel_field_plus_task_id.csv`
  - command: `/opt/conda/bin/python scripts/evaluate_policy.py --checkpoint outputs/training/bc_ppo/20260601_phase60_coverage_route_target_agents_ppo/phase60_coverage_route_target_agents_ppo/checkpoints/checkpoint_best_eval.pt --policy-config configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_safe.yaml --env-config configs/env/debug_coverage_route_target_agents_seed23.yaml --tasks coverage --agent_counts 4 --scaling_mode fixed_map --obs_variant multi_channel_field+task_id --episodes 100 --output-dir outputs/debug_long/20260601_phase63_coverage_route_target_validation/eval_phase60_best_route_target_seed23_100ep`

Phase63 result:

- `success_rate=0.68`
- `coverage_ratio=0.795507`
- `collision_rate=0.004842`
- `path_length=0.879962`
- `repeated_coverage_ratio=0.986651`
- `demand_revisit_excess=23.553093`
- `return=22.621489`

Comparison with previous phase60 seed 7 100-episode eval:

- seed 7 CSV: `outputs/debug_long/20260601_phase60_coverage_route_target_agents/eval_phase60_best_route_target_100ep/coverage_N4_multi_channel_field_plus_task_id.csv`
- seed 7 metrics: `success_rate=0.72`, `coverage_ratio=0.801508`, `collision_rate=0.002905`, `path_length=0.881318`, `demand_revisit_excess=23.424259`.
- seed 23 is slightly lower success but reproduces the same general behavior: high coverage ratio, low collision, similar path length, and similar demand revisit excess.

Current coverage conclusion:

- Coverage is now repaired/reproducible under the new per-agent route-target observation/decision mode.
- Two independent 100-episode seeds give `success_rate=0.68-0.72`.
- The main remaining quality issue is still repeated coverage (`repeated_coverage_ratio≈0.986`) and `demand_revisit_excess≈23.5`; this is not blocking expert usability but is the next improvement target if coverage quality must be raised further.

## Theme AP: goal_nav repeat-seed audit

Implemented on 2026-06-01:

- Added repeat-seed goal_nav eval config:
  - `configs/env/debug_goal_nav_seed23.yaml`
  - It keeps the phase36 goal-nav env settings unchanged except for `seed: 23`.
- Evaluated phase36 best `factorized_group` checkpoint:
  - checkpoint: `outputs/training/bc_ppo/20260530_phase36_goalnav_factorized_group_ppo/phase36_goalnav_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt`
  - policy config: `configs/policy/debug_ppo_goal_nav_factorized_group_finetune.yaml`
  - eval env: `configs/env/debug_goal_nav_seed23.yaml`
  - output CSV: `outputs/debug_long/20260601_phase64_goalnav_validation/eval_phase36_best_seed23_100ep/goal_nav_N4_multi_channel_field_plus_task_id.csv`
  - command: `/opt/conda/bin/python scripts/evaluate_policy.py --checkpoint outputs/training/bc_ppo/20260530_phase36_goalnav_factorized_group_ppo/phase36_goalnav_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt --policy-config configs/policy/debug_ppo_goal_nav_factorized_group_finetune.yaml --env-config configs/env/debug_goal_nav_seed23.yaml --tasks goal_nav --agent_counts 4 --scaling_mode fixed_map --obs_variant multi_channel_field+task_id --episodes 100 --output-dir outputs/debug_long/20260601_phase64_goalnav_validation/eval_phase36_best_seed23_100ep`

Phase64 result:

- `success_rate=0.66`
- `goal_coverage_ratio=0.804167`
- `collision_rate=0.072423`
- `path_length=0.553423`
- `completion_time=129.4`
- `remaining_goal_ratio=0.195833`
- `return=5.738511`

Comparison with previous phase36 seed 7 evidence:

- training final/best eval seed 7: `success_rate=0.80`, `goal_coverage_ratio=0.8725`, `collision_rate=0.063120`, `path_length=0.486920`.
- `eval_best_50` seed 7: `success_rate=0.78`, `goal_coverage_ratio=0.877`, `collision_rate=0.056029`, `path_length=0.510952`.

Current goal_nav conclusion:

- `goal_nav` remains usable under `factorized_group`, but the new seed 23 100-episode audit shows weaker robustness than previously assumed.
- Do not overstate goal_nav as two-seed high-stability: current evidence is seed 7 `0.78-0.80` vs seed 23 `0.66`.
- The main remaining issue is collision/safety and seed generalization, not basic task learning.
- If strict four-task robustness is required before multi-task training, next best action is a short safety/generalization continuation from phase36 with conservative PPO and possibly stronger collision/obstacle reward, then re-evaluate both seed 7 and seed 23.

Documentation update:

- Added `docs/specialist_expert_baselines_zh.md`.
- Purpose: freeze the current single-task specialist baseline table before downstream multi-task work.
- The table lists each task's reusable checkpoint, policy config, env/eval config, main evidence, and caveat.

## Theme AQ: goal_nav phase65 safety/generalization continuation

Implemented on 2026-06-01:

- Added safety/generalization continuation env:
  - `configs/env/debug_goal_nav_seed23_safety.yaml`
  - Same goal-nav task setup as phase36/phase64 seed23, but with stronger reward penalties for `collision`, `obstacle_collision`, and `safety_violation`.
  - This is a reward-only hardening experiment; task success threshold and task mechanics are unchanged.
- Added conservative reference-PPO config:
  - `configs/policy/debug_ppo_goal_nav_factorized_group_safety_ref.yaml`
  - Main differences from phase36:
    - `learning_rate=5e-6`
    - `clip_coef=0.01`
    - `target_kl=0.001`
    - `reference_policy_coef=0.75`
    - `total_updates=80`
- PPO continuation command:
  - `/opt/conda/bin/python scripts/train_ppo.py --config configs/policy/debug_ppo_goal_nav_factorized_group_safety_ref.yaml --env-config configs/env/debug_goal_nav_seed23_safety.yaml --tasks goal_nav --agent_counts 4 --scaling_mode fixed_map --init_checkpoint outputs/training/bc_ppo/20260530_phase36_goalnav_factorized_group_ppo/phase36_goalnav_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt --obs_variant multi_channel_field+task_id --eval_episodes 30 --total_updates 80 --headless --run_timestamp 20260601_phase65_goalnav_safety_ref --run_name phase65_goalnav_safety_ref`
- PPO run:
  - `outputs/training/bc_ppo/20260601_phase65_goalnav_safety_ref/phase65_goalnav_safety_ref/`
  - best checkpoint: `outputs/training/bc_ppo/20260601_phase65_goalnav_safety_ref/phase65_goalnav_safety_ref/checkpoints/checkpoint_best_eval.pt`
  - best training eval: update 60, `eval_success_rate=0.766667`, `eval_reward=6.130733`.

Independent original-env validation:

- seed 23, original reward/env:
  - CSV: `outputs/debug_long/20260601_phase65_goalnav_safety_ref/eval_phase65_best_seed23_100ep/goal_nav_N4_multi_channel_field_plus_task_id.csv`
  - `success_rate=0.71`
  - `goal_coverage_ratio=0.824667`
  - `collision_rate=0.064165`
  - `path_length=0.542029`
  - `completion_time=123.7`
  - `remaining_goal_ratio=0.175333`
  - `return=6.666644`
- seed 7, original reward/env:
  - CSV: `outputs/debug_long/20260601_phase65_goalnav_safety_ref/eval_phase65_best_seed7_100ep/goal_nav_N4_multi_channel_field_plus_task_id.csv`
  - `success_rate=0.72`
  - `goal_coverage_ratio=0.824500`
  - `collision_rate=0.065032`
  - `path_length=0.536958`
  - `completion_time=121.82`
  - `remaining_goal_ratio=0.175500`
  - `return=6.916867`

Interpretation:

- Phase65 improves the weak seed23 audit relative to phase36:
  - `success_rate: 0.66 -> 0.71`
  - `collision_rate: 0.072423 -> 0.064165`
  - `goal_coverage_ratio: 0.804167 -> 0.824667`
- But it lowers seed7 relative to phase36:
  - phase36 seed7 was `success_rate=0.78-0.80`;
  - phase65 seed7 is `success_rate=0.72`.
- Therefore phase65 is a robustness alternative, not an unconditional replacement for phase36.
- Keep phase36 as the best seed7/high-success checkpoint; use phase65 if balanced seed7/seed23 behavior matters more than peak seed7 success.

## Theme AR: goal_nav seed robustness diagnosis

Implemented on 2026-06-01:

- Added diagnostic script:
  - `scripts/debug_long/diagnose_goal_nav_seed_robustness.py`
- Purpose:
  - explain why `goal_nav` is usable but less seed-robust;
  - separate env/task hardness from learned-policy failure modes;
  - record per-episode initial geometry, action alignment, pair collision, obstacle collision, goal coverage, and success.
- Script supports both learned checkpoints and heuristic baselines via `--baseline-policy`.

Diagnostic outputs:

- Directory: `outputs/debug_long/20260601_phase66_goalnav_seed_robustness/`
- Human analysis:
  - `outputs/debug_long/20260601_phase66_goalnav_seed_robustness/analysis.md`
- Main CSV/JSON outputs:
  - `phase36_peak_v2_episodes.csv`
  - `phase36_peak_v2_seed_summary.csv`
  - `phase36_peak_v2_success_failure_summary.csv`
  - `phase65_balanced_v2_episodes.csv`
  - `phase65_balanced_v2_seed_summary.csv`
  - `phase65_balanced_v2_success_failure_summary.csv`
  - `greedy_goal_baseline_seed_summary.csv`
  - `phase36_seed23_100diag_episodes.csv`
  - `phase65_seed23_100diag_episodes.csv`

Commands run:

- phase36 multi-seed:
  - `/opt/conda/bin/python scripts/debug_long/diagnose_goal_nav_seed_robustness.py --checkpoint outputs/training/bc_ppo/20260530_phase36_goalnav_factorized_group_ppo/phase36_goalnav_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt --policy-config configs/policy/debug_ppo_goal_nav_factorized_group_finetune.yaml --env-config configs/env/goal_nav.yaml --seeds 7 11 17 23 29 --episodes 40 --obs_variant multi_channel_field+task_id --output-dir outputs/debug_long/20260601_phase66_goalnav_seed_robustness --label phase36_peak_v2`
- phase65 multi-seed:
  - `/opt/conda/bin/python scripts/debug_long/diagnose_goal_nav_seed_robustness.py --checkpoint outputs/training/bc_ppo/20260601_phase65_goalnav_safety_ref/phase65_goalnav_safety_ref/checkpoints/checkpoint_best_eval.pt --policy-config configs/policy/debug_ppo_goal_nav_factorized_group_safety_ref.yaml --env-config configs/env/goal_nav.yaml --seeds 7 11 17 23 29 --episodes 40 --obs_variant multi_channel_field+task_id --output-dir outputs/debug_long/20260601_phase66_goalnav_seed_robustness --label phase65_balanced_v2`
- greedy baseline:
  - `/opt/conda/bin/python scripts/debug_long/diagnose_goal_nav_seed_robustness.py --baseline-policy greedy_goal --env-config configs/env/goal_nav.yaml --seeds 7 11 17 23 29 --episodes 40 --obs_variant multi_channel_field+task_id --output-dir outputs/debug_long/20260601_phase66_goalnav_seed_robustness --label greedy_goal_baseline`
- seed23 100-episode diagnostics:
  - phase36: `/opt/conda/bin/python scripts/debug_long/diagnose_goal_nav_seed_robustness.py --checkpoint outputs/training/bc_ppo/20260530_phase36_goalnav_factorized_group_ppo/phase36_goalnav_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt --policy-config configs/policy/debug_ppo_goal_nav_factorized_group_finetune.yaml --env-config configs/env/goal_nav.yaml --seeds 23 --episodes 100 --obs_variant multi_channel_field+task_id --output-dir outputs/debug_long/20260601_phase66_goalnav_seed_robustness --label phase36_seed23_100diag`
  - phase65: `/opt/conda/bin/python scripts/debug_long/diagnose_goal_nav_seed_robustness.py --checkpoint outputs/training/bc_ppo/20260601_phase65_goalnav_safety_ref/phase65_goalnav_safety_ref/checkpoints/checkpoint_best_eval.pt --policy-config configs/policy/debug_ppo_goal_nav_factorized_group_safety_ref.yaml --env-config configs/env/goal_nav.yaml --seeds 23 --episodes 100 --obs_variant multi_channel_field+task_id --output-dir outputs/debug_long/20260601_phase66_goalnav_seed_robustness --label phase65_seed23_100diag`

Key results:

- Five eval seeds, 40 episodes per seed:
  - phase36 peak: `success_rate=0.755`, `goal_coverage_ratio=0.8635`, `pair_collision_rate=0.0606`.
  - phase65 balanced: `success_rate=0.770`, `goal_coverage_ratio=0.8763`, `pair_collision_rate=0.0560`.
  - greedy goal baseline: `success_rate=0.775`, `goal_coverage_ratio=0.9371`, `pair_collision_rate=0.0281`.
- Phase36 success vs failure episodes:
  - success: `goal_coverage=1.0`, `pair_collision=0.0174`, `obstacle_collision=0.0474`, `mean_action_alignment=0.704`, `mean_action_norm=0.809`, `steps=88.7`.
  - failure: `goal_coverage=0.443`, `pair_collision=0.1937`, `obstacle_collision=0.0895`, `mean_action_alignment=0.281`, `mean_action_norm=0.494`, `steps=200`.
- Phase65 success vs failure episodes:
  - success: `goal_coverage=1.0`, `pair_collision=0.0228`, `obstacle_collision=0.0460`, `mean_action_alignment=0.711`, `mean_action_norm=0.809`, `steps=87.5`.
  - failure: `goal_coverage=0.462`, `pair_collision=0.1673`, `obstacle_collision=0.0738`, `mean_action_alignment=0.263`, `mean_action_norm=0.541`, `steps=200`.
- Seed23 100-episode explanation:
  - phase36 seed23 overall `success_rate=0.66`;
  - episodes 0-39: `0.725`;
  - episodes 40-69: `0.567`;
  - episodes 70-99: `0.667`.
  - phase65 improves the hard middle/later portions: `0.667` and `0.733`, giving overall `0.71`.

Conclusion:

- Seed23 is not uniquely broken; robustness varies by task instance.
- The dominant failure mode is closed-loop recovery/collision robustness:
  - failures have high pair collisions and obstacle collisions;
  - after entering bad local configurations, action alignment to remaining goals collapses and action norm drops;
  - episodes time out at 200 steps with about 40-46% goal coverage.
- Initial geometry matters, but weakly:
  - failures tend to have more goals (`~4.3-4.4` vs `~3.9`) and slightly harder goal-agent geometry;
  - phase65 failures often have goals closer to obstacles.
- The learned policy is worse than the greedy baseline in pair collisions and goal coverage, so this is not purely environment hardness.

Recommended next fixes:

- Do not lower success thresholds.
- Add collision/recovery-focused DAgger data from failed or near-failed phase36/phase65 rollouts.
- Consider explicit goal assignment / anti-crowding auxiliary context for goal_nav, analogous to route targets used for coverage.
- Consider small learned-free inter-agent repulsion or sequential group context for goal_nav, followed by BC/PPO continuation.
- Validate any replacement checkpoint on seeds `7/11/17/23/29` and at least one 100-episode seed23 run.

## Theme AS: unified per-agent task-target hints for goal_nav robustness

Implemented on 2026-06-01:

- Generalized the coverage route-target observation idea into a task-level optional observation extension:
  - config key: `include_task_targets_in_agents`
  - location: `envs/centralized_env.py`
  - default: `false`
  - backward-compatible alias: existing `include_route_targets_in_agents` still works.
- When enabled, each agent token appends:
  - target delta normalized by map size: `[dx, dy]`
  - target position normalized by map size: `[tx, ty]`
  - agent observation shape becomes `[N, 10]`.
- Per-task target semantics:
  - `goal_nav` / `risk_nav`: Hungarian assignment to remaining goals.
  - if remaining goals are fewer than agents, only one agent is assigned to each remaining goal and extra agents receive their current position as an idle target. This directly reduces crowding at late-stage single-goal states.
  - `coverage`: reuses persistent `route_hint_targets`.
  - `formation`: Hungarian assignment to formation slots.
- Added regression test:
  - `tests/test_task_fields.py::test_goal_nav_task_targets_can_be_appended_to_agent_observation`
- Test command:
  - `/opt/conda/bin/python -m pytest -q tests/test_task_fields.py tests/test_variable_policies.py tests/test_policy_action_distribution.py tests/test_rewards_basic.py`
  - Result: `43 passed in 5.59s`.

Configs added:

- `configs/env/debug_goal_nav_task_targets.yaml`
- `configs/policy/debug_bc_goal_nav_factorized_group_task_targets.yaml`
- `configs/policy/debug_ppo_goal_nav_factorized_group_task_targets_safe.yaml`

Phase67 goal_nav task-target BC:

- Generated success-only heuristic dataset:
  - dataset: `outputs/debug_long/20260601_phase67_goalnav_task_targets/goal_nav_task_targets_success_e100.npz`
  - samples: `3769`
  - successful episodes: `100`
  - attempts: `148`
  - teacher success over attempts: `0.675676`
  - mean successful collision rate: `0.002538`
- BC run:
  - `outputs/training/bc/20260601_125256/debug_bc_goal_nav_factorized_group_task_targets_goal_nav_N4_multi_channel_field_plus_task_id/`
  - checkpoint: `outputs/training/bc/20260601_125256/debug_bc_goal_nav_factorized_group_task_targets_goal_nav_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0024.pt`
  - warm-start: phase36 best checkpoint, skipping only `agent_encoder.0.weight` because agent observation changed from 6-D to 10-D.
- BC final 30-episode eval:
  - `success_rate=0.966667`
  - `goal_coverage_ratio=0.993333`
  - `collision_rate=0.007881`
  - `path_length=0.434771`
- BC five-seed diagnostic, seeds `7/11/17/23/29`, 40 episodes per seed:
  - overall `success_rate=0.95`
  - overall `goal_coverage_ratio=0.978333`
  - overall `pair_collision_rate=0.007909`
  - seed 23: `success_rate=0.925`, `goal_coverage_ratio=0.96`, `pair_collision_rate=0.009979`
- BC seed23 100-episode diagnostic:
  - `success_rate=0.89`
  - `goal_coverage_ratio=0.9415`
  - `pair_collision_rate=0.017221`
- Analysis doc:
  - `outputs/debug_long/20260601_phase67_goalnav_task_targets/analysis.md`

Phase68 goal_nav task-target PPO:

- PPO run:
  - `outputs/training/bc_ppo/20260601_phase68_goalnav_task_targets_ppo/phase68_goalnav_task_targets_ppo/`
  - init: phase67 BC checkpoint `checkpoint_0024.pt`
  - config: `configs/policy/debug_ppo_goal_nav_factorized_group_task_targets_safe.yaml`
  - env: `configs/env/debug_goal_nav_task_targets.yaml`
- Training eval:
  - update 20: `success_rate=0.967`
  - update 40: `success_rate=0.967`
  - update 60: `success_rate=0.967`
  - final collision: `0.006`
- Independent seed23 100-episode checkpoint sweep:
  - `checkpoint_0020.pt`: `success_rate=0.89`, `goal_coverage_ratio=0.9395`, `pair_collision_rate=0.017782`
  - `checkpoint_0040.pt`: `success_rate=0.89`, `goal_coverage_ratio=0.9395`, `pair_collision_rate=0.016639`
  - `checkpoint_0060.pt` / `checkpoint_best_eval.pt`: `success_rate=0.88`, `goal_coverage_ratio=0.9355`, `pair_collision_rate=0.019207`
- Recommended robust PPO checkpoint:
  - `outputs/training/bc_ppo/20260601_phase68_goalnav_task_targets_ppo/phase68_goalnav_task_targets_ppo/checkpoints/checkpoint_0040.pt`
- checkpoint_0040 five-seed diagnostic, seeds `7/11/17/23/29`, 40 episodes per seed:
  - overall `success_rate=0.965`
  - overall `goal_coverage_ratio=0.984333`
  - overall `pair_collision_rate=0.005324`
  - seed 23: `success_rate=0.95`, `goal_coverage_ratio=0.97`, `pair_collision_rate=0.006831`
- Analysis doc:
  - `outputs/debug_long/20260601_phase68_goalnav_task_targets_ppo/analysis.md`

Interpretation:

- This is the first goal_nav branch that directly fixes the seed robustness failure mode found in phase66.
- The improvement comes from giving the shared per-agent actor an explicit target-assignment context, not from lowering thresholds or changing action shape.
- It aligns with the coverage route-target solution and is extensible to risk_nav/formation, so it is a better multi-task-compatible direction than a goal_nav-only reward trick.
- Current recommended goal_nav expert should be updated from phase36 to phase68 `checkpoint_0040.pt` for robust PPO use, with phase67 BC as the strongest pure imitation checkpoint.

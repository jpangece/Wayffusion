# 建议的下一步工作

## 高优先级

### 1. 转移主线到 coverage specialist 修复

建议：

- 继续 coverage 的 expert-v2 BC -> spatial-head PPO 主线
- 优先观察 `coverage_ratio` 是否能稳定跨过 `0.5`、`0.6`、再接近 `0.82`
- 当前 best-final coverage PPO 已经把 specialist 推到更高一点的成功区：
  - best eval `success≈0.15`
  - `coverage_ratio≈0.673`
  - `collision≈0.00019`
  - run root:
    `outputs/training/bc_ppo/20260529_phase24_coverage_successonly_v2_bestfinal/phase24_coverage_successonly_v2_bestfinal/`
- 当前更具体的 coverage 主线是：
  - `expert-v2 dataset -> spatial-head BC -> spatial-head PPO`
  - 如果继续推进，优先用成功策略轨迹继续做 success-heavy BC / DAgger，而不是继续单纯加 reward scale
  - 这条成功轨迹强化链已经做过一轮：success-heavy BC、sector-bias、stronger-repulsion 都没有超过 `phase10`
  - 因此下一步应升级到真正的 group-level coordination actor，而不是继续做轻量偏置微调
- canonical heuristic BC 只保留为结构探针，不作为 coverage 主线

原因：

- `goal_nav` 当前已经有一条可复用 specialist 收敛链，并且 PPO final eval 已稳定在 `success_rate_mean=0.8`
- `coverage` 现在才是剩余的主要阻塞项
- 可选 `global slot head` 已经接入代码并通过跨任务兼容性测试，因此真正的 group-level actor 分支现在具备可实验条件

### 2. 刷新正式 verification 文件

建议：

- 重新生成一个和当前目录结构、测试数量、命令入口一致的 `outputs/verification.md`

原因：

- 当前文件已明显陈旧，会误导后续 agent 和用户

### 3. 统一配置命名

建议：

- 在 `ppo_mlp / mlp_ppo`
- `ppo_cnn_deepsets / cnn_deepsets_ppo`

两套命名中选一套作为主路径，并把另一套标记为 legacy 或 alias。

原因：

- 当前重复命名已经进入维护负担区间

### 4. 为评估脚本补 `--latest`

建议：

- 给 `evaluate_policy.py` 和 `evaluate_scaling.py` 增加最新 run 自动发现

原因：

- 训练目录现在有时间层，手工填 checkpoint 路径更容易出错

## 中优先级

### 5. 在 coverage 主线稳定后，再刷新 canonical specialist / multitask runs

建议：

- 用当前新的 `checkpoints/` 与 `snapshot/` 规范，跑一条标准多任务 PPO 训练线
- 同时保留 eval media 和 final eval artifacts

原因：

- 这样后续所有 agent 都能引用“新结构下的标准 run”

### 6. 为 agent_memory 增加自动生成脚本

建议：

- 后续可考虑增加脚本，把当前测试数、最新 run、最新 checkpoint 自动写回 manifest 或某张卡

原因：

- 当前 memory cards 是人工审计产物，后面若频繁修改，自动刷新会更稳

## 低优先级

### 7. 处理 `sitecustomize.py` 重复

建议：

- 明确是否保留双副本

### 8. 历史 outputs 分层归档

建议：

- 把旧结构 run 和新结构 run 分层标记

原因：

- 未来自动扫 `outputs/` 做比较时会更干净

## 路由建议

如果后续 agent 需要继续主线开发，建议优先路线是：

1. Keep `goal_nav` factorized-group as repaired:
   `outputs/training/bc_ppo/20260530_phase36_goalnav_factorized_group_ppo/phase36_goalnav_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt`.
2. Keep `formation` factorized-group as provisionally repaired:
   `outputs/training/bc_ppo/20260530_phase39_formation_factorized_group_ppo/phase39_formation_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt`.
3. Risk-nav under factorized-group is still not repaired:
   current best 50-episode eval is `success_rate_mean=0.24`, with collision `≈0.177`.
4. Next risk-nav route should not be simple repulsion/reference or low-collision-only BC; both failed.
5. For risk-nav, try a richer DAgger dataset from learner states relabeled by the risk heuristic, or add a task-specific risk/goal spatial head while keeping `policy_class: factorized_group`.
6. Coverage remains unresolved under h200 and weaker than old h300 under new architecture:
   new factorized-group h300 best is `success_rate_mean=0.40`; old spatial-head h300 best is `0.56`.
7. For coverage, next new-architecture route is group-specific coverage suppression or persistent group targets, not more heuristic V4/band-sweep data.
8. Refresh `outputs/verification.md`, checkpoint latest helpers, and canonical multitask runs after risk_nav and coverage have new-architecture specialists.

## 当前 continuation 路线：只推进 factorized_group

约束：

- 不再调旧 `CNNDeepSetsPolicy` 主线。
- 不改环境动力学、任务定义、成功阈值或核心任务要素。
- 允许 reward-only 配置实验，但必须显式记录为 debug/training config。

立即执行：

1. `risk_nav`: 用 `scripts/debug_long/collect_dagger_dataset.py` 从 factorized-group learner rollout 采样状态，并用 heuristic teacher relabel，输出到 `outputs/debug_long/20260530_factorized_group_continuation/`。
2. `risk_nav`: 用原 success dataset + DAgger dataset 训练 factorized-group BC，再用 `configs/policy/debug_ppo_risk_nav_factorized_group_dagger_safe.yaml` 和 `configs/env/debug_risk_nav_safety_completion.yaml` 做保守 PPO。
3. `coverage`: 从 factorized-group h300 best checkpoint 继续 completion-focused PPO，判断剩余问题是 episode budget/credit assignment 还是架构表达不足。
4. 每个新增数据集、训练 run、代码或配置修改都必须同步写回 `agent_memory/cards/03_recent_modifications.md` 和本卡。

Risk-nav continuation update:

- DAgger BC improved factorized-group risk-nav to 30-episode `success_rate_mean=0.533`, but collision remains high at `0.114`.
- Next action is conservative PPO from `outputs/training/bc/20260530_121227/debug_bc_risk_nav_factorized_group_risk_nav_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0040.pt` using `configs/policy/debug_ppo_risk_nav_factorized_group_dagger_safe.yaml` and `configs/env/debug_risk_nav_safety_completion.yaml`.

Risk-nav status after phase41:

- Treat `risk_nav` as provisionally repaired under `factorized_group`.
- Current best checkpoint: `outputs/training/bc_ppo/20260530_phase41_risknav_factorized_group_dagger_safe/phase41_risknav_factorized_group_dagger_safe/checkpoints/checkpoint_best_eval.pt`.
- Evidence: independent 100-episode eval `success_rate_mean=0.65`, `collision_rate_mean=0.0208`.
- Remaining validation: run at least one seed repeat after coverage is stabilized.

Coverage next:

- Do not use h300 as the canonical success claim because the user disallowed environment changes except reward.
- Continue with h200 and reward-only completion shaping first.

Coverage h200 experiment to run:

- Start from factorized-group coverage BC checkpoint `outputs/training/bc/20260530_083743/debug_bc_coverage_factorized_group_perm_coverage_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0024.pt`.
- Train with `configs/policy/debug_ppo_coverage_factorized_group_completion_h200.yaml` and `configs/env/debug_coverage_completion_focus.yaml`.
- Evaluate with at least 30 episodes during training and 100 episodes for the best checkpoint before claiming repair.

Coverage phase42 result:

- Direct utility-slot PPO failed and was stopped early.
- Do not continue phase42.
- Next action: train BC with the utility-enabled factorized-group policy on the existing coverage dataset, then PPO fine-tune from that BC checkpoint.

Coverage utility BC result:

- Utility-head BC failed despite low supervised loss: 30-episode success `0.0`.
- Do not PPO from `outputs/training/bc/20260530_135952/...`.
- Next action: coverage learner-state DAgger using phase37 factorized-group coverage learner as rollout policy and `coverage_expert_v2` as teacher, then BC/PPO with the original non-utility factorized-group config.

Coverage teacher diagnostic result:

- Existing coverage teachers are not h200-successful; do not expect expert-v2/v3/v4 DAgger alone to solve coverage.
- Next action: run `configs/policy/debug_ppo_coverage_factorized_group_h200_stable.yaml` from the original factorized-group coverage BC checkpoint under `configs/env/debug_coverage_completion_focus.yaml`.

Coverage phase43 result:

- Stable h200 PPO from original BC did not exceed the original `0.20` success baseline.
- Current best canonical h200 factorized-group coverage remains around `success_rate_mean=0.20`.
- Next route should be reward-only coverage curriculum/shaping or a newly designed high-success h200 coverage teacher; do not repeat expert-v2/v3/v4 DAgger or utility-head PPO as-is.

Coverage ratio-curriculum next:

- Run `configs/policy/debug_ppo_coverage_factorized_group_h200_ratio_curriculum.yaml` from the original factorized-group BC checkpoint under `configs/env/debug_coverage_ratio_curriculum.yaml`.
- If it improves success above `0.20`, do independent 100-episode eval. If not, coverage remains blocked on high-success h200 expert/curriculum design.

Coverage phase44 result:

- Ratio-curriculum reward-only PPO failed to improve beyond `0.20` baseline; best was `0.1667`.
- If using lower-threshold curriculum next, mark it explicitly as training-only/diagnostic and validate final checkpoint under canonical `coverage.success_ratio=0.82`, h200.

Coverage milestone PPO next:

- Run `configs/policy/debug_ppo_coverage_factorized_group_h200_milestone.yaml` with `configs/env/debug_coverage_milestone_reward.yaml` from the original factorized-group BC checkpoint.
- If best 30-episode success exceeds `0.20`, run 100-episode canonical eval. If it fails, coverage remains blocked on stronger curriculum/high-success h200 expert design.

Coverage phase45 canonical validation:

- Milestone PPO is not a canonical repair. Canonical 100-episode success was `0.17` with repeated coverage ratio `0.991`.
- Next credible route: implement an anti-revisit / frontier-sweep training signal or a high-success h200 teacher, then validate under original `configs/env/coverage.yaml`.

Coverage anti-revisit PPO next:

- Run `configs/policy/debug_ppo_coverage_factorized_group_h200_antirevisit.yaml` with `configs/env/debug_coverage_antirevisit_reward.yaml` from the original factorized-group BC checkpoint.
- Canonical validation remains mandatory under `configs/env/coverage.yaml`.

Coverage phase46 canonical validation:

- Anti-revisit PPO is not a canonical repair. Canonical 100-episode success is `0.20`, repeated coverage remains `0.99125`.
- Stop repeating scalar reward-only PPO from the same BC checkpoint.
- Next credible implementation: add a factorized-group-compatible persistent target/frontier assignment mechanism or create a true high-success h200 coverage teacher, then validate under canonical `configs/env/coverage.yaml`.

Coverage success-heavy BC result:

- Success-only BC from phase46 successful trajectories failed: 50-episode canonical success `0.140`.
- Stop using current success-only data as the main coverage fix.
- Next implementation should be architectural but compatible with `factorized_group`: persistent per-agent/group frontier target assignment, or alternatively build a true h200 coverage teacher before BC/PPO.

Post-continuation status:

- Regression tests passed: `26 passed`.
- Do not claim coverage repaired. Current canonical h200 coverage is still around `0.20` success.
- Best next coverage route is not another scalar reward tweak; implement persistent per-agent/group frontier targets in the new architecture, or first build a true high-success h200 coverage teacher.

Coverage frontier-slot next:

- Train `configs/policy/debug_bc_coverage_factorized_group_frontier_perm.yaml` on `outputs/debug_long/20260530_factorized_group_continuation/coverage_success_from_phase46_canonical_plus_v3.npz`.
- If BC canonical eval is not worse than baseline, run `configs/policy/debug_ppo_coverage_factorized_group_frontier.yaml` under canonical or anti-revisit reward and validate with canonical 100 episodes.

Coverage frontier-low next:

- Run `configs/policy/debug_ppo_coverage_factorized_group_frontier_low.yaml` from the original coverage BC checkpoint under anti-revisit reward.
- If it does not exceed `0.20`, frontier bias as currently implemented should be considered ineffective and the next route should be a better h200 teacher rather than more frontier tuning.

Coverage frontier-low result:

- Phase48 failed and was stopped early. Low-strength frontier bias did not preserve baseline behavior.
- The next coverage path should be a high-success h200 teacher/trajectory generator, not additional frontier-bias tuning.

Coverage teacher prototype result:

- Local greedy and waypoint-lookahead h200 teacher prototypes failed.
- If continuing coverage, build an explicit route-planning/sweep teacher or add recurrent/persistent target memory with proper state handling; do not rely on local greedy scoring.

Frontier continuation verification:

- Regression tests passed: `28 passed`.
- No active training/evaluation process is running.
- Coverage is still unresolved; current implemented frontier bias is compatible but empirically ineffective.

Documentation update:

- Current explanation of pure PPO unreliability and BC/DAgger necessity is in `docs/specialist_ppo_reliability_zh.md`.
- Future agents should read that doc before proposing another pure-PPO or scalar reward-only coverage run.

Coverage curriculum v1 next:

- Run `configs/policy/debug_ppo_coverage_factorized_group_train_curriculum_v1.yaml` under `configs/env/debug_coverage_train_curriculum_v1.yaml` from original factorized-group coverage BC checkpoint.
- First criterion: modified training-env reward and success should rise clearly.
- Second criterion: if training-env succeeds, evaluate transfer under canonical `configs/env/coverage.yaml` separately.

Coverage phase49 result:

- Modified training env converged: 100-episode success `0.73`.
- Canonical transfer failed: 100-episode success `0.14`.
- Next action: bridge curriculum closer to canonical, starting from phase49 best checkpoint.

Coverage phase50 result:

- Bridge v2 100-episode success is `0.56`; phase49 v1 remains the better converged modified training env at `0.73`.
- Canonical transfer remains poor (`0.13`).
- If the user wants convergence in modified coverage training env, phase49 is currently the best. If canonical transfer is required, next route must change coverage task design more substantially or add route-planning supervision.

Coverage curriculum verification:

- Regression tests passed: `29 passed`.
- No active training/evaluation process is running.

Coverage phase51 next:

- Run `configs/policy/debug_ppo_coverage_factorized_group_seq_frontier_antirepeat.yaml` under `configs/env/debug_coverage_train_curriculum_v3_antirepeat.yaml`.
- Recommended init checkpoint: phase50 bridge best:
  - `outputs/training/bc_ppo/20260531_phase50_coverage_train_curriculum_v2_bridge/phase50_coverage_train_curriculum_v2_bridge/checkpoints/checkpoint_best_eval.pt`
- This is the first real frontier-enabled factorized-group run after fixing policy factory propagation for `coverage_frontier_*`.
- Track success rate, `coverage_ratio`, `collision_rate`, `path_length`, `repeated_coverage_ratio`, and new `demand_revisit_excess`.
- If phase51 reward is very negative but success/coverage rise and revisit metrics drop, that is expected under the stronger anti-repeat reward. Do not reject the run by return alone.
- If success collapses below phase50, reduce `repeated_demand_coverage` first (e.g. `-3.0`) before changing architecture; keep sequential group context on for this branch.

Coverage phase52 next:

- Phase51 collapsed and was stopped. Do not resume it unless specifically diagnosing frontier behavior.
- Run phase52 from phase50 bridge best:
  - Env: `configs/env/debug_coverage_train_curriculum_v3_moderate_antirepeat.yaml`
  - Policy: `configs/policy/debug_ppo_coverage_factorized_group_seq_moderate_antirepeat.yaml`
- First eval criterion at update 20:
  - should be closer to phase50 bridge (`success≈0.50`) than phase51 (`success=0.133`);
  - repeated metrics may remain high initially, but `coverage_ratio` should not drop below `0.72`.
- If phase52 still collapses, the issue is likely architecture perturbation from newly inserted sequential parameters; next fallback is same moderate anti-repeat env with `use_sequential_group_context=false`, then only later re-enable sequential after BC/refit.

Coverage phase53 next:

- Phase52 confirmed that weak sequential context still damages the phase50 checkpoint.
- Run phase53:
  - Env: `configs/env/debug_coverage_train_curriculum_v3_moderate_antirepeat.yaml`
  - Policy: `configs/policy/debug_ppo_coverage_factorized_group_moderate_antirepeat.yaml`
  - Init: phase50 bridge best.
- Criterion:
  - update 20 should stay near control eval `success=0.50`, `coverage≈0.747`;
  - if it stays stable, continue to update 160 and compare revisit metrics against the control eval (`repeated≈0.9913`, `demand_revisit_excess≈28.14`);
  - if success drops, anti-repeat PPO is itself destabilizing and needs either lower LR/reference_coef stronger or BC/DAgger on successful anti-repeat trajectories.

Coverage phase54 next:

- Phase53 dropped below the phase50 control and was stopped.
- Run phase54:
  - Env: `configs/env/debug_coverage_train_curriculum_v3_moderate_antirepeat.yaml`
  - Policy: `configs/policy/debug_ppo_coverage_factorized_group_moderate_antirepeat_safe.yaml`
  - Init: phase50 bridge best.
- At update 20, compare against control eval:
  - target `success_rate >= 0.45`;
  - `coverage_ratio >= 0.74`;
  - revisit metrics should not increase above phase50 control (`repeated≈0.9913`, `demand_revisit_excess≈28.14`).
- If phase54 cannot preserve success, stop reward-PPO continuation and switch to data-side repair: collect successful low-revisit trajectories under phase50/v1 curricula, then BC/refit before PPO.

Coverage phase54 result:

- Phase54 is the current best modified-env continuation:
  - v3 moderate 100-episode success `0.56`;
  - phase50 control on same env `0.50`;
  - return improved from `-64.43` to `-44.03`;
  - `demand_revisit_excess` improved from `28.14` to `27.36`.
- Canonical transfer did not improve:
  - phase54 canonical 100-episode success `0.12`;
  - phase50 canonical was `0.13`.

Coverage next after phase54:

- Do not claim canonical coverage solved.
- Best next route is a closer-to-canonical safe continuation from phase54 best:
  - keep original factorized-group architecture;
  - keep conservative PPO settings from phase54;
  - use `demand_quantile` closer to canonical/default (`0.57` or `0.55`) and `success_ratio=0.81-0.82`;
  - keep anti-repeat terms moderate, not huge.
- Alternative route if bridge still fails: collect successful low-revisit trajectories from phase54/v3 moderate and refit with BC before any sequential group context experiment.

Coverage phase55 result:

- Phase55 canonical bridge completed with best/final 30-episode success `0.30`.
- It is below phase54 v3-moderate (`0.56`) and does not repair canonical coverage.
- Do not keep extending phase55 with PPO; route/horizon generalization is the blocker.

Coverage next after phase55:

- Data-side repair is now the recommended path:
  - collect trajectories from phase54 best and phase55 best on v3-moderate and v4-bridge envs;
  - filter for success and lower `demand_revisit_excess`;
  - train/refit factorized-group BC on these trajectories;
  - then use phase54-safe PPO settings for short continuation.
- If implementing sequential/group decision again, first make its added path zero-impact at initialization or train it with BC; direct PPO warm-start damaged behavior in phase51/52.

Coverage phase56 result:

- Current-policy low-revisit success-only BC/refit did not improve v4 bridge:
  - BC/refit final success `0.30`, same as phase55.
- Do not repeat the same success-only BC filtering loop with current policy trajectories.

Coverage next after phase56:

- Build a stronger coverage teacher, preferably route-planning/sweep based.
- Requirements for a useful teacher before BC:
  - v4 bridge success clearly above `0.5`;
  - canonical success above current `0.12-0.20` baseline;
  - lower `demand_revisit_excess` than phase54/55 where possible.
- If teacher prototype cannot meet this, prioritize implementing persistent group/frontier target memory in the policy or environment observation rather than more scalar PPO tuning.

Coverage phase57 result:

- V5 sweep teacher meets the teacher-quality requirement:
  - canonical diagnostic success `0.66`;
  - v4 bridge diagnostic success `0.82`;
  - collision `0`;
  - much lower `demand_revisit_excess≈23.7`.
- BC from V5 data still only reaches canonical `success=0.30`, despite `coverage_ratio=0.759`.

Coverage next after phase57:

- Implement persistence in the decision mechanism:
  - persistent per-agent or per-group coverage target/route memory;
  - zero-impact default, compatible with existing `factorized_group`;
  - should use current observation's visited/demand channels to keep or advance targets.
- After implementing persistence:
  - first evaluate deterministic untrained/teacher-like target bias on canonical;
  - then BC on V5 data;
  - then short phase54-safe PPO.

Coverage phase58 result:

- Stateless lawnmower route head failed:
  - phase54 + route head canonical success `0.033`;
  - route-head BC epoch4 canonical success `0.133`;
  - both worse than V5 BC without route head (`0.30`).
- Do not continue this exact route-head branch.

Coverage next after phase58:

- If staying within feed-forward Gym policy, keep V5 teacher as an expert baseline/data source but do not expect BC to fully imitate persistence.
- For a real fix, add one of:
  - recurrent policy state;
  - environment observation channels for persistent assigned target/route progress;
  - wrapper-level stateful policy for coverage experts.
- Any such change must be marked architectural and tested for PPO logprob consistency, because mutable memory inside `forward()` is unsafe for PPO minibatch replay.

Coverage phase59 next:

- Use the newly implemented route-hint observation path, not the failed phase58 stateless lawnmower head.
- First verification:
  - run targeted tests for coverage route hints and factorized-group route-hint policy support.
- Diagnostic:
  - evaluate whether a phase54/V5-BC checkpoint plus `use_coverage_route_hint_head` can use `configs/env/debug_coverage_route_hint_canonical.yaml`.
  - Report success, coverage, collision, path length, repeated coverage, and `demand_revisit_excess`.
- Data path if diagnostic is viable:
  - regenerate V5 teacher data under `configs/env/debug_coverage_route_hint_canonical.yaml` so observations include the persistent route-hint channel;
  - BC with `configs/policy/debug_bc_coverage_factorized_group_route_hint_perm.yaml`;
  - then conservative PPO with `configs/policy/debug_ppo_coverage_factorized_group_route_hint_safe.yaml`.
- Validation rule:
  - route-hint env is a coverage task-design change, not original canonical `configs/env/coverage.yaml`.
  - Keep both results separate; do not claim original canonical coverage is solved unless original canonical eval improves independently.

Coverage phase59 update:

- Direct route-hint bias on phase54/V5-BC checkpoints failed (`success_rate=0.0`), including after sequential suppression.
- Do not keep tuning direct route-hint strength against old checkpoints.
- Next concrete step:
  - generate V5 teacher data under `configs/env/debug_coverage_route_hint_canonical.yaml`;
  - train `debug_bc_coverage_factorized_group_route_hint_perm`;
  - evaluate under the same route-hint env and original canonical separately.

Coverage phase59 BC retry:

- The route-hint V5 dataset is available and teacher quality is acceptable (`terminal_success≈0.642`).
- Retry BC after the route-hint head pooling fix:
  - dataset: `outputs/debug_long/20260601_phase59_coverage_route_hint/v5_route_hint_canonical_e120.npz`
  - config: `configs/policy/debug_bc_coverage_factorized_group_route_hint_perm.yaml`
  - env: `configs/env/debug_coverage_route_hint_canonical.yaml`
  - init checkpoint: V5-BC no-route checkpoint `outputs/training/bc/20260601_022614/debug_bc_coverage_factorized_group_perm_coverage_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0024.pt`

Coverage phase59 next after BC:

- Route-hint-head BC was only `success_rate=0.26`; do not PPO from it.
- Run route-hint-observation-only BC:
  - config: `configs/policy/debug_bc_coverage_factorized_group_route_hint_obs_perm.yaml`
  - same dataset/env/init checkpoint as above.
- If obs-only BC does not exceed `0.30`, route hints alone are not solving imitation; next better route is to encode per-agent route target coordinates in the agent observation or add a proper recurrent actor instead of a shared heatmap.

Coverage phase60 next:

- Route-hint obs-only BC was also `success_rate=0.26`; shared heatmap hints are not enough.
- Use the new per-agent route-target observation path:
  - generate V5 data with `configs/env/debug_coverage_route_target_agents_canonical.yaml`;
  - train `configs/policy/debug_bc_coverage_factorized_group_route_target_agents_perm.yaml`;
  - warm-start from V5-BC is allowed because `scripts/train_bc.py` now filters shape-mismatched tensors.
- If BC crosses `success_rate>0.40`, run conservative PPO with `configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_safe.yaml`.

Coverage phase60 update:

- Route-target-agent BC succeeded:
  - run: `outputs/training/bc/20260601_034310/debug_bc_coverage_factorized_group_route_target_agents_perm_coverage_N4_multi_channel_field_plus_task_id/`
  - final 50-episode success `0.66`, coverage `0.79784`, collision `0.00555`, demand revisit excess `23.93`.
- Next action:
  - run conservative PPO from checkpoint `checkpoint_0032.pt` using `configs/policy/debug_ppo_coverage_factorized_group_route_target_agents_safe.yaml` and `configs/env/debug_coverage_route_target_agents_canonical.yaml`.
  - Validate best PPO checkpoint with 100 episodes under the same route-target env.
  - Separately evaluate under original canonical only as an ablation; original canonical will not have the route-target agent features and is not expected to match this architecture.

Coverage phase60 final:

- Treat coverage as repaired under the new per-agent route-target observation/decision mode.
- Best checkpoint:
  - `outputs/training/bc_ppo/20260601_phase60_coverage_route_target_agents_ppo/phase60_coverage_route_target_agents_ppo/checkpoints/checkpoint_best_eval.pt`
- Independent 100-episode evidence:
  - `success_rate=0.72`
  - `coverage_ratio=0.801508`
  - `collision_rate=0.002905`
  - `path_length=0.881318`
  - `demand_revisit_excess=23.424259`
- Remaining coverage work:
  - optional repeat seed for robustness;
  - decide whether to migrate original canonical coverage to include route-target agent features by default, or keep it as a separate expert-training env.
- Next global task:
  - repeat/validate formation under factorized_group and update the repaired-specialist table, because formation was previously only provisionally repaired.

Formation phase61 final:

- Treat formation as repaired under `factorized_group`.
- Best checkpoint:
  - `outputs/training/bc_ppo/20260530_phase39_formation_factorized_group_ppo/phase39_formation_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt`
- Evidence:
  - seed 7, 100 episodes: `success_rate=0.77`, `collision_rate=0.002313`, `formation_error=0.072942`.
  - seed 23, 100 episodes: `success_rate=0.78`, `collision_rate=0.002313`, `formation_error=0.073723`.
- Important note:
  - The decisive fix was template-aware success evaluation for `line` / `arc`, not retraining.
  - `line` and `arc` should not be judged by the same full-circle angular-uniformity condition as `circle` / `diamond`.
- Four-task status after phase61:
  - `goal_nav`: repaired under factorized_group.
  - `coverage`: repaired under the new per-agent route-target observation mode.
  - `risk_nav`: provisionally repaired; best 100-episode evidence remains `success_rate≈0.65`, low collision, but a repeat seed is still recommended.
  - `formation`: repaired under factorized_group after template-aware success fix.
- Recommended next action:
  - run one repeat-seed validation for `risk_nav` to bring it to the same evidence standard as coverage/formation.

Risk_nav phase62 final:

- Repeat-seed validation is complete.
- Best checkpoint:
  - `outputs/training/bc_ppo/20260530_phase41_risknav_factorized_group_dagger_safe/phase41_risknav_factorized_group_dagger_safe/checkpoints/checkpoint_best_eval.pt`
- Evidence:
  - seed 7, 100 episodes: `success_rate=0.65`, `goal_coverage_ratio=0.85`, `collision_rate=0.020797`, `path_length=0.698778`, `cumulative_risk_exposure=33.393036`.
  - seed 23, 100 episodes: `success_rate=0.65`, `goal_coverage_ratio=0.841667`, `collision_rate=0.018796`, `path_length=0.712840`, `cumulative_risk_exposure=34.320727`.
- Treat `risk_nav` as repaired/reproducible under the phase41 `factorized_group` specialist path, with the caveat that safety/risk metrics are still materially worse than the cleaner tasks.

Four-task status after phase62:

- `goal_nav`: repaired under `factorized_group`; prior best independent evidence around `success_rate≈0.78-0.80`.
- `coverage`: repaired under the new per-agent route-target observation/decision mode; seed 7/23 100-episode evidence `success_rate=0.72/0.68`, `coverage_ratio≈0.80/0.796`, `collision_rate≈0.0029/0.0048`.
- `risk_nav`: repaired/reproducible under `factorized_group`; two 100-episode seeds both `success_rate=0.65`, low-but-not-zero collision and non-trivial risk exposure.
- `formation`: repaired under `factorized_group` after template-aware success fix; two 100-episode seeds `success_rate=0.77-0.78`.

Coverage phase63 final:

- Repeat-seed validation is complete.
- Best checkpoint:
  - `outputs/training/bc_ppo/20260601_phase60_coverage_route_target_agents_ppo/phase60_coverage_route_target_agents_ppo/checkpoints/checkpoint_best_eval.pt`
- Evidence:
  - seed 7, 100 episodes: `success_rate=0.72`, `coverage_ratio=0.801508`, `collision_rate=0.002905`, `path_length=0.881318`, `demand_revisit_excess=23.424259`.
  - seed 23, 100 episodes: `success_rate=0.68`, `coverage_ratio=0.795507`, `collision_rate=0.004842`, `path_length=0.879962`, `demand_revisit_excess=23.553093`.
- Treat coverage as repaired/reproducible under the route-target-agent specialist path.

Goal_nav phase64 audit:

- Repeat-seed validation exposed weaker robustness than earlier seed 7 evidence.
- Best checkpoint remains:
  - `outputs/training/bc_ppo/20260530_phase36_goalnav_factorized_group_ppo/phase36_goalnav_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt`
- Evidence:
  - seed 7 training/final eval: `success_rate=0.80`, `goal_coverage_ratio=0.8725`, `collision_rate=0.063120`, `path_length=0.486920`.
  - seed 7 independent 50 episodes: `success_rate=0.78`, `goal_coverage_ratio=0.877`, `collision_rate=0.056029`, `path_length=0.510952`.
  - seed 23 independent 100 episodes: `success_rate=0.66`, `goal_coverage_ratio=0.804167`, `collision_rate=0.072423`, `path_length=0.553423`.
- Treat goal_nav as usable/repaired for basic expert training, but not as robust as coverage/formation under repeat-seed validation.
- If more single-task hardening is required, prioritize goal_nav safety/generalization continuation from phase36 before all4 training.

Goal_nav phase65 safety continuation:

- Completed a short conservative continuation from phase36 using `configs/policy/debug_ppo_goal_nav_factorized_group_safety_ref.yaml` and `configs/env/debug_goal_nav_seed23_safety.yaml`.
- Best run:
  - `outputs/training/bc_ppo/20260601_phase65_goalnav_safety_ref/phase65_goalnav_safety_ref/checkpoints/checkpoint_best_eval.pt`
- Independent original-env validation:
  - seed 23, 100 episodes: `success_rate=0.71`, `goal_coverage_ratio=0.824667`, `collision_rate=0.064165`, `path_length=0.542029`.
  - seed 7, 100 episodes: `success_rate=0.72`, `goal_coverage_ratio=0.824500`, `collision_rate=0.065032`, `path_length=0.536958`.
- Interpretation:
  - phase65 improves the weak seed23 result over phase36 (`0.66 -> 0.71`) and reduces seed23 collision;
  - phase65 lowers seed7 compared with phase36 (`0.78-0.80 -> 0.72`);
  - keep phase36 as the peak seed7/high-success checkpoint, and keep phase65 as the balanced robustness alternative.

Recommended next action:

- The four single-task specialists are now repaired/usable enough for downstream multi-task work under the new decision mode, but goal_nav seed robustness and risk_nav safety remain the main caveats.
- If improving beyond current repaired status, prioritize `goal_nav` seed-robust safety, `risk_nav` safety/risk exposure reduction, and coverage repeat-coverage reduction; do not revisit old non-factorized or shared-waypoint architecture.
- Before launching all4 multi-task training, create a concise checkpoint/config table for the four specialists and freeze these as expert baselines.

Goal_nav robustness diagnosis after phase66:

- Root cause is not a single bad eval seed.
- Multi-seed diagnostic over seeds `7/11/17/23/29`, 40 episodes each:
  - phase36 peak: `success_rate=0.755`, `goal_coverage_ratio=0.8635`, `pair_collision_rate=0.0606`.
  - phase65 balanced: `success_rate=0.770`, `goal_coverage_ratio=0.8763`, `pair_collision_rate=0.0560`.
  - greedy goal baseline: `success_rate=0.775`, `goal_coverage_ratio=0.9371`, `pair_collision_rate=0.0281`.
- Failure episodes differ sharply from success episodes:
  - phase36 failures: `goal_coverage≈0.443`, `pair_collision≈0.194`, `obstacle_collision≈0.0895`, `mean_action_alignment≈0.281`, `steps=200`.
  - phase65 failures: `goal_coverage≈0.462`, `pair_collision≈0.167`, `obstacle_collision≈0.0738`, `mean_action_alignment≈0.263`, `steps=200`.
- Interpretation:
  - failures are closed-loop collision/recovery failures, not simply missing goal perception;
  - after bad local multi-agent configurations, the policy stops making well-aligned progress to remaining goals.
- Next concrete repair if goal_nav hardening continues:
  - collect failed/near-failed phase36/phase65 rollouts;
  - teacher-relabel with distinct remaining-goal assignment and safer detours;
  - BC/DAgger refit plus conservative PPO;
  - optionally add explicit goal assignment/anti-crowding context or small repulsion for goal_nav.

Goal_nav task-target repair after phase67/68:

- Implemented the recommended explicit goal assignment/anti-crowding context as a general optional env observation feature:
  - `include_task_targets_in_agents`
  - works for goal_nav/risk_nav/formation/coverage;
  - default-off, old checkpoints/configs compatible.
- Best current goal_nav robust PPO checkpoint:
  - `outputs/training/bc_ppo/20260601_phase68_goalnav_task_targets_ppo/phase68_goalnav_task_targets_ppo/checkpoints/checkpoint_0040.pt`
- Supporting BC checkpoint:
  - `outputs/training/bc/20260601_125256/debug_bc_goal_nav_factorized_group_task_targets_goal_nav_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0024.pt`
- Evidence:
  - phase67 BC five-seed 40ep: `success_rate=0.95`, `goal_coverage_ratio=0.978333`, `pair_collision_rate=0.007909`.
  - phase67 BC seed23 100ep: `success_rate=0.89`, `goal_coverage_ratio=0.9415`, `pair_collision_rate=0.017221`.
  - phase68 PPO checkpoint_0040 five-seed 40ep: `success_rate=0.965`, `goal_coverage_ratio=0.984333`, `pair_collision_rate=0.005324`.
  - phase68 PPO checkpoint_0040 seed23 100ep: `success_rate=0.89`, `goal_coverage_ratio=0.9395`, `pair_collision_rate=0.016639`.
- Interpretation:
  - This resolves the previous goal_nav robustness caveat enough to replace phase36 as the recommended robust goal_nav expert.
  - Do not use phase68 `checkpoint_best_eval.pt` blindly if optimizing seed23 robustness; `checkpoint_0040.pt` was better than update60 on the independent seed23 100ep sweep.

Updated recommended next action:

- Treat all four single-task specialists as repaired/usable under the new target-aware factorized-group direction.
- For downstream multi-task work, prefer a shared convention of per-agent target hints where available instead of task-specific one-off policy heads.
- Next hardening target is `risk_nav`: try the same `include_task_targets_in_agents` path for risk_nav before inventing a different architecture.

Phase69 active long-run status:

- A four-task best-specialist parallel long run is currently active.
- Run timestamp:
  - `20260601_long_best_specialists_143934`
- tmux session:
  - `wayffusion_best_20260601_long_best_specialists_143934`
- Monitor:
  - `tail -f outputs/training/parallel_best_specialists/20260601_long_best_specialists_143934/parallel.log`
  - `tmux attach -t wayffusion_best_20260601_long_best_specialists_143934`
- Summary after completion:
  - `outputs/training/parallel_best_specialists/20260601_long_best_specialists_143934/summary.csv`
- Per-task logs:
  - `outputs/training/parallel_best_specialists/20260601_long_best_specialists_143934/goal_nav_task_targets_long.log`
  - `outputs/training/parallel_best_specialists/20260601_long_best_specialists_143934/coverage_route_targets_long.log`
  - `outputs/training/parallel_best_specialists/20260601_long_best_specialists_143934/risk_nav_dagger_safe_long.log`
  - `outputs/training/parallel_best_specialists/20260601_long_best_specialists_143934/formation_factorized_group_long.log`
- Expected PPO output dirs:
  - `outputs/training/bc_ppo/20260601_long_best_specialists_143934/goal_nav_task_targets_long/`
  - `outputs/training/bc_ppo/20260601_long_best_specialists_143934/coverage_route_targets_long/`
  - `outputs/training/bc_ppo/20260601_long_best_specialists_143934/risk_nav_dagger_safe_long/`
  - `outputs/training/bc_ppo/20260601_long_best_specialists_143934/formation_factorized_group_long/`
- Completion behavior:
  - `scripts/run_ppo_parallel_best_specialists_all4.sh` will email the configured `EMAIL_TO` after all four jobs finish.
  - SMTP was verified by a notification test and by the 1-update smoke run. Do not expose SMTP credentials.

Next action after Phase69 finishes:

- Read `summary.csv` first and check every row has `status=success`.
- For each task, inspect `best_eval_summary.json`, `training_metrics.csv`, and final eval CSVs under the corresponding PPO output dir.
- Report success rate together with task metrics, collision rate, and path length.
- Compare against the pre-Phase69 baselines:
  - `goal_nav`: phase68 checkpoint_0040, expected robust multi-seed success around `0.89-0.965` depending on eval seed mix.
  - `coverage`: phase60 best, expected seed7/23 100ep success `0.72/0.68`, coverage ratio around `0.80`.
  - `risk_nav`: phase41 best, expected seed7/23 100ep success `0.65`, with risk exposure still the main weakness.
  - `formation`: phase39 best with template-aware success, expected seed7/23 100ep success `0.77/0.78`.
- If a long-run task regresses, keep the pre-Phase69 checkpoint as the expert baseline and treat the regression as PPO drift, not as evidence that the specialist branch is invalid.

User-facing launch shortcut:

- A one-command tmux launcher is now available:
  - `bash scripts/tmux_start_best_specialists_all4.sh`
- It starts `scripts/run_ppo_parallel_best_specialists_all4.sh` inside a detached tmux session and writes:
  - `launch_inside_tmux.sh`
  - `launch_summary.txt`
  - `launcher.out`
  - `parallel.log`
  under `outputs/training/parallel_best_specialists/<RUN_TIMESTAMP>/`.
- Default long-run budgets in the wrapper:
  - `goal_nav`: `1000` updates
  - `coverage`: `1200` updates
  - `risk_nav`: `1500` updates
  - `formation`: `1000` updates
  - eval episodes per eval: `100`
- Parameter location:
  - edit the top parameter block of `scripts/tmux_start_best_specialists_all4.sh`, or override variables in the shell.
- Example override:
  - `GOAL_TOTAL_UPDATES=2000 COVERAGE_TOTAL_UPDATES=2500 RISK_TOTAL_UPDATES=3000 FORMATION_TOTAL_UPDATES=2000 EVAL_EPISODES=100 bash scripts/tmux_start_best_specialists_all4.sh`

TensorBoard cleanup note after 2026-06-03:

- New training runs default to `tensorboard_metric_mode: core`.
- TensorBoard should now be used as a concise monitoring dashboard, while `training_metrics.csv` / `eval_metrics.csv` remain the full-fidelity data source.
- Future PPO TensorBoard tags are namespaced by task set:
  - `bc_ppo/goal_nav/train/...`
  - `bc_ppo/coverage/train/...`
  - `bc_ppo/risk_nav/train/...`
  - `bc_ppo/formation/train/...`
- If a debugging run needs the old all-scalar behavior, add this to the policy config:
  - `tensorboard_metric_mode: all`
- Existing old event files, including `20260601_145359_best_specialists_all4_long`, are not rewritten. To see the cleaner TensorBoard layout, launch a new run after the cleanup patch.

Active PPO-main run after 2026-06-03:

- A four-task PPO-main long run is currently active.
- This run starts from BC checkpoints, not from PPO best checkpoints.
- Run timestamp:
  - `20260603_ppo_main_all4_162355`
- tmux session:
  - `wayffusion_ppo_main_20260603_ppo_main_all4_162355`
- Monitor:
  - `tail -f outputs/training/parallel_ppo_main_specialists/20260603_ppo_main_all4_162355/parallel.log`
  - `tmux attach -t wayffusion_ppo_main_20260603_ppo_main_all4_162355`
- Per-task logs:
  - `outputs/training/parallel_ppo_main_specialists/20260603_ppo_main_all4_162355/goal_nav_ppo_main_from_bc.log`
  - `outputs/training/parallel_ppo_main_specialists/20260603_ppo_main_all4_162355/coverage_ppo_main_from_bc.log`
  - `outputs/training/parallel_ppo_main_specialists/20260603_ppo_main_all4_162355/risk_nav_ppo_main_from_bc.log`
  - `outputs/training/parallel_ppo_main_specialists/20260603_ppo_main_all4_162355/formation_ppo_main_from_bc.log`
- Summary after completion:
  - `outputs/training/parallel_ppo_main_specialists/20260603_ppo_main_all4_162355/summary.csv`
- Expected output dirs:
  - `outputs/training/bc_ppo/20260603_ppo_main_all4_162355/goal_nav_ppo_main_from_bc/`
  - `outputs/training/bc_ppo/20260603_ppo_main_all4_162355/coverage_ppo_main_from_bc/`
  - `outputs/training/bc_ppo/20260603_ppo_main_all4_162355/risk_nav_ppo_main_from_bc/`
  - `outputs/training/bc_ppo/20260603_ppo_main_all4_162355/formation_ppo_main_from_bc/`
- Completion behavior:
  - `scripts/run_ppo_main_parallel_specialists_all4.sh` sends email after all four jobs finish.

Next action after active PPO-main run finishes:

- First inspect `summary.csv`; all rows should be `status=success`.
- For each task, inspect:
  - `training_metrics.csv`
  - `eval_metrics.csv`
  - `best_eval_summary.json`
  - `checkpoints/checkpoint_best_eval.pt`
- For coverage, compare against BC/Stage1 using:
  - `scripts/debug_long/compare_bc_safe_ppo_main.py`
- Do not judge PPO-main only by `eval_reward`.
- Required PPO-main diagnostics to review:
  - `eval_success_rate`
  - task coverage/goal/formation/risk quality metrics
  - `eval_collision_rate`
  - `approx_kl`
  - `kl_early_stop`
  - `reference_action_mse`
  - `policy_std_mean`
  - `clip_frac`
- If Stage2/PPO-main does not beat BC or Stage1, record that BC/teacher remains the main performance source and PPO-main did not yet provide a validated net improvement.

MAPPO baseline next action after 2026-06-04:

- A minimal MAPPO-style baseline now exists:
  - policy config: `configs/policy/mappo_shared.yaml`
  - trainer: `algorithms/mappo.py::MAPPOTrainer`
  - training entry: `scripts/train_mappo.py`
- Its purpose is diagnostic:
  - test whether PPO instability comes from treating the whole swarm as one high-dimensional joint action with one joint logprob/ratio;
  - compare against existing centralized PPO and `factorized_group` under the same task and same `N`.
- Suggested first controlled comparison:
  - PPO joint/deepsets:
    - `/opt/conda/bin/python scripts/train_ppo.py --config configs/policy/ppo_cnn_deepsets.yaml --tasks goal_nav --agent_counts 4 --total_updates 100 --eval_episodes 20 --headless`
  - PPO factorized_group:
    - use the best existing task-specific factorized_group config for the same task and `N=4`.
  - MAPPO shared:
    - `/opt/conda/bin/python scripts/train_mappo.py --config configs/policy/mappo_shared.yaml --tasks goal_nav --agent_counts 4 --total_updates 100 --eval_episodes 20 --headless`
- Key diagnostics to compare:
  - `eval_success_rate`, `eval_reward`, `eval_collision_rate`, `eval_path_length`;
  - `approx_kl`, `clip_frac`, `ratio_mean`, `policy_std_mean`, `grad_norm`;
  - seed robustness over at least 2-3 seeds before claiming the architecture improves training.
- Current MAPPO validation status:
  - only smoke-tested, not performance-tested;
  - output directory from smoke: `outputs/training/mappo/20260604_mappo_smoke/smoke_goal_cov/`.

Active pure MAPPO all4 long run after 2026-06-05:

- A four-task, no-BC, pure MAPPO long run is currently active.
- tmux session:
  - `wayffusion_mappo_20260605_mappo_pure_all4_061548`
- Monitor:
  - `tail -f outputs/training/parallel_mappo_specialists/20260605_mappo_pure_all4_061548/parallel.log`
  - `tmux attach -t wayffusion_mappo_20260605_mappo_pure_all4_061548`
- Per-task logs:
  - `outputs/training/parallel_mappo_specialists/20260605_mappo_pure_all4_061548/goal_nav_mappo_pure.log`
  - `outputs/training/parallel_mappo_specialists/20260605_mappo_pure_all4_061548/coverage_mappo_pure.log`
  - `outputs/training/parallel_mappo_specialists/20260605_mappo_pure_all4_061548/risk_nav_mappo_pure.log`
  - `outputs/training/parallel_mappo_specialists/20260605_mappo_pure_all4_061548/formation_mappo_pure.log`
- Per-task training output dirs:
  - `outputs/training/mappo/20260605_mappo_pure_all4_061548/goal_nav_mappo_pure/`
  - `outputs/training/mappo/20260605_mappo_pure_all4_061548/coverage_mappo_pure/`
  - `outputs/training/mappo/20260605_mappo_pure_all4_061548/risk_nav_mappo_pure/`
  - `outputs/training/mappo/20260605_mappo_pure_all4_061548/formation_mappo_pure/`
- Summary after completion:
  - `outputs/training/parallel_mappo_specialists/20260605_mappo_pure_all4_061548/summary.csv`
- This run uses:
  - `configs/policy/mappo_shared_long.yaml`
  - `1000` updates per task
  - `EVAL_EPISODES=50`
  - `bc_warm_start=0`; no checkpoint is loaded.
- First result checkpoints/eval rows should appear around update 50 because `eval_interval=50`.
- After completion, inspect for each task:
  - `training_metrics.csv`
  - `eval_metrics.csv`
  - `best_eval_summary.json`
  - `checkpoints/checkpoint_best_eval.pt`
- Judge MAPPO from scratch by:
  - `eval_success_rate`
  - task-specific metrics: goal coverage / coverage ratio and revisit excess / risk exposure / formation error
  - `eval_collision_rate`
  - `eval_path_length`
  - PPO stability diagnostics: `approx_kl`, `clip_frac`, `ratio_mean`, `policy_std_mean`, `grad_norm`
- If pure MAPPO remains low-success while BC warm-start/factorized specialists are high-success, record that architecture-level agent-wise ratios alone are not sufficient and the remaining gap likely comes from sparse team reward/exploration/credit assignment rather than only joint-logprob scaling.

Waypoint-level MARL note after 2026-06-05 refactor, superseded by 2026-06-06 final-waypoint action refactor:

- Historical mainline at that time was:
  - env: `envs/waypoint_marl_env.py::WaypointMultiUAVEnv`
  - policy: `policies/mappo_waypoint_policy.py::MAPPOWaypointPolicy`
  - trainer: `algorithms/mappo_waypoint.py::MAPPOWaypointTrainer`
  - script: `scripts/train_mappo_waypoint.py`
  - config: `configs/env/waypoint_missions.yaml` and `configs/policy/mappo_waypoint*.yaml`
- Current mainline after 2026-06-06:
  - env action API: final waypoint `[2]`, not candidate index;
  - candidate policy: `policies/candidate_selection_policy.py::CandidateSelectionWaypointPolicy`;
  - direct policy: `policies/direct_waypoint_policy.py::DirectWaypointPolicy`;
  - compatibility alias: `policies/mappo_waypoint_policy.py::MAPPOWaypointPolicy`;
  - policy output contract: `policies/policy_output.py::PolicyOutput`.
- Smoke status:
  - all six waypoint pytest commands passed;
  - `validate_waypoint_mappo.py` passed and generated GIFs;
  - 10-update debug training passed and wrote checkpoint/TensorBoard/CSV.
- Before launching serious long runs, inspect:
  - `outputs/debug/waypoint_mappo_validation/20260605_refactor_smoke_evalcsv/validation_summary.json`
  - `outputs/debug/waypoint_mappo_validation/20260605_refactor_smoke_evalcsv/mappo_train/eval_metrics.csv`
  - `outputs/training/mappo_waypoint/20260605_refactor_smoke_evalcsv/waypoint_debug_train_4tasks/training_metrics.csv`
  - `outputs/training/mappo_waypoint/20260605_refactor_smoke_evalcsv/waypoint_debug_train_4tasks/eval_metrics.csv`
  - `outputs/training/mappo_waypoint/20260605_refactor_smoke_evalcsv/waypoint_debug_train_4tasks/media/`
- Suggested next long-run command:
  - `/opt/conda/bin/python scripts/train_mappo_waypoint.py --config configs/policy/mappo_waypoint.yaml --env-config configs/env/waypoint_missions.yaml --tasks area_coverage belief_search priority_inspection connectivity_expansion --num_agents 4 --num_envs 8 --total_updates 1000 --eval_episodes 20 --record_eval_episodes 2 --record_format gif --headless`
- Metrics to judge:
  - `eval_success_rate`
  - `eval_reward`
  - `eval_action_validity_rate`
  - `eval_collision_rate`
  - task metrics:
    - `coverage_ratio`
    - `searched_probability_mass`
    - `weighted_poi_completion`
    - `connectivity_violation_rate`
    - `effective_explored_radius`
  - PPO diagnostics:
    - `approx_kl`
    - `clip_frac`
    - `ratio_mean`
    - `entropy`
    - `explained_variance`
- Likely next improvements if long-run learning is weak:
  - add task-specific candidate feature channels for expected coverage gain, belief mass, POI value/deadline, connectivity bridge value, target intercept probability;
  - add per-agent reward/advantage mixing option now that `per_agent_rewards [T,B,N]` is stored;
  - add curriculum configs for map size, no-fly zones, candidate count, max steps, and success thresholds without weakening final eval thresholds;
  - add heuristic behavior cloning for waypoint categorical actions only after pure MAPPO baseline is measured.

After 2026-06-06 cleanup:

- Treat waypoint MARL as the only active code path.
- Do not use old commands such as `scripts/train_ppo.py`, `scripts/train_bc.py`, `scripts/train_mappo.py`, or old four-task configs; they were removed from the working tree.
- Current valid commands:
  - validation:
    - `/opt/conda/bin/python scripts/check/validate_waypoint_mappo.py --tasks area_coverage belief_search connectivity_expansion --num_agents 3 --total_updates 5 --eval_episodes 2 --record_eval_episodes 1 --record_format gif --headless`
  - training:
    - `/opt/conda/bin/python scripts/train_mappo_waypoint.py --config configs/policy/mappo_waypoint.yaml --env-config configs/env/waypoint_missions.yaml --tasks area_coverage belief_search priority_inspection connectivity_expansion --num_agents 4 --num_envs 8 --total_updates 1000 --eval_episodes 20 --record_eval_episodes 2 --record_format gif --headless`
- If someone needs old centralized code or old specialist PPO/BC scripts, recover from git history rather than reintroducing them into the MARL branch.

Environment reference for future agents:

- Read `docs/waypoint_environment_guide_zh.md` before modifying env/reward/policy code.
- Treat these as the active contracts:
  - env API: `WaypointMultiUAVEnv.reset/step/get_global_state`
  - default action: per-agent final waypoint coordinate `[2]`
  - action space: `Box([0,0], [map_size,map_size], shape=(2,))`
  - candidate index mode: `action_mode=candidate_index_legacy` only for diagnostics
  - policy output: `PolicyOutput(env_action, train_action, logprob, entropy, value, aux)`
  - candidate-selection actor: internal categorical logits over `[B,N,K]`, but env receives final waypoint `[B,N,2]`
  - direct actor: Gaussian delta `[B,N,2]`, clipped to final waypoint `[B,N,2]`
  - critic input: global state from `get_global_state`
  - reward output: `team_reward`, `per_agent_rewards`, `components`, `metrics`, `success`
- If changing observation/action/policy-output fields, update:
  - `docs/waypoint_environment_guide_zh.md`
  - waypoint tests
  - `CandidateSelectionWaypointPolicy` and/or `DirectWaypointPolicy`
  - `MAPPOWaypointTrainer`
  - `agent_memory/cards/03_recent_modifications.md`

After 2026-06-06 final-waypoint action refactor:

- Current valid validation commands:
  - candidate-selection:
    - `/opt/conda/bin/python scripts/check/validate_waypoint_mappo.py --tasks area_coverage belief_search connectivity_expansion --policy-class candidate_selection_waypoint --num_agents 3 --total_updates 5 --eval_episodes 2 --record_eval_episodes 1 --record_format gif --headless`
  - direct waypoint:
    - `/opt/conda/bin/python scripts/check/validate_waypoint_mappo.py --tasks area_coverage belief_search --policy-class direct_waypoint --num_agents 3 --total_updates 2 --eval_episodes 1 --record_eval_episodes 1 --record_format gif --headless`
- Recent validated outputs:
  - `outputs/debug/waypoint_mappo_validation/20260606_waypoint_action_candidate_validate_v2/`
  - `outputs/debug/waypoint_mappo_validation/20260606_waypoint_action_direct_validate_v2/`
- Current pytest target:
  - `/opt/conda/bin/python -m pytest -q tests/test_waypoint_env_api.py tests/test_waypoint_scenarios.py tests/test_waypoint_rewards.py tests/test_candidate_selection_policy.py tests/test_direct_waypoint_policy.py tests/test_mappo_waypoint_trainer.py tests/test_waypoint_rendering.py`
- Next engineering work:
  - run a longer pure MAPPO comparison between `candidate_selection_waypoint` and `direct_waypoint`;
  - add `HybridPlannerWaypointPolicy` only if direct/candidate results show complementary weaknesses;
  - improve candidate-selection features with task-specific expected gain, but keep generation in policy/planner, not env;
  - consider per-agent or mixed advantage using stored `per_agent_rewards [T,B,N]`.

Superseded note for 2026-06-06 ActionAdapter + handwritten particle backend:

- The notes from that intermediate refactor are historical only.
- Do not run the old particle/kinematic backend commands.
- The active backend contract is the real MPE core replacement below.

After 2026-06-06 real MPE core backend replacement:

- Treat the ActionAdapter + self-written particle backend notes above as superseded.
- Only supported runnable dynamics backend:
  - `dynamics_backend.name=mpe_core`
  - `MPECoreBackend`
- Removed/unsupported backend names:
  - `mpe_particle`
  - `mpe_like_particle`
  - `kinematic_point`
- Removed/unsupported classes:
  - `MPEParticleBackend`
  - `KinematicPointBackend`
- Current actual local MPE source:
  - default configured source: `third_party_openai_mpe`
  - path: `third_party/openai_mpe/core.py`
  - reason: source selection is now explicit and does not auto-fallback based on installed packages.
  - manual alternative: set `dynamics_backend.source: mpe2` to use `/opt/conda/lib/python3.11/site-packages/mpe2/_mpe_utils/core.py`.
  - if the configured source cannot be imported, backend construction fails; edit the config manually.
  - note: installing `mpe2` upgraded `gymnasium` to `1.3.0`; pip reported this conflicts with `lerobot==0.1.0` requiring `gymnasium==0.29.1`.
  - latest smoke: `outputs/debug/waypoint_mappo_validation/20260606_explicit_local_core_smoke/` passed with `mpe_source=third_party_openai_mpe`.
- Current valid validation commands:
  - candidate-selection + real MPE core:
    - `/opt/conda/bin/python scripts/check/validate_waypoint_mappo.py --tasks area_coverage belief_search connectivity_expansion --policy-class candidate_selection_waypoint --dynamics-backend mpe_core --action-adapter waypoint_velocity_tracker --num_agents 3 --total_updates 5 --eval_episodes 2 --record_eval_episodes 1 --record_format gif --headless`
  - direct waypoint + real MPE core:
    - `/opt/conda/bin/python scripts/check/validate_waypoint_mappo.py --tasks area_coverage belief_search --policy-class direct_waypoint --dynamics-backend mpe_core --action-adapter waypoint_velocity_tracker --num_agents 3 --total_updates 2 --eval_episodes 1 --record_eval_episodes 1 --record_format gif --headless`
  - removed-backend negative test:
    - `/opt/conda/bin/python scripts/check/validate_waypoint_mappo.py --tasks area_coverage --policy-class direct_waypoint --dynamics-backend mpe_particle --num_agents 3 --total_updates 1 --eval_episodes 1 --headless`
    - expected: failure with `mpe_particle was removed because it was a self-written MPE-like backend. Use mpe_core.`
- Current pytest target:
  - `/opt/conda/bin/python -m pytest -q tests/test_mpe_core_backend.py tests/test_action_adapters.py tests/test_waypoint_env_api.py tests/test_candidate_selection_policy.py tests/test_direct_waypoint_policy.py tests/test_mappo_waypoint_trainer.py tests/test_waypoint_rewards.py tests/test_waypoint_rendering.py`
- Recent validated outputs:
  - `outputs/debug/waypoint_mappo_validation/20260606_mpe_core_candidate_validate/`
  - `outputs/debug/waypoint_mappo_validation/20260606_mpe_core_direct_validate/`
- Metrics that prove real MPE backend use:
  - `uses_real_mpe_core=true`
  - `mpe_source=third_party_openai_mpe` for the default config
  - `mpe_source=mpe2` only when `dynamics_backend.source: mpe2` is set manually
  - `mpe_world_step_calls > 0`
- Next engineering work:
  - Investigate why 80-setting `WaypointVelocityTracker` calibration failed all hard constraints:
    - `geofence_violation_rate` failed in 80/80 settings;
    - `no_fly_violation_rate` failed in 39/80 settings;
    - `arrival_rate` failed in 44/80 settings.
  - First diagnostic should separate controller quality from probe/heuristic waypoint feasibility:
    - inspect top-5 GIFs under `outputs/debug/adapter_calibration/local_core_velocity_tracker_v1/media/`;
    - check whether synthetic/task target paths command agents too close to map boundaries or through no-fly zones;
    - consider adding a safety-margin waypoint projector before adapter calibration, not by reintroducing handwritten dynamics.
  - Do not treat `configs/env/waypoint_missions_tuned.yaml` as active; full calibration failed and the smoke-generated tuned config was removed.
  - Continue MAPPO health/performance work from baseline local-core outputs:
    - `outputs/debug/mappo_health_check/candidate_4task_baseline_adapter_seed0/`;
    - `outputs/debug/mappo_health_check/candidate_connectivity_baseline_adapter_seed1/`;
    - `outputs/debug/mappo_health_check/direct_2task_baseline_adapter_seed0/`.
  - Candidate-selection MAPPO health is finite and can produce nonzero success; direct waypoint debug is finite but barely moves under current scaling, so inspect `DirectWaypointPolicy` action scale/log_std and adapter command magnitude before long direct-policy training.
  - SMTP env file is now supported:
    - use `scripts/check/send_smtp_report.py --env-file .secrets/wayffusion_mail.env ...`;
    - user env-file aliases `SMTP_*` / `EMAIL_TO` are mapped to `WAYFFUSION_SMTP_*`;
    - do not print or commit `.secrets/wayffusion_mail.env`.
  - Latest debug report:
    - `outputs/debug/final_debug_report.md`;
    - SMTP send status: `outputs/debug/email_sent.json`.

After 2026-06-06 WaypointVelocityTracker scale recalibration:

- Do not add `RealDroneWaypointTracker` unless the user explicitly reverses the latest instruction. The active adapter remains:
  - `action_adapter.name=waypoint_velocity_tracker`.
- Active default scale:
  - `map_size=1.0` means roughly `100m x 100m`;
  - `max_waypoint_distance=max_command_distance=0.20` (`20m`);
  - `max_speed=0.04` (`4m/s`);
  - `dynamics_backend.dt=0.1`, `substeps=20`, so one env decision is about `2s`;
  - `acceptance_radius=0.020` (`2m`);
  - `slowdown_radius=0.08` (`8m`);
  - `velocity_gain=4.0`, `max_accel=0.10`, `control_smoothing=0.45`.
- The following files must stay synchronized:
  - `configs/env/waypoint_missions.yaml`:
    - `max_waypoint_distance`;
    - `max_speed`;
    - `action_adapter.max_command_distance`;
    - `action_adapter.max_speed`;
    - `dynamics_backend.max_speed`;
  - `configs/policy/mappo_waypoint.yaml`:
    - `max_waypoint_distance`;
  - `configs/policy/mappo_waypoint_debug.yaml`:
    - `max_waypoint_distance`;
  - `configs/policy/direct_waypoint_debug.yaml`:
    - `max_delta`.
- New adapter profiles:
  - `configs/env/adapters/waypoint_velocity_tracker_conservative.yaml`;
  - `configs/env/adapters/waypoint_velocity_tracker_default.yaml`;
  - `configs/env/adapters/waypoint_velocity_tracker_aggressive.yaml`.
- Latest calibration outputs:
  - `outputs/debug/waypoint_adapter_scale_calibration_20260606/`;
  - `outputs/debug/waypoint_adapter_scale_calibration_20260606_refined/`.
- Key result:
  - `def_020_s04_a100_2s` was the selected default:
    - score `1.922`;
    - local probe arrival `0.50`;
    - task progress `0.191`;
    - heuristic task success mean `1.00`;
    - task no-fly/geofence corrections `0 / 0`.
- Updated docs:
  - `docs/waypoint_adapter_calibration_zh.md`.
- New diagnostics now available in env info, rollout CSV, and eval records:
  - `mean_distance_to_waypoint_m`;
  - `arrival_rate`;
  - `command_update_rate`;
  - `command_reject_rate`;
  - `mean_desired_speed_mps`.
- Recommended next validation:
  - rerun candidate-selection MAPPO health with the new default config;
  - rerun direct-waypoint smoke because `max_delta` changed from `0.22` to `0.20` and the adapter response changed;
  - inspect whether direct policy still barely moves or now produces useful waypoint deltas.

After first MAPPO specialist smoke:

- New tuning controller:
  - `scripts/debug/tune_mappo_specialists.py`
- Specialist configs:
  - `configs/policy/specialists/mappo_area_coverage.yaml`;
  - `configs/policy/specialists/mappo_belief_search.yaml`;
  - `configs/policy/specialists/mappo_priority_inspection.yaml`;
  - `configs/policy/specialists/mappo_connectivity_expansion.yaml`.
- First debug smoke output:
  - `outputs/debug/mappo_specialists/20260606_specialist_smoke01/`
- Current task status from 10-update smoke:
  - `area_coverage`: `CONVERGED` under short smoke criteria only;
  - `belief_search`: `CONVERGED` under short smoke criteria only;
  - `priority_inspection`: `NEED_MORE_TUNING`;
  - `connectivity_expansion`: `NEED_MORE_TUNING`.
- Do not launch formal long training for all four tasks yet.
- Recommended next tuning commands:
  - continue debug sweep for priority:
    - `/opt/conda/bin/python scripts/debug/tune_mappo_specialists.py --stage debug --tasks priority_inspection --debug-max-trials-per-task 6 --debug-total-updates 20 --debug-num-envs 4 --debug-rollout-steps 128 --debug-eval-interval 5 --debug-eval-episodes 4 --debug-record-eval-episodes 1 --record-format gif --headless`
  - continue debug sweep for connectivity:
    - `/opt/conda/bin/python scripts/debug/tune_mappo_specialists.py --stage debug --tasks connectivity_expansion --debug-max-trials-per-task 6 --debug-total-updates 20 --debug-num-envs 4 --debug-rollout-steps 128 --debug-eval-interval 5 --debug-eval-episodes 4 --debug-record-eval-episodes 1 --record-format gif --headless`
- Once priority/connectivity show positive, stable debug trends:
  - run formal training with `--stage training`, but keep output under `outputs/training/mappo_specialists/`.
- Need inspect:
  - why `priority_inspection` reward is strongly negative despite weighted completion around `0.65`;
  - whether POI repeat/deadline/distance penalties dominate;
  - why `connectivity_expansion` progress can reach `~0.46` but success remains `0`;
  - whether success requires both coverage and effective radius or if connectivity action filtering makes behavior too conservative.

After MAPPO specialist debug round 2:

- `priority_inspection` is now debug-converged:
  - output root: `outputs/debug/mappo_specialists/20260606_specialist_debug02/`;
  - best run: `priority_inspection__cand__N4__E4__rs128__lr2e-4__seed0__poi_vf`;
  - success `1.0`;
  - weighted POI completion `0.7779122805078398`;
  - reward `34.27439065393992`.
- `connectivity_expansion` is still `NEED_MORE_TUNING`, not DONE.
  - diagnostic outputs:
    - `outputs/debug/mappo_specialists/20260606_connectivity_debug03/`;
    - `outputs/debug/mappo_specialists/20260606_connectivity_aggressive04/`;
    - `outputs/debug/mappo_specialists/20260606_connectivity_policyfeat05/`;
    - `outputs/debug/mappo_specialists/20260606_connectivity_prior06/`;
    - `outputs/debug/mappo_specialists/20260606_connectivity_lowent07/`;
    - `outputs/debug/mappo_specialists/20260606_connectivity_chaincand08/`.
  - best current observation:
    - effective radius can pass `success_radius=0.45`;
    - connectivity violation is usually `0.0`;
    - coverage remains below `success_coverage=0.35`, with best observed around `0.277` in debug03 and around `0.23` in later feature/prior trials.
  - greedy/random baseline under aggressive connectivity scale also fails:
    - greedy coverage around `0.20`;
    - random coverage around `0.29`;
    - both success `0.0`.
- Do not launch formal 1000-update training for `connectivity_expansion` yet.
- Formal training can be considered for `area_coverage`, `belief_search`, and `priority_inspection`, but still needs longer seed/robustness validation before claiming final task completion.
- Next connectivity work should prioritize:
  - inspect task reward components and success geometry rather than lowering thresholds;
  - improve connectivity-specific candidate generation toward higher area coverage without clustering;
  - compare stochastic vs deterministic eval because deterministic categorical eval can collapse toward low-movement candidates when logits are weak;
  - inspect whether action validity/safety filtering rejects too many rollout actions (`action_validity_rate` in some connectivity training runs remains around `0.90-0.93`);
  - consider task-specific reward scaling that makes coverage gain dominate radius-only expansion once radius already exceeds `success_radius`.
- Current code status:
  - `connectivity_chain_candidates` exists as an optional policy experiment but defaults to `false` because the first chain/fan trial was too conservative.
  - `candidate_prior_coef` exists but connectivity specialist default is reset to `0.0` because prior `0.08` did not improve convergence.
  - candidate-level connectivity features remain available to the policy:
    - radial gain;
    - radius from base;
    - communication margin.

After switching specialist mainline to direct waypoint MAPPO:

- Default MAPPO configs now use `policy_class: direct_waypoint`:
  - `configs/policy/mappo_waypoint.yaml`;
  - `configs/policy/mappo_waypoint_debug.yaml`;
  - `configs/policy/specialists/mappo_area_coverage.yaml`;
  - `configs/policy/specialists/mappo_belief_search.yaml`;
  - `configs/policy/specialists/mappo_priority_inspection.yaml`;
  - `configs/policy/specialists/mappo_connectivity_expansion.yaml`.
- Direct MAPPO semantics:
  - actor outputs continuous Gaussian delta per UAV;
  - env receives final waypoint `[B, N, 2]`;
  - PPO logprob/ratio/entropy are per-agent continuous-action quantities;
  - no candidate index and no categorical candidate sampling in the default specialist path.
- Candidate-selection results above should now be treated as historical baselines, not current specialist status.
- Before any formal training, rerun direct-specialist debug sweeps for all four tasks:
  - `/opt/conda/bin/python scripts/debug/tune_mappo_specialists.py --stage debug --tasks area_coverage belief_search priority_inspection connectivity_expansion --debug-max-trials-per-task 3 --debug-total-updates 20 --debug-num-envs 4 --debug-rollout-steps 128 --debug-eval-interval 5 --debug-eval-episodes 4 --debug-record-eval-episodes 1 --record-format gif --timestamp <timestamp> --headless`
- First direct smoke results:
  - `outputs/debug/mappo_specialists/20260606_direct_specialist_smoke/`;
  - `outputs/debug/mappo_specialists/20260606_direct_default_train_smoke/`;
  - both validate runtime only, not convergence.
- Direct-policy tuning priorities:
  - monitor `mean_distance_to_waypoint_m`, `mean_desired_speed_mps`, `arrival_rate`, `command_reject_rate`;
  - if direct policy barely moves, increase exploration via `log_std_init` or task-specific LR before changing adapter;
  - if action validity drops, reduce `max_delta`/`log_std_max` before changing env safety;
  - compare stochastic training metrics against deterministic eval because Gaussian mean can collapse to near-zero deltas early.

After Direct MAPPO specialist tuner and debug sweeps:

- Use the Direct-specific tuner, not the generic/candidate-era tuner:
  - `scripts/debug/tune_direct_mappo_specialists.py`.
- Use Direct-specific configs:
  - `configs/policy/direct_specialists/direct_area_coverage.yaml`;
  - `configs/policy/direct_specialists/direct_belief_search.yaml`;
  - `configs/policy/direct_specialists/direct_priority_inspection.yaml`;
  - `configs/policy/direct_specialists/direct_connectivity_expansion.yaml`.
- Direct outputs:
  - debug: `outputs/debug/mappo_direct_specialists/`;
  - formal: `outputs/training/mappo_direct_specialists/`.
- Current Direct task status:
  - `belief_search`: short debug `CONVERGED`, but still needs longer seed/robustness validation.
  - `area_coverage`: `NEED_MORE_TUNING`; best coverage around `0.474`, success `0.25`, target success threshold not reached.
  - `priority_inspection`: `NEED_MORE_TUNING`; best weighted POI completion around `0.556`, success `0.25`.
  - `connectivity_expansion`: `NEED_MORE_TUNING`; easy communication curriculum can reach success, target stability not proven.
- Key Direct diagnostics:
  - action scale is mostly valid; eval action validity is generally `>=0.97`;
  - rollout validity can be marginal under high exploration (`~0.91-0.96`);
  - PPO updates are very weak: `approx_kl` often around `1e-7`, `clip_frac=0.0`;
  - this suggests trying stronger PPO updates or longer horizons before reward surgery for area/priority.
- Recommended next Direct commands:
  - longer area/priority continuation from the best scale:
    - `/opt/conda/bin/python scripts/debug/tune_direct_mappo_specialists.py --stage debug --tasks area_coverage priority_inspection --trial-tags md025_explore lr3e-4 --debug-max-trials-per-task 2 --debug-total-updates 50 --debug-num-envs 4 --debug-rollout-steps 128 --debug-eval-interval 10 --debug-eval-episodes 6 --debug-record-eval-episodes 1 --record-format gif --timestamp <timestamp> --headless`
  - target-config connectivity validation:
    - `/opt/conda/bin/python scripts/debug/tune_direct_mappo_specialists.py --stage debug --tasks connectivity_expansion --trial-tags conn_md015 conn_md020_ent --debug-max-trials-per-task 2 --debug-total-updates 50 --debug-num-envs 4 --debug-rollout-steps 128 --debug-eval-interval 10 --debug-eval-episodes 6 --debug-record-eval-episodes 1 --record-format gif --timestamp <timestamp> --headless`
- Do not claim Direct MAPPO specialists are complete yet.
- Do not launch formal `outputs/training/mappo_direct_specialists/` for `area_coverage`, `priority_inspection`, or `connectivity_expansion` yet.
- For `belief_search`, a formal long run is allowed only as robustness validation, not as proof that all Direct specialists are solved.

After GAE/action-scale repair and full-flow launch:

- Full Direct MAPPO flow is running in tmux:
  - session: `wayffusion_direct_full_20260606`;
  - timestamp: `20260606_direct_full_run02`;
  - stdout log: `outputs/debug/mappo_direct_specialists/full_tuning/20260606_direct_full_run02/tmux_stdout.log`.
- Check status with:
  - `tmux list-sessions | rg wayffusion_direct_full_20260606 || true`;
  - `tail -n 120 outputs/debug/mappo_direct_specialists/full_tuning/20260606_direct_full_run02/tmux_stdout.log`.
- When the run finishes, inspect:
  - `outputs/debug/mappo_direct_specialists/full_tuning/20260606_direct_full_run02/summary.json`;
  - `outputs/debug/mappo_direct_specialists/full_tuning/20260606_direct_full_run02/report.md`.
- Action-scale smoke outputs are under:
  - `outputs/debug/mappo_direct_specialists/action_scale_smoke/20260606_direct_full_run02/`.
- Formal long-training outputs, if any task passes smoke, are under:
  - `outputs/training/mappo_direct_specialists/formal/20260606_direct_full_run02/`.
- Note:
  - `20260606_direct_full_run01` is a partial historical run that was intentionally stopped after discovering that formal training was launched before all task smoke checks.
  - The script has been corrected so run02 performs all smoke checks first, then formal training.
- First smoke observed:
  - task/trial: `area_coverage` / `md020_std18`;
  - initial metrics around update 14:
    - `action_validity_rate=0.9736`;
    - `delta_clip_fraction=0.4502`;
    - `approx_kl=8.4e-05`;
    - `clip_frac=0.0`.
- This is not a convergence result yet; it only confirms action diagnostics are being logged and clipping is not saturated.
- If the full flow finishes with `NEED_MORE_TUNING`:
  - For `delta_clip_fraction > 0.5`, lower `log_std_init/log_std_max` or `max_delta`.
  - If `approx_kl` returns to `~1e-7` while task metrics are flat, inspect `advantage_std`, `reward_norm`, and reward scale before raising LR.
  - If `area_coverage` stalls near `0.45-0.50`, next round should modify coverage reward weights, not adapter.
  - If `priority_inspection` stalls near `0.50-0.60`, next round should use POI curriculum and repeat/first-visit reward shaping.
  - If `connectivity_expansion` only succeeds on easy comm, run curriculum `comm_radius 0.42 -> 0.39 -> 0.36`; do not count easy as target convergence.
- TensorBoard for current run:
  - CSV metrics have been exported into `tensorboard_csv/` subdirectories.
  - A background exporter is running:
    - `tmux attach -t wayffusion_direct_tb_export_20260606`;
    - log: `outputs/debug/mappo_direct_specialists/full_tuning/20260606_direct_full_run02/csv_to_tensorboard.log`.
  - TensorBoard is running on port `6007`:
    - `tmux attach -t wayffusion_tensorboard_6007`.
  - Correct manual start command:
    - `/opt/conda/bin/tensorboard --logdir_spec smoke:outputs/debug/mappo_direct_specialists/action_scale_smoke/20260606_direct_full_run02,formal:outputs/training/mappo_direct_specialists/formal/20260606_direct_full_run02 --bind_all --port 6007`.
  - Do not use a bare comma-separated `--logdir`; this TensorBoard version treats it as one invalid path.

Future output-layout rule:

- Do not create new debug outputs under nested roots like:
  - `outputs/debug/mappo_direct_specialists/action_scale_smoke/<timestamp>/...`.
- Use phase roots directly under `outputs/debug`:
  - `outputs/debug/<YYYYMMDD_HHMM_phase_change_on_xxx>/<task>/<short_run>/`.
- Future formal/training roots should mirror this:
  - `outputs/training/<YYYYMMDD_HHMM_phase_change_on_xxx>/<task>/<short_run>/`.
- Use minute-level timestamps, not second-level timestamps, unless explicitly needed.
- Keep run names short:
  - `smoke__md0p20__s0__tag`;
  - `formal__u1500__md0p20__s0__tag`.
- Reports and summaries should live at the phase root:
  - `report.md`;
  - `summary.json`.
- CSV can remain complete, but TensorBoard should default to core metrics only:
  - use `tensorboard_metric_mode: core`;
  - use `--metric-mode all` only when explicitly debugging raw metrics.

Direct MAPPO v2 monitoring and next actions:

- Active old run remains untouched:
  - `tmux attach -t wayffusion_direct_full_20260606`;
  - stdout: `outputs/debug/mappo_direct_specialists/full_tuning/20260606_direct_full_run02/tmux_stdout.log`.
- Active new v2 run:
  - `tmux attach -t wayffusion_direct_v2_20260606_1611`;
  - stdout: `outputs/debug/20260606_1611_phase_change_on_direct_obs_reward_v2/tmux_stdout.log`.
- New v2 TensorBoard:
  - URL: `http://<server>:6008`;
  - tmux: `wayffusion_tensorboard_v2_6008`;
  - log: `outputs/debug/20260606_1611_phase_change_on_direct_obs_reward_v2/tensorboard_6008.log`.
- New v2 output roots:
  - debug/smoke/report: `outputs/debug/20260606_1611_phase_change_on_direct_obs_reward_v2`;
  - formal training: `outputs/training/20260606_1611_phase_change_on_direct_obs_reward_v2`.
- V2 config files:
  - env: `configs/env/waypoint_missions_direct_tune_v2.yaml`;
  - policies: `configs/policy/direct_specialists_v2/*.yaml`.

What to watch in v2:

- For all tasks:
  - `eval_success_rate`;
  - task metric (`coverage_ratio`, `searched_probability_mass`, `weighted_poi_completion`, `effective_explored_radius`);
  - `action_validity_rate`;
  - `delta_clip_fraction`;
  - `approx_kl`;
  - `clip_frac`;
  - `value_loss`;
  - `advantage_std`;
  - `policy_std_mean` / `log_std_mean`.
- If `delta_clip_fraction > 0.50` again:
  - lower `log_std_init/log_std_max`, or lower `max_delta`;
  - do not raise learning rate first.
- If `delta_clip_fraction` is healthy but `approx_kl ~ 1e-7` and task metric is flat:
  - increase PPO update strength next round:
    - `learning_rate=3e-4`;
    - `epochs=6`;
    - or lower reward normalization constraints only after checking reward scale.
- If `area_coverage` remains around `0.43-0.48`:
  - reward shaping still insufficient;
  - try stronger anti-overlap/revisit penalty and local frontier/progress signal, but do not modify base MPE dynamics.
- If `belief_search` regresses after target-channel masking:
  - inspect whether target masking removed an accidental leakage shortcut;
  - compare searched probability mass, not only detection success.
- If `priority_inspection` remains near `0.5-0.6`:
  - increase approach shaping or use POI curriculum;
  - do not call it converged unless target weighted completion threshold is met.
- If `connectivity_expansion` only works with easy comm:
  - keep it marked `NEED_MORE_TUNING`;
  - use curriculum `comm_radius 0.42 -> 0.39 -> 0.36`, with final target eval at base target comm settings.

Important caveat:

- The first v2 area smoke has healthier action clipping (`delta_clip_fraction≈0.398`) and coverage progress (`coverage_ratio≈0.435`) but `success_rate=0.0`.
- This is a positive debug signal, not convergence.
- Do not report any Direct specialist as complete until the full v2 report shows target-task multi-seed formal success.

## Domain-randomized specialist follow-up

- Keep active pre-randomization runs as baselines; do not stop or relabel them.
- New runs using `configs/env/waypoint_missions.yaml` now randomize:
  - base position in the lower-left `0.06-0.16` ranges;
  - spawns around the episode base;
  - static plus `0-1` lightweight random no-fly zone;
  - belief peaks/target;
  - POI positions/weights/deadlines.
- The base config intentionally keeps sensor/comm/base communication radii and all motion/policy scales fixed.
- Next robustness phase should:
  1. use a new minute-level phase directory;
  2. train seeds `0,1,2`;
  3. evaluate on fixed held-out seeds;
  4. group results by `no_fly_zone_count`;
  5. report success/task metric and safety violations, not reward alone.
- If early learning collapses, use a no-fly curriculum from one zone to 1-3 zones, but retain full randomization for target evaluation.
- Post-change coverage is normalized over navigable cells, so state this before comparing against historical full-grid coverage.

## Mandatory output handling for next runs

- Allocate the next unused `phaseNNN` before any debug run.
- Put all runs for the same debugging requirement directly under that phase directory.
- Put formal runs directly under `outputs/training`; do not mirror the debug phase hierarchy.
- Keep `run_registry.yaml` current before, during, and after every run.
- Do not migrate or rename active/historical runs unless the user explicitly requests archival.

Direct MAPPO specialist next-run rule:

- choose a validated phase such as `phase07_domain_randomization`;
- use `utils/experiment_layout.py`;
- never restore parameter-packed run names;
- full formal single-config runs should be `<runtime>/<task>/sN`;
- verify summary/report parameter fields before launching long jobs.

## Fixed/random evaluation rule

- New waypoint MAPPO runs should normally leave `configs/eval/waypoint_eval.yaml` at `randomization_mode: both`.
- Use `--train-randomization random` for robustness training unless an ablation explicitly needs fixed training.
- Use `--eval-randomization fixed` or `random` only for focused diagnostics; formal comparisons should retain `both`.
- Interpret `eval_generalization_gap` as fixed success rate minus randomized success rate. A large positive gap indicates layout overfitting.
- Existing unprefixed `eval_success_rate` and `eval_reward` refer to randomized evaluation in `both` mode and remain the checkpoint-selection metrics.

## After phase10 protocol validation

- Phase10 status is `READY_FOR_TUNING`, meaning the protocol works, not that either specialist converged.
- Use the phase10 run as a diagnostic baseline:
  `outputs/debug/mappo_direct_specialists/phase10_mappo_specialist_protocol_check/20260607_1117/`.
- Belief search:
  - investigate high action clipping (`recent mean about 0.87`);
  - PPO updates remain extremely small (`approx_kl about 8e-09`, `clip_frac=0`);
  - next tuning should reduce raw action clipping before increasing PPO update strength.
- Area coverage:
  - action clipping is healthier (`recent mean about 0.49`);
  - coverage remains around `0.25` at final eval under the stricter threshold;
  - next tuning should focus on learning signal/reward and ineffective coverage motion, not the evaluation protocol.
- Continue to use random training plus `both` evaluation for specialist tuning.

## Formal specialist run status after speed audit

Runtime `20260607_1208` was explicitly stopped and deleted at the user's
request. There is no active formal Wayffusion training process.

The next formal run should use:

- `--env_backend process`;
- 16 environments per task if all four GPUs and approximately 64 CPU cores are
  dedicated to the suite;
- `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for each task parent;
- the existing randomized train/eval protocol;
- the existing formal layout and SMTP monitor.

The one-command launcher already defaults to this faster backend:

```bash
bash scripts/formal_mappo_specialists/launch_tmux.sh
```

Do not use `--env_backend thread`; measured throughput was substantially worse
than sync. Process vectorization measured about `595 env-steps/s` at 16 envs,
versus about `91 env-steps/s` for sync, while retaining the same MPE dynamics.

Before the next full formal launch, a practical resource choice is:

- four concurrent tasks × 16 process envs = 64 environment workers;
- if the host is shared or latency becomes unstable, use `NUM_ENVS=12` to leave
  CPU headroom.

No convergence claim should be derived from the deleted run because it ended
before its first evaluation.

## After phase11 action/obs scale probe

Phase11 showed that Direct MAPPO's first blocker was action/logprob mismatch
from heavy Gaussian delta clipping. Lower log-std plus task field masking and
spatial context reduced clipping substantially. `belief_search` improved to a
best debug success rate of `0.8333`, but the other tasks still need reward and
curriculum work.

Next phase should be:

- phase name: `phase12_reward_curriculum`;
- output root:
  `outputs/debug/mappo_direct_specialists/phase12_reward_curriculum/<runtime>/`;
- no output in `outputs/training`;
- no BC;
- no candidate-selection fallback;
- no MPE dynamics/backend changes.

Recommended phase12 plan:

- `area_coverage`: enable optional `w_uncovered_approach` and
  `w_repeat_footprint`, increase overlap/repeat penalties, reduce distance
  penalty, and monitor `coverage_ratio`, `overlap_ratio`,
  `repeat_footprint_ratio`, and `delta_clip_fraction`.
- `belief_search`: keep the phase11 action/obs improvements and validate a
  slightly longer run; do not over-tune unless clipping rises again.
- `priority_inspection`: enable/raise `w_approach`, raise first-visit reward,
  strengthen repeat penalty, and use a clearly labelled easy POI curriculum
  (`num_pois_range` reduced) only as a learning-signal diagnostic, not target
  success.
- `connectivity_expansion`: loosen the too-conservative phase11 action scale
  slightly, raise expansion/coverage rewards, add small
  `w_connected_radius_hold`, and keep target connectivity constraints visible
  in the report.

Phase12 report must use the structure:

1. phenomenon;
2. cause analysis;
3. improvement strategy;
4. concrete operation;
5. experiment;
6. observed result;
7. feedback/next action.

If phase12 still leaves tasks below threshold, do not mark them as converged.
Use `NEED_MORE_TUNING` and launch a follow-up phase with a narrower hypothesis.

## After phases 12-16 Direct MAPPO debug

Current best debug status:

- `area_coverage`: phase13 reached `eval_success_rate=1.0` and
  `coverage_ratio=0.8548` under randomized debug eval.
- `belief_search`: phase13 reached `eval_success_rate=0.8333` and
  `searched_probability_mass=0.5467`.
- `priority_inspection`: phase14 reached `eval_success_rate=1.0` and
  `weighted_poi_completion=0.8746` with the easy POI curriculum and Gaussian
  POI target field.
- `connectivity_expansion`: still `NEED_MORE_TUNING`.

Validated useful changes:

- DirectWaypointPolicy `use_field_moment_context` is useful and should be kept
  for future Direct specialists.
- Constant LR is better than linear LR for these short debug phases; linear LR
  caused late-stage `approx_kl` collapse.
- Priority needs Gaussian/continuous target field, not sparse one-cell POI
  markers only.
- Connectivity needs `connectivity_action_filter: true` for safety. Without it,
  reward-only training can learn high disconnected coverage.

Connectivity diagnosis:

- Phase13: radius could exceed target (`0.56`) but coverage was low (`0.27`).
- Phase15: coverage could approach target (`0.54`) but violation was high
  (`~0.77`) and radius was low at high coverage.
- Phase16: hard filter forced violation to `0.0` and radius stayed high
  (`0.49-0.63`), but coverage stayed low (`0.26-0.32`).
- Therefore connectivity is not solved by more scalar reward weight alone.

Recommended next connectivity phase:

- phase name: `phase17_connectivity_curriculum`;
- output root:
  `outputs/debug/mappo_direct_specialists/phase17_connectivity_curriculum/<runtime>/`;
- keep Direct MAPPO, no BC, no candidate policy;
- keep MPE dynamics and waypoint adapter unchanged;
- train with `connectivity_action_filter: true`;
- use a curriculum:
  1. easier comm curriculum, for example `comm_radius=0.42`,
     `base_comm_radius=0.52`;
  2. target eval still on current `comm_radius=0.36`,
     `base_comm_radius=0.45`;
  3. if easy train succeeds but target eval fails, anneal
     `0.42 -> 0.39 -> 0.36`;
- lower `max_delta` if action validity remains below `0.90`;
- consider explicit connected-coverage observation if curriculum fails:
  per-agent distance to base, nearest relay gap, and connected uncovered
  frontier summary.

Before any formal training:

- do not use phase14 priority as final target proof until it is validated on
  target POI count (`num_pois_range=[6,10]`) and multiple seeds;
- run at least seeds `0,1,2` for area/belief/priority debug winners;
- connectivity remains blocked at debug stage and should not enter formal
  training yet.

## Active phase17-20 connectivity soft-constraint run

User rejected phase16 hard-filter route for future multi-task policy learning.
Current active replacement route:

- `phase17_connectivity_margin_penalty`: dense disconnect-margin penalty.
- `phase18_connectivity_action_risk_penalty`: predicted action disconnect risk
  penalty only, no replacement.
- `phase19_connectivity_lagrangian_cost`: rollout-level Lagrangian cost
  multiplier for connectivity cost.
- `phase20_connectivity_auxiliary_prediction`: Direct policy auxiliary heads for
  next connectivity violation and coverage gain.

Hard requirements for these phases:

- `connectivity_action_filter` must stay `false`.
- `connectivity_candidate_filter` must stay `false` for this Direct debug path.
- Do not use fallback connectivity action replacement as the training method.
- Do not add task-specific rules to Direct actor action postprocess.
- Results must come from full long runs, not smoke/toy/short validation.

Active tmux session:

- `wf_conn_soft_20260608_1206`

Active command:

```bash
/opt/conda/bin/python scripts/debug_connectivity_soft_phases/run_phase17_20.py \
  --phase all \
  --runtime 20260608_1206 \
  --output-root outputs/debug/mappo_direct_specialists \
  --total-updates 2000 \
  --num-envs 8 \
  --rollout-steps 128 \
  --eval-interval 100 \
  --eval-episodes 20 \
  --epochs 4 \
  --minibatch-size 512 \
  --env-backend process
```

Primary output log:

- `outputs/debug/mappo_direct_specialists/phase17_20_20260608_1206.log`

Phase outputs are expected at:

- `outputs/debug/mappo_direct_specialists/phase17_connectivity_margin_penalty/20260608_1206/connectivity_expansion/trial01/s0/`
- `outputs/debug/mappo_direct_specialists/phase18_connectivity_action_risk_penalty/20260608_1206/connectivity_expansion/trial01/s0/`
- `outputs/debug/mappo_direct_specialists/phase19_connectivity_lagrangian_cost/20260608_1206/connectivity_expansion/trial01/s0/`
- `outputs/debug/mappo_direct_specialists/phase20_connectivity_auxiliary_prediction/20260608_1206/connectivity_expansion/trial01/s0/`

How to monitor:

```bash
tmux attach -t wf_conn_soft_20260608_1206
tail -f outputs/debug/mappo_direct_specialists/phase17_20_20260608_1206.log
tail -f outputs/debug/mappo_direct_specialists/phase17_connectivity_margin_penalty/20260608_1206/connectivity_expansion/trial01/s0/training_metrics.csv
```

Next action after phase17 finishes:

- read `phase17_connectivity_margin_penalty/20260608_1206/summary.json`;
- inspect `report.md`;
- confirm `connectivity_action_filter_enabled == 0` and
  `connectivity_filter_replacement_count == 0`;
- if phase17 violation decreases but coverage remains low, continue to phase18
  as already scheduled;
- if phase17 fails runtime, fix the runtime issue and rerun phase17 long-run,
  not a smoke test.

## Active phase17 trial02 branch

A second phase17 long-run branch was started after online analysis showed
trial01 became too conservative at the first eval.

- tmux session: `wf_conn_p17_t2_20260608_1206`
- output:
  `outputs/debug/mappo_direct_specialists/phase17_connectivity_margin_penalty/20260608_1206/connectivity_expansion/trial02/s0/`
- log:
  `outputs/debug/mappo_direct_specialists/phase17_trial02_20260608_1206.log`

Trial02 hypothesis:

- reduce margin/disconnect pressure enough to allow connected expansion;
- raise coverage/uncovered/repeat terms so coverage does not collapse into
  hover/revisit behavior;
- keep hard connectivity filter disabled.

Monitor after update 100:

```bash
python - <<'PY'
import csv
from pathlib import Path
p=Path('outputs/debug/mappo_direct_specialists/phase17_connectivity_margin_penalty/20260608_1206/connectivity_expansion/trial02/s0/training_metrics.csv')
rows=list(csv.DictReader(p.open()))
evals=[r for r in rows if r.get('eval_success_rate') not in ('', None)]
print(rows[-1].get('update'), len(evals), evals[-1] if evals else 'no eval yet')
PY
```

Compare trial01 vs trial02 on:

- `eval_connectivity_expansion_coverage_ratio_mean`
- `eval_connectivity_expansion_effective_explored_radius_mean`
- `eval_connectivity_expansion_connectivity_violation_rate_mean`
- `eval_connectivity_expansion_connectivity_margin_penalty_mean`
- `repeat_footprint_ratio`
- `uncovered_approach_gain`
- `connectivity_action_filter_enabled`
- `connectivity_filter_replacement_count`

## Post phase17-20 connectivity soft-constraint conclusion

Date: 2026-06-09.

All long runs under runtime `20260608_1206` have completed. None of the
soft-constraint phases solved target `connectivity_expansion` yet. Do not mark
the task as converged.

Best observed behaviors:

- `phase17/trial02` is the best base for the next attempt:
  best random eval success `0.35`, coverage `0.531`, effective radius `0.480`,
  but connectivity violation `0.228`; last eval reached coverage `0.582` but
  violation worsened to `0.448`.
- `phase18` confirms action-risk penalty is useful: last random eval violation
  dropped to `0.063` with action risk `0.025`, but coverage only `0.437`.
- `phase19` is too conservative as configured: random eval violation `0.000`,
  connected ratio `1.000`, but coverage only about `0.247`.
- `phase20` auxiliary losses learn correctly, but behavior is still not enough:
  last random eval coverage `0.399`, radius `0.420`, violation `0.151`.

Interpretation:

- The policy can currently optimize two of coverage, expansion radius, and low
  violation, but not all three together.
- `eval_reward` alone is misleading for connectivity; use success, coverage,
  effective radius, connectivity violation, connected ratio, and filter flags.
- Hard filter stayed off in these runs (`connectivity_action_filter_enabled=0`,
  `connectivity_filter_replacement_count=0`), so the results are valid
  soft-constraint evidence.

Recommended next phase:

- Start `phase21_connected_coverage_reward` from the `phase17/trial02` reward
  family, not from the phase19 Lagrangian family.
- Add dense reward for connected new coverage:
  `connected_new_coverage_gain`, optionally penalize
  `disconnected_new_coverage`.
- Keep a small phase18-style action-risk penalty as a secondary guard.
- Consider adding observation features for distance/margin to the nearest
  base-connected component before more scalar reward sweeps.

## Active phase21 connected-coverage reward run

Started on 2026-06-09.

- tmux session: `wf_conn_p21_20260609_1034`
- output:
  `outputs/debug/mappo_direct_specialists/phase21_connected_coverage_reward/20260609_1034/connectivity_expansion/trial01/s0/`
- phase report:
  `outputs/debug/mappo_direct_specialists/phase21_connected_coverage_reward/20260609_1034/report.md`
- run log:
  `outputs/debug/mappo_direct_specialists/phase21_connected_coverage_reward/20260609_1034/connectivity_expansion/trial01/s0/stdout.log`

Monitor command:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
p=Path('outputs/debug/mappo_direct_specialists/phase21_connected_coverage_reward/20260609_1034/connectivity_expansion/trial01/s0/training_metrics.csv')
df=pd.read_csv(p)
cols=[
  'update','eval_random_success_rate',
  'eval_random_connectivity_expansion_coverage_ratio_mean',
  'eval_random_connectivity_expansion_effective_explored_radius_mean',
  'eval_random_connectivity_expansion_connectivity_violation_rate_mean',
  'eval_random_connectivity_expansion_connected_new_coverage_gain_mean',
  'eval_random_connectivity_expansion_disconnected_new_coverage_gain_mean',
  'eval_random_connectivity_expansion_connected_new_coverage_ratio_mean',
  'eval_random_connectivity_expansion_disconnected_new_coverage_ratio_mean',
]
evals=df[df.get('eval_random_success_rate').notna()]
print(evals[[c for c in cols if c in evals.columns]].tail())
print(df[['update','coverage_ratio','connected_to_base_ratio','connectivity_violation_rate','connected_new_coverage_gain','disconnected_new_coverage_gain','action_disconnect_risk_mean','connectivity_action_filter_enabled','connectivity_filter_replacement_count']].tail())
PY
```

Decision rule after update 100 or 200:

- if connected coverage grows and violation is below phase17/trial02, continue;
- if coverage grows but `disconnected_new_coverage_gain` or violation remains
  high, open `phase21/trial02` with stronger
  `w_disconnected_new_coverage` and/or slightly stronger action-risk penalty;
- if coverage collapses, lower the disconnected penalty and raise
  `w_connected_new_coverage`;
- do not enable `connectivity_action_filter` for this learning path.

Phase21 trial status:

- `trial01`: stopped. It was the only branch with temporary nonzero success,
  but latest update `1000` eval fell back to success `0.0`, coverage `0.504`,
  radius `0.490`, violation `0.145`.
- `trial02`: stopped. It kept violation near zero but coverage/radius collapsed
  or stagnated; too conservative.
- `trial03`: stopped. It achieved high coverage/radius at times, but violation
  rose badly, recreating disconnected expansion.

Final phase21 report:

- `outputs/debug/mappo_direct_specialists/phase21_connected_coverage_reward/20260609_1034/phase21_stopped_report_zh.md`
- `outputs/debug/mappo_direct_specialists/phase21_connected_coverage_reward/20260609_1034/phase21_stopped_summary.json`

Conclusion:

- Do not continue scalar reward-only tuning for this task.
- Next phase should add connectivity observation features while keeping
  `connectivity_action_filter=false`.

Recommended next phase idea:

- `phase22_connectivity_observation_features`
- add per-agent/global observation features:
  - distance to nearest base-connected component;
  - communication margin to the base-connected component;
  - connected frontier / connected uncovered approach score;
  - relay gap or relay importance proxy.
- keep Direct waypoint policy and soft reward path;
- keep connected coverage reward from phase21 but use less aggressive scalar
  weight sweeps.

Future phase comparison metrics:

- `eval_random_success_rate`
- `eval_random_connectivity_expansion_coverage_ratio_mean`
- `eval_random_connectivity_expansion_effective_explored_radius_mean`
- `eval_random_connectivity_expansion_connectivity_violation_rate_mean`
- `eval_random_connectivity_expansion_connected_to_base_ratio_mean`
- `eval_random_connectivity_expansion_connected_new_coverage_gain_mean`
- `eval_random_connectivity_expansion_disconnected_new_coverage_gain_mean`
- `eval_random_connectivity_expansion_action_disconnect_risk_mean`
- `eval_random_connectivity_expansion_connectivity_action_filter_enabled_mean`
- `eval_random_connectivity_expansion_connectivity_filter_replacement_count_mean`

Default media rule:

- Every future experiment should save media by default.
- Prefer `record_eval_episodes >= 1` and `record_format=gif` for debug.
- Formal long runs should keep `record_eval_episodes >= 2` unless explicitly
  disabled for a documented resource reason.

### Next: Monitor Phase23 connectivity relaxed-success route

Active run:

- tmux: `wf_conn_p23_20260609_1517`
- metrics:
  `outputs/debug/mappo_direct_specialists/phase23_relaxed_success_graph_reward/20260609_1517/connectivity_expansion/trial01/s0/training_metrics.csv`

Decision at first evals:

- Continue if relaxed success becomes nonzero and violation remains below `0.20` with coverage near/above `0.45` and radius near/above `0.35`.
- If safe but coverage low, open `phase23/trial02` with stronger coverage/expand pressure and lower repeat penalty.
- If coverage good but violation too high, open `phase23/trial02` with lower `max_delta` or stronger disconnected/action-risk penalties.
- Do not add new connectivity-specific observation fields unless phase23 clearly fails; user prefers unified multitask observation.
- Do not enable hard connectivity action filter for learning phases.

### Immediate Phase23 monitoring rule

Watch trial01 and trial02 at eval updates 100/200:

- trial01 problem: coverage/radius pass relaxed thresholds but violation is too high (`0.334` at eval 100).
- trial02 target: keep coverage near or above `0.45`, radius above `0.35`, and reduce violation below `0.20`.
- If trial02 is safe but coverage collapses, tune toward an intermediate preset rather than adding observations.
- If trial02 still violates, consider phase24 curriculum/cost-schedule route with the same observation schema.

### Monitor Phase24 connected-only reward

Active run:

- tmux: `wf_conn_p24_20260609_1545`
- metrics:
  `outputs/debug/mappo_direct_specialists/phase24_connected_only_reward/20260609_1545/connectivity_expansion/trial01/s0/training_metrics.csv`

Decision at eval 100:

- Goal: relaxed success > phase23 with violation <= `0.20`.
- If coverage/radius too low but violation good, start phase24/trial02 with stronger connected coverage / expand reward.
- If coverage good but violation high, start phase24/trial02 with stronger disconnected/action-risk penalties or lower `max_delta`.
- Keep no hard filter and avoid new connectivity-specific observation features.

### Phase24 follow-up after update 100

Phase24/trial01 is currently the best connectivity route:

- relaxed success `0.95` at eval update `100`;
- no hard filter;
- connected coverage dominates disconnected coverage.

Next:

- Do not stop phase24/trial01 yet; watch eval update `200` and `300` for stability.
- If violation stays below `0.20`, open a later phase to tighten success toward official `0.55/0.45/0.10`.
- If violation drifts upward, add schedule/curriculum or slightly stronger `w_disconnect` / `w_action_disconnect_risk`, not new observation fields.

### Current Active Task: Monitor Phase25 tighter connected curriculum

Phase24 has completed. Treat it as repaired only for the relaxed debug threshold, not the formal target.

Best Phase24 evidence:

- checkpoint:
  `outputs/debug/mappo_direct_specialists/phase24_connected_only_reward/20260609_1545/connectivity_expansion/trial01/s0/checkpoints/checkpoint_best_eval.pt`
- best update: `900`
- relaxed eval success: `1.0`
- coverage: `0.466104`
- radius: `0.749350`
- violation: `0.063939`
- connected_new_coverage_ratio: `0.979412`
- hard filter: off, replacement count `0`

Active Phase25 run:

- tmux: `wf_conn_p25_20260610_0212`
- output:
  `outputs/debug/mappo_direct_specialists/phase25_tight_connected_curriculum/20260610_0212/connectivity_expansion/trial01/s0/`
- launcher log:
  `outputs/debug/mappo_direct_specialists/phase25_tight_connected_curriculum/20260610_0212/launcher_tmux.log`
- initialized from Phase24 best checkpoint using `--init-checkpoint`.

Phase25 target:

- coverage >= `0.50`
- radius >= `0.40`
- connectivity violation <= `0.15`
- keep `connectivity_action_filter_enabled=0`
- keep `connectivity_filter_replacement_count=0`

Immediate monitoring command:

```bash
/opt/conda/bin/python - <<'PY'
import csv
from pathlib import Path
p = Path('outputs/debug/mappo_direct_specialists/phase25_tight_connected_curriculum/20260610_0212/connectivity_expansion/trial01/s0/training_metrics.csv')
rows = list(csv.DictReader(p.open())) if p.exists() else []
for r in rows:
    if r.get('eval_success_rate') not in ('', None):
        print(
            r.get('update'),
            'succ', r.get('eval_success_rate'),
            'cov', r.get('eval_connectivity_expansion_coverage_ratio_mean'),
            'radius', r.get('eval_connectivity_expansion_effective_explored_radius_mean'),
            'viol', r.get('eval_connectivity_expansion_connectivity_violation_rate_mean'),
            'conn_cov_ratio', r.get('eval_connectivity_expansion_connected_new_coverage_ratio_mean'),
            'filter', r.get('eval_connectivity_expansion_connectivity_action_filter_enabled_mean'),
            'replace', r.get('eval_connectivity_expansion_connectivity_filter_replacement_count_mean'),
        )
PY
```

Decision rule:

- If update `100/200` meets coverage `0.50`, radius `0.40`, violation `0.15`, keep running and plan phase26 formal target `0.55/0.45/0.10`.
- If safe but under coverage, next phase should increase connected frontier/expand reward or use gradual threshold schedule.
- If coverage improves but violation rises, lower `max_delta` or raise disconnect/action-risk pressure.
- Do not add many connectivity-specific observation tricks; user wants future multitask compatibility.
- Do not re-enable hard connectivity action filter for training.

### Phase25 trial02 safety branch now active

Reason:

- Phase25/trial01 update `100` has good coverage/radius but violation is slightly high:
  - success `0.80`
  - coverage `0.543667`
  - radius `0.743184`
  - violation `0.164275`
  - hard filter `0`
- This is close to phase25 target, so open a parameter-only safety branch instead of changing observation or architecture.

Active trial02:

- tmux: `wf_conn_p25_t2_20260610_0212`
- output:
  `outputs/debug/mappo_direct_specialists/phase25_tight_connected_curriculum/20260610_0212/connectivity_expansion/trial02/s0/`
- tuning preset: `tight_safety`
- key changes vs trial01:
  - `max_delta/max_waypoint_distance: 0.17 -> 0.16`
  - lower exploration: `log_std_init -2.45 -> -2.55`, `log_std_max -1.05 -> -1.15`
  - stronger soft safety: `w_disconnect 13.5 -> 16.0`, `w_action_disconnect_risk 1.9 -> 2.6`, `w_connectivity_margin 1.9 -> 2.4`

Immediate comparison:

```bash
/opt/conda/bin/python - <<'PY'
import csv
from pathlib import Path
root = Path('outputs/debug/mappo_direct_specialists/phase25_tight_connected_curriculum/20260610_0212/connectivity_expansion')
for trial in ['trial01', 'trial02']:
    p = root / trial / 's0' / 'training_metrics.csv'
    rows = list(csv.DictReader(p.open())) if p.exists() else []
    evals = [r for r in rows if r.get('eval_success_rate') not in ('', None)]
    print('\\n', trial, 'last_update', rows[-1].get('update') if rows else 'n/a')
    for r in evals[-3:]:
        print(
            'eval', r.get('update'),
            'succ', r.get('eval_success_rate'),
            'cov', r.get('eval_connectivity_expansion_coverage_ratio_mean'),
            'radius', r.get('eval_connectivity_expansion_effective_explored_radius_mean'),
            'viol', r.get('eval_connectivity_expansion_connectivity_violation_rate_mean'),
            'risk', r.get('eval_connectivity_expansion_action_disconnect_risk_mean'),
        )
PY
```

### Current Active Task: Monitor Phase26 formal target

Phase25/trial01 has reached the tighter debug target:

- update `200`
- success `1.0`
- coverage `0.510046`
- radius `0.768290`
- violation `0.051491`
- hard filter `0`

Phase26 is now active to test the formal proxy target:

- tmux: `wf_conn_p26_20260610_0238`
- output:
  `outputs/debug/mappo_direct_specialists/phase26_formal_connected_target/20260610_0238/connectivity_expansion/trial01/s0/`
- initialized from:
  `outputs/debug/mappo_direct_specialists/phase25_tight_connected_curriculum/20260610_0212/connectivity_expansion/trial01/s0/checkpoints/checkpoint_best_eval.pt`

Formal target for phase26:

- coverage >= `0.55`
- radius >= `0.45`
- connectivity violation <= `0.10`
- hard filter remains off.

Monitor command:

```bash
/opt/conda/bin/python - <<'PY'
import csv
from pathlib import Path
paths = {
    'p25_t1': Path('outputs/debug/mappo_direct_specialists/phase25_tight_connected_curriculum/20260610_0212/connectivity_expansion/trial01/s0/training_metrics.csv'),
    'p25_t2': Path('outputs/debug/mappo_direct_specialists/phase25_tight_connected_curriculum/20260610_0212/connectivity_expansion/trial02/s0/training_metrics.csv'),
    'p26_t1': Path('outputs/debug/mappo_direct_specialists/phase26_formal_connected_target/20260610_0238/connectivity_expansion/trial01/s0/training_metrics.csv'),
}
for name, p in paths.items():
    rows = list(csv.DictReader(p.open())) if p.exists() else []
    print('\\n', name, 'last_update', rows[-1].get('update') if rows else 'n/a')
    for r in [r for r in rows if r.get('eval_success_rate') not in ('', None)][-3:]:
        print(
            'eval', r.get('update'),
            'succ', r.get('eval_success_rate'),
            'cov', r.get('eval_connectivity_expansion_coverage_ratio_mean'),
            'radius', r.get('eval_connectivity_expansion_effective_explored_radius_mean'),
            'viol', r.get('eval_connectivity_expansion_connectivity_violation_rate_mean'),
            'conn_cov_ratio', r.get('eval_connectivity_expansion_connected_new_coverage_ratio_mean'),
            'filter', r.get('eval_connectivity_expansion_connectivity_action_filter_enabled_mean'),
        )
PY
```

Decision rule:

- If phase26 reaches success with coverage >= `0.55` and violation <= `0.10`, keep running and then launch repeat seeds.
- If phase26 coverage stays near `0.51` with safe violation, open phase27 coverage-frontier curriculum.
- If phase26 violation rises above `0.10`, open a safety-tight branch from phase25/trial02 or lower `max_delta`.
- Do not add more task-specific observation fields unless reward/curriculum clearly fails.

### Current Active Task: Monitor Phase27 formal safety balance

Phase26 update `100` crossed coverage/radius but missed formal violation:

- success `0.70`
- coverage `0.556144`
- radius `0.688906`
- violation `0.127930`
- hard filter `0`

Phase27 is active as a safety-balanced branch:

- tmux: `wf_conn_p27_20260610_0252`
- output:
  `outputs/debug/mappo_direct_specialists/phase27_formal_safety_balance/20260610_0252/connectivity_expansion/trial01/s0/`
- initialized from:
  `outputs/debug/mappo_direct_specialists/phase26_formal_connected_target/20260610_0238/connectivity_expansion/trial01/s0/checkpoints/checkpoint_best_eval.pt`

Resource note:

- Stopped `wf_conn_p25_20260610_0212` after phase25/trial01 had already reached tightened success and its checkpoint was inherited by phase26.
- Stopped `wf_conn_p25_t2_20260610_0212` because it was safer but lower value than trial01 and not the formal-target branch.
- Current active sessions should be only:
  - `wf_conn_p26_20260610_0238`
  - `wf_conn_p27_20260610_0252`

Monitor command:

```bash
/opt/conda/bin/python - <<'PY'
import csv
from pathlib import Path
paths = {
    'p25_t1': Path('outputs/debug/mappo_direct_specialists/phase25_tight_connected_curriculum/20260610_0212/connectivity_expansion/trial01/s0/training_metrics.csv'),
    'p26_t1': Path('outputs/debug/mappo_direct_specialists/phase26_formal_connected_target/20260610_0238/connectivity_expansion/trial01/s0/training_metrics.csv'),
    'p27_t1': Path('outputs/debug/mappo_direct_specialists/phase27_formal_safety_balance/20260610_0252/connectivity_expansion/trial01/s0/training_metrics.csv'),
}
for name, p in paths.items():
    rows = list(csv.DictReader(p.open())) if p.exists() else []
    print('\\n', name, 'last_update', rows[-1].get('update') if rows else 'n/a')
    for r in [r for r in rows if r.get('eval_success_rate') not in ('', None)][-3:]:
        print(
            'eval', r.get('update'),
            'succ', r.get('eval_success_rate'),
            'cov', r.get('eval_connectivity_expansion_coverage_ratio_mean'),
            'radius', r.get('eval_connectivity_expansion_effective_explored_radius_mean'),
            'viol', r.get('eval_connectivity_expansion_connectivity_violation_rate_mean'),
            'conn_cov_ratio', r.get('eval_connectivity_expansion_connected_new_coverage_ratio_mean'),
            'filter', r.get('eval_connectivity_expansion_connectivity_action_filter_enabled_mean'),
        )
PY
```

Decision rule:

- If phase27 reaches coverage >= `0.55`, radius >= `0.45`, violation <= `0.10`, keep running and schedule repeat seeds.
- If phase27 violation is safe but coverage drops below `0.55`, open phase28 scheduled coverage-pressure branch.
- If phase27 still violates, use a stronger curriculum/schedule; do not enable hard action filter.

### Current Active Task: Continue Phase26 only

Phase27 was stopped after update `100` because it underperformed phase26:

- phase27 success `0.60`
- coverage `0.570651`
- violation `0.147820`
- connected_new_coverage_ratio `0.35`

Active session:

- `wf_conn_p26_20260610_0238`

Phase26 update `200` is near formal convergence:

- success `0.85`
- coverage `0.585064`
- radius `0.796046`
- violation `0.108739`
- hard filter `0`

Next:

- Wait for phase26 update `300`.
- If phase26 violation <= `0.10`, keep running and then repeat seeds.
- If phase26 still hovers around `0.105-0.115`, open phase28 with a schedule/curriculum rather than static stronger safety weights.

### Phase26 update 300 status

Phase26 update `300`:

- success `0.65`
- coverage `0.567747`
- radius `0.757831`
- violation `0.097835`
- hard filter `0`

Interpretation:

- Average formal metrics pass, but per-episode success is not stable.
- Continue phase26 to update `400`.
- Do not launch phase28 until update `400` confirms instability.

### Current Active Task: Monitor Phase28 stability fine-tune

Phase26 was stopped after update `400` because late training drifted:

- update `400` success `0.40`
- coverage `0.626674`
- violation `0.158631`
- hard filter `0`

Best phase26 checkpoint remains update `200`:

- success `0.85`
- coverage `0.585064`
- radius `0.796046`
- violation `0.108739`

Active phase28:

- tmux: `wf_conn_p28_20260610_0327`
- output:
  `outputs/debug/mappo_direct_specialists/phase28_formal_stability_finetune/20260610_0327/connectivity_expansion/trial01/s0/`
- initialized from phase26 best checkpoint.

Monitor command:

```bash
/opt/conda/bin/python - <<'PY'
import csv
from pathlib import Path
p = Path('outputs/debug/mappo_direct_specialists/phase28_formal_stability_finetune/20260610_0327/connectivity_expansion/trial01/s0/training_metrics.csv')
rows = list(csv.DictReader(p.open())) if p.exists() else []
print('last_update', rows[-1].get('update') if rows else 'n/a')
for r in [r for r in rows if r.get('eval_success_rate') not in ('', None)][-5:]:
    print(
        'eval', r.get('update'),
        'succ', r.get('eval_success_rate'),
        'cov', r.get('eval_connectivity_expansion_coverage_ratio_mean'),
        'radius', r.get('eval_connectivity_expansion_effective_explored_radius_mean'),
        'viol', r.get('eval_connectivity_expansion_connectivity_violation_rate_mean'),
        'conn_cov_ratio', r.get('eval_connectivity_expansion_connected_new_coverage_ratio_mean'),
        'filter', r.get('eval_connectivity_expansion_connectivity_action_filter_enabled_mean'),
    )
PY
```

Decision:

- If phase28 update `100/200` improves over phase26 best (`success 0.85`, violation `0.1087`), keep it.
- If phase28 cannot improve, next step should be explicit scheduled curriculum/cost multiplier, not more static reward tweaks.

### Connectivity current status after Phase29

Phase29 robust eval shows phase26 best checkpoint passes the formal proxy on 100 randomized episodes:

- checkpoint:
  `outputs/debug/mappo_direct_specialists/phase26_formal_connected_target/20260610_0238/connectivity_expansion/trial01/s0/checkpoints/checkpoint_best_eval.pt`
- eval output:
  `outputs/debug/mappo_direct_specialists/phase29_checkpoint_robust_eval/20260610_0352/connectivity_expansion/eval_phase26_best/`
- success_rate `0.82`
- coverage_ratio `0.581916`
- effective_explored_radius `0.799667`
- connectivity_violation_rate `0.095783`
- hard filter `0`
- replacement count `0`
- media saved for 2 episodes.

Status:

- connectivity_expansion is `CONVERGED_SINGLE_SEED_PROXY`.
- Do not keep static reward tweaking this seed unless repeat-seed validation fails.

Recommended next actions:

- Run repeat-seed validation/training for phase26 recipe, preferably seeds `1` and `2`, initialized either from scratch using phase26 config sequence or from phase25/phase26 checkpoint if the goal is continuation robustness.
- If repeat seeds fail due violation, implement explicit schedule/cost multiplier; do not use hard action filter.
- If repeat seeds pass, update documentation that connectivity_expansion is repaired under Direct MAPPO soft-constraint route.

### Connectivity status after Phase30 repeat eval

Current best checkpoint:

- `outputs/debug/mappo_direct_specialists/phase26_formal_connected_target/20260610_0238/connectivity_expansion/trial01/s0/checkpoints/checkpoint_best_eval.pt`

Evidence:

- Phase29 seed-base default, 100 episodes:
  - success `0.82`
  - coverage `0.581916`
  - radius `0.799667`
  - violation `0.095783`
- Phase30 seed `5000`, 100 episodes:
  - success `0.79`
  - coverage `0.581512`
  - radius `0.816807`
  - violation `0.089600`
- Phase30 seed `9000`, 100 episodes:
  - success `0.77`
  - coverage `0.570067`
  - radius `0.783409`
  - violation `0.109440`

Status:

- `connectivity_expansion`: `MOSTLY_CONVERGED_DIRECT_MAPPO_SOFT_CONSTRAINT`
- Hard filter is off in all reported evals.
- Coverage/radius are stable.
- Violation margin is narrow and can exceed `0.10` by around `0.01` under some random seeds.

If continuing:

- Do not add observation tricks first.
- Do not re-enable hard filter.
- Prefer one of:
  - train repeat seeds from phase26 recipe;
  - implement explicit connectivity cost multiplier schedule;
  - evaluate whether formal `max_violation_rate=0.10` should be relaxed slightly to `0.11-0.12` for randomized debug, while keeping the metric reported transparently.

### Active Phase31 training-seed reproducibility

Run root:

`outputs/debug/mappo_direct_specialists/phase31_connectivity_training_seed_reproducibility/20260610_0954/`

tmux:

`wf_phase31_20260610_0954`

Monitor:

```bash
scripts/debug_connectivity_soft_phases/status_phase31.sh 20260610_0954
```

TensorBoard:

```bash
/opt/conda/bin/tensorboard \
  --logdir outputs/debug/mappo_direct_specialists/phase31_connectivity_training_seed_reproducibility/20260610_0954 \
  --bind_all \
  --port 6011
```

Protocol:

- seeds `1,2,3`;
- each seed: Phase24 2000 -> Phase25 2000 -> Phase26 2000;
- no cross-seed checkpoint reuse;
- no short run or early termination;
- hard connectivity action/candidate filters disabled;
- final deterministic randomized eval: 100 episodes per seed.

After completion:

- read `aggregate_summary.json` and `aggregate_report.md`;
- pass requires all seed chains to be structurally valid and hard-filter-free;
- inspect mean/std/worst success, coverage, radius, violation, connected/disconnected new coverage, and action validity;
- if passed, freeze connectivity and move to the other specialists;
- if failed, choose Phase32A/B/C/D from the recorded failure taxonomy rather than adding another static reward sweep.

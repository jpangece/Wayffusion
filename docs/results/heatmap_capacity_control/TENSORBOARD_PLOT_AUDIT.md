# TensorBoard Plot Audit: Capacity-Controlled Heatmap Ablation

Date: 2026-07-24

## 1. Executive summary

All 15 required runs were present: five OFF, five REAL, and five ZERO. Every run contained one `training_metrics.csv`, one `best_eval_summary.json`, one TensorBoard event file, and a snapshot directory. The event files were readable, contained 111 normalized scalar tags, and had no duplicate scalar steps, non-monotonic/restarted sequences, or non-finite scalar values. Each CSV contained 338 columns and 200 ordered update rows, with evaluation values at the expected updates 20, 40, …, 200 (`tensorboard_plots/audit_manifest.json`).

TensorBoard-to-CSV comparisons produced 1,665 PASS results and no WARNING or FAIL results. PASS uses a recorded tolerance of `max(1e-4, 1e-6 × maximum compared value magnitude)` to accommodate TensorBoard float32 serialization. The largest observed absolute difference among the principal plotted metrics was `0.0001220703125` for gradient norm; all matching steps were present (`tensorboard_plots/audit_manifest.json`, `tensorboard_csv_consistency`).

The exporter produced 24 figures in both PNG and SVG, with one plotted-data CSV per figure. Eleven primary time-series metrics came from TensorBoard, two task metrics (`goal_achieved`, `arrival_rate`) came exclusively from CSV, and learning rate was skipped because it was absent from both sources (`tensorboard_plots/plot_manifest.json`).

## 2. Raw-data locations

- Read-only input root: `/Users/a...../heatmap_phd_evidence_inputs/outputs`
- Runs: `priority_inspection_heatmap_{off,real,zero}_ablation_seed{0..4}`
- Repository output: `docs/results/heatmap_capacity_control/tensorboard_plots/`
- Reproduction command: `MPLCONFIGDIR=/tmp/wayffusion-mpl XDG_CACHE_HOME=/tmp/wayffusion-cache /opt/anaconda3/envs/ml-env/bin/python scripts/analysis/export_heatmap_ablation_tensorboard_plots.py --raw-root "$HOME/heatmap_phd_evidence_inputs/outputs" --output-dir docs/results/heatmap_capacity_control/tensorboard_plots`

## 3. Run inventory

| Condition | Seeds | CSV/run | Best summary/run | Event files/run | Snapshot/run | Updates | Evaluation updates |
|---|---|---:|---:|---:|---:|---:|---|
| OFF | 0–4 | 1 | 1 | 1 | yes | 1–200 | 20,40,60,80,100,120,140,160,180,200 |
| REAL | 0–4 | 1 | 1 | 1 | yes | 1–200 | same |
| ZERO | 0–4 | 1 | 1 | 1 | yes | 1–200 | same |

Exact paths and per-run audit records are in `tensorboard_plots/audit_manifest.json` under `runs`.

## 4. TensorBoard tag inventory

The normalized inventory contains 111 tags, and every inventory count is 15 runs. The plot-relevant tags are:

- Performance: `eval_reward`, `eval_success_rate`, `mean_rollout_reward`, `weighted_poi_completion`.
- Optimization: `policy_loss`, `value_loss`, `entropy`, `grad_norm`, `approx_kl`, `clip_frac`, `explained_variance`.
- Supporting evaluation variants: `eval_fixed_*`, `eval_random_*`, generalization gaps, action validity, weighted completion and visited-POI ratios.
- Supporting rollout/control/task tags: action validity, collision/geofence/no-fly counts, policy standard deviation, delta statistics, connectivity metrics, coverage metrics, raw/penalized rewards, auxiliary losses and goal/coverage diagnostics.

The exact complete tag list, point counts, first/last steps, duplicates, restart flags, and non-finite counts is recorded per run and in normalized form in `tensorboard_plots/audit_manifest.json`. Training tags used here have 200 points at updates 1–200; evaluation tags have 10 points at updates 20–200.

## 5. CSV column inventory

The normalized inventory contains 338 columns in every run. It includes `update`, all TensorBoard-backed metrics, detailed rollout/control/task fields, fixed/random/formal evaluation summaries, artifact paths, and run counters. The fields used from CSV because no TensorBoard tag existed are `goal_achieved` and `arrival_rate`; `learning_rate` is absent. The exact 338-column list and its across-run occurrence count are in `tensorboard_plots/audit_manifest.json` under `normalized_csv_column_inventory`.

Each file has 200 rows, first update 1, last update 200, no duplicate or non-monotonic updates, and no non-finite numeric values. Blank cells are structurally concentrated in evaluation-only and checkpoint-path columns on non-evaluation/non-checkpoint rows; they are recorded per column and run rather than treated as numeric observations (`tensorboard_plots/audit_manifest.json`, `runs[].csv.missing_values_by_column`).

## 6. TensorBoard–CSV consistency

Only same-named, semantically unambiguous tags/columns were mapped. Comparisons use matching update steps and record maximum/mean absolute difference and missing steps. All 1,665 run/tag mappings passed; there were no missing matching steps for the principal plotted metrics.

| Metric | Runs passing | Maximum absolute difference | Figure source |
|---|---:|---:|---|
| Evaluation reward | 15/15 | 3.7519611e-05 | TensorBoard |
| Evaluation success rate | 15/15 | 0 | TensorBoard |
| Mean rollout reward | 15/15 | 5.9138983e-08 | TensorBoard |
| Weighted POI completion | 15/15 | 2.9802322e-08 | TensorBoard |
| Policy loss | 15/15 | 0 | TensorBoard |
| Value loss | 15/15 | 3.0517578e-05 | TensorBoard |
| Entropy | 15/15 | 2.9802322e-08 | TensorBoard |
| Gradient norm | 15/15 | 1.2207031e-04 | TensorBoard |
| Approximate KL / clip fraction / explained variance | 15/15 each | 0 | TensorBoard |

Complete comparisons are in `tensorboard_plots/audit_manifest.json` under `tensorboard_csv_consistency`.

## 7. Generated plot inventory

The authoritative inventory is `tensorboard_plots/plot_manifest.md` and its JSON counterpart. It lists 24 plots, each with PNG, SVG, plotted-data CSV, plot kind, and source. Primary time-series plots contain five thin seed curves per condition, an unsmoothed five-seed mean, and a sample-standard-deviation band; missing points are neither interpolated nor mixed across sources.

Primary performance plots: evaluation reward, evaluation success, mean rollout reward, weighted POI completion, goal achieved, and arrival rate. Optimization plots: policy loss, value loss, entropy, gradient norm, approximate KL, clip fraction, and explained variance. Paired plots cover the four requested REAL-minus-ZERO contrasts and final reward/success by condition. Five diagnostic plots cover best update, mean policy-vs-value loss, final reward-vs-success, gradient clipping context, and entropy endpoints.

## 8. Metric definitions

- Evaluation reward/success: the formal `eval_reward` and `eval_success_rate` values at the ten scheduled evaluation checkpoints.
- Mean rollout reward: `mean_rollout_reward` for each training update.
- Weighted POI completion: rollout `weighted_poi_completion`, the weighted fraction of visited POIs.
- Goal achieved / arrival rate: training-row `goal_achieved` and action-adapter `arrival_rate`; these exist only in CSV.
- Policy/value loss, entropy, gradient norm, approximate KL, clip fraction, explained variance: trainer update diagnostics with their repository-produced names.
- “Final” means the value at update 200. “Mean evaluation” means the arithmetic mean of the ten evaluation checkpoints within one training seed. Paired differences are computed seed-by-seed as REAL minus ZERO; evaluation checkpoints are not treated as independent replicates.

## 9. Descriptive observations

At update 200, mean evaluation reward was OFF `-633.25` (sample SD `347.57`), REAL `-293.73` (`111.99`), and ZERO `-368.14` (`164.81`) (`tensorboard_plots/evaluation_reward_plotted_data.csv`). Final REAL-minus-ZERO reward was positive for all five seeds: `115.17`, `4.30`, `33.42`, `177.82`, and `41.34` (`tensorboard_plots/real_minus_zero_final_evaluation_reward_by_seed_plotted_data.csv`). These are descriptive paired outcomes, not significance or causality claims.

At update 200, mean success was OFF `0.025`, REAL `0.100`, and ZERO `0.075`. Paired REAL-minus-ZERO final success values were `0.125`, `0`, `-0.125`, `0.125`, and `0` (`tensorboard_plots/evaluation_success_rate_plotted_data.csv`; `tensorboard_plots/real_minus_zero_final_evaluation_success_by_seed_plotted_data.csv`).

Final weighted POI completion means were OFF `0.295`, REAL `0.376`, and ZERO `0.355`, with sample SDs near `0.101`, `0.100`, and `0.121` respectively (`tensorboard_plots/weighted_poi_completion_plotted_data.csv`). The observed REAL–ZERO separation is smaller than the displayed between-seed bands, so the figure supports inspection, not a statistical conclusion.

## 10. Optimization-health observations

At update 200, policy-loss means remained near zero for all conditions, while value-loss means were OFF `29.31`, REAL `26.82`, and ZERO `27.56`; ZERO had the largest final cross-seed value-loss SD (`22.43`) (`tensorboard_plots/policy_loss_plotted_data.csv`; `tensorboard_plots/value_loss_plotted_data.csv`). Entropy means at update 200 were tightly grouped near `0.846` (`tensorboard_plots/entropy_plotted_data.csv`).

Gradient norm is shown with a 0.5 clipping-threshold reference recorded in the manifest, but the plotted logged gradient norm should not be interpreted without checking whether logging occurs before or after clipping (`tensorboard_plots/diagnostic_gradient_norm_with_clip_threshold.png`; `plot_manifest.json`). Approximate KL, clip fraction, and explained variance are provided as unsmoothed diagnostics, not proof of convergence.

## 11. Seed-variance observations

The mean (over ten evaluation checkpoints) REAL-minus-ZERO reward differences by seed were `10.53`, `-18.93`, `125.25`, `31.96`, and `-41.11`; their mean was `21.54` with sample SD `64.32` (`tensorboard_plots/real_minus_zero_mean_evaluation_reward_by_seed_plotted_data.csv`). The corresponding success differences were `0`, `0.0125`, `-0.0625`, `0.0125`, and `0.0625`, mean `0.005`, sample SD `0.0447` (`tensorboard_plots/real_minus_zero_mean_evaluation_success_by_seed_plotted_data.csv`). Direction varies across seeds for checkpoint-averaged contrasts even though the final reward contrast is positive in all five seeds.

## 12. Important limitations

1. There are five independent training seeds per condition; ten repeated evaluation checkpoints within a seed are longitudinal measurements, not ten independent samples.
2. No smoothing or interpolation is applied. Bands are sample SD, not confidence intervals.
3. The audit is descriptive and does not establish causality, statistical significance, or convergence.
4. Evaluation values are based on eight configured evaluation episodes per checkpoint; success rates are correspondingly discrete.
5. CSV-only curves are never mixed pointwise with TensorBoard curves.
6. “Best evaluation update” follows each raw `best_eval_summary.json`; it is diagnostic and selection-biased by construction.

## 13. Data-quality issues

- No required run or artifact was missing; all snapshots were present.
- No TensorBoard file was unreadable. No event scalar was non-finite, duplicated by step, or non-monotonic/restarted.
- No CSV update was duplicated or non-monotonic, and no numeric CSV value was non-finite.
- CSV blank cells reflect sparse evaluation/checkpoint columns. The audit preserves their counts.
- TensorBoard stores scalar values at float32 precision; small differences from CSV were handled by the explicit recorded tolerance.
- Learning rate was unavailable in both TensorBoard and CSV and was therefore not plotted.

## 14. Recommended figures for PhD review

The curated set is documented in `PHD_FIGURE_INDEX.md`: evaluation reward, evaluation success, weighted POI completion, policy loss, value loss, entropy, paired final REAL-minus-ZERO reward, and final reward by condition. These eight jointly show performance trajectories, semantic-task behavior, optimization health, paired-seed behavior, and between-condition endpoint variation without presenting diagnostics as inferential statistics.

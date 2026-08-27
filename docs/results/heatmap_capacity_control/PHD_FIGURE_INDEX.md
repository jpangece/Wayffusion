# PhD Figure Index: Capacity-Controlled Heatmap Ablation

This index selects eight figures from `tensorboard_plots/plot_manifest.json`. Seed curves and sample-standard-deviation bands are descriptive; no figure establishes statistical significance or causality.

## 1. Evaluation reward trajectory

- Filename: `tensorboard_plots/evaluation_reward.png` (SVG: `evaluation_reward.svg`)
- Metric: formal evaluation reward at updates 20–200
- Data source: TensorBoard; exact values in `evaluation_reward_plotted_data.csv`
- Why it matters: primary task-return trajectory across OFF, REAL, and capacity-matched ZERO.
- Caption: Thin curves show five independent training seeds; thick curves and bands show the five-seed mean and sample SD without smoothing. REAL has a higher mean endpoint than OFF and ZERO in these runs, while substantial seed variation remains.
- Key question: Does the semantic heatmap condition show a repeatable reward advantage beyond the capacity-matched ZERO input?
- Caveat: Ten checkpoints within a seed are repeated measurements, not independent samples.

## 2. Evaluation success trajectory

- Filename: `tensorboard_plots/evaluation_success_rate.png` (SVG: `evaluation_success_rate.svg`)
- Metric: formal evaluation success rate
- Data source: TensorBoard; `evaluation_success_rate_plotted_data.csv`
- Why it matters: success is a thresholded mission outcome complementary to reward.
- Caption: The plot shows actual scheduled evaluation checkpoints only. Success remains low and discrete because each checkpoint aggregates eight evaluation episodes, so small vertical changes can represent one episode.
- Key question: Do reward differences correspond to more completed successful missions?
- Caveat: Low, discrete rates limit resolution and should not be overinterpreted.

## 3. Weighted POI completion trajectory

- Filename: `tensorboard_plots/weighted_poi_completion.png` (SVG: `weighted_poi_completion.svg`)
- Metric: rollout weighted fraction of visited POI priority
- Data source: TensorBoard; `weighted_poi_completion_plotted_data.csv`
- Why it matters: this is the task-specific completion measure most directly related to the semantic priority heatmap.
- Caption: The plot tracks weighted completion throughout training for all conditions and seeds. At update 200 the mean is highest for REAL, but its separation from ZERO is modest relative to the displayed seed dispersion.
- Key question: Does semantic priority information improve priority-weighted task completion rather than only reward shaping terms?
- Caveat: This is a rollout metric, not the formal evaluation success rate.

## 4. Policy loss

- Filename: `tensorboard_plots/policy_loss.png` (SVG: `policy_loss.svg`)
- Metric: clipped PPO policy loss
- Data source: TensorBoard; `policy_loss_plotted_data.csv`
- Why it matters: checks whether comparative performance is accompanied by visibly abnormal actor optimization.
- Caption: Unsmoothed per-update policy loss is shown for every seed, with condition-level mean and sample SD. The figure is an optimization diagnostic and should be read with KL and clip fraction rather than as a performance score.
- Key question: Does heatmap consumption create a distinct or unstable actor-update regime?
- Caveat: Loss magnitude near zero does not itself demonstrate convergence or policy quality.

## 5. Value loss

- Filename: `tensorboard_plots/value_loss.png` (SVG: `value_loss.svg`)
- Metric: value-function loss
- Data source: TensorBoard; `value_loss_plotted_data.csv`
- Why it matters: exposes critic-fit behavior after the heatmap feature widens the centralized critic input.
- Caption: All five seeds reveal large temporal and cross-seed variability that a mean-only curve would hide. The endpoint means are similar, while ZERO shows the largest endpoint sample SD in this bundle.
- Key question: Does the enabled critic exhibit a capacity- or semantics-specific fitting problem?
- Caveat: Value loss depends on return scale and nonstationary targets; cross-condition comparison is descriptive.

## 6. Entropy trajectory

- Filename: `tensorboard_plots/entropy.png` (SVG: `entropy.svg`)
- Metric: policy entropy
- Data source: TensorBoard; `entropy_plotted_data.csv`
- Why it matters: shows whether exploration collapses or diverges differently across the three matched protocols.
- Caption: Entropy trajectories are plotted without smoothing for each seed. Endpoint means are tightly grouped near 0.846, providing no obvious endpoint separation among conditions in this descriptive view.
- Key question: Are observed performance differences confounded by materially different exploration levels?
- Caveat: Entropy alone does not characterize action usefulness or state-dependent exploration.

## 7. Paired final REAL-minus-ZERO reward

- Filename: `tensorboard_plots/real_minus_zero_final_evaluation_reward_by_seed.png` (SVG counterpart available)
- Metric: update-200 evaluation reward difference, REAL minus ZERO, paired by training seed
- Data source: TensorBoard; `real_minus_zero_final_evaluation_reward_by_seed_plotted_data.csv`
- Why it matters: isolates semantic input from the capacity-matched zero-input architecture while retaining the five independent seed pairs.
- Caption: Each labeled point is one training seed; all five final differences are positive in this bundle. Magnitudes vary widely, so the plot communicates consistency at this endpoint without adding a significance claim.
- Key question: At the final checkpoint, does REAL outperform its capacity control within each matched seed?
- Caveat: This is one selected endpoint and should be read alongside the full trajectory and checkpoint-mean paired figure.

## 8. Final evaluation reward by condition

- Filename: `tensorboard_plots/final_evaluation_reward_by_condition.png` (SVG counterpart available)
- Metric: update-200 evaluation reward
- Data source: TensorBoard; `final_evaluation_reward_by_condition_plotted_data.csv`
- Why it matters: gives a compact endpoint comparison while displaying every training seed.
- Caption: Bars show condition means with sample-SD error bars; labeled points show individual seeds. REAL has the least-negative mean endpoint, while OFF exhibits the widest visible spread and includes one particularly low seed.
- Key question: How large is endpoint variation within and between OFF, REAL, and ZERO?
- Caveat: Bars are descriptive means, error bars are sample SD rather than confidence intervals, and no significance test is implied.

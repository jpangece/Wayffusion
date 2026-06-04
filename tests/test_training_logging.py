import shutil
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts._common import build_metric_logger, log_scalar_metrics, tensorboard_dir


def _scalar_tags(run_dir: Path) -> set[str]:
    tags: set[str] = set()
    for event_file in tensorboard_dir(run_dir).glob("events.out.tfevents*"):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags.update(accumulator.Tags().get("scalars", []))
    return tags


def test_tensorboard_logger_writes_event_file_and_console_output(capsys):
    run_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_artifacts" / "tb_logging_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    writer, log_record = build_metric_logger(
        run_dir,
        namespace="ppo/train",
        step_key="update",
        tensorboard_enabled=True,
        console_interval=1,
        key_order=["mean_rollout_reward", "policy_loss", "eval_reward"],
    )

    assert writer is not None
    log_record({"update": 1, "mean_rollout_reward": 1.5, "policy_loss": 0.25})
    log_record({"update": 2, "eval_reward": 2.0, "eval_success_rate": 0.5, "checkpoint_path": "dummy.pt"})
    writer.close()

    tb_dir = tensorboard_dir(run_dir)
    assert any(path.name.startswith("events.out.tfevents") for path in tb_dir.iterdir())

    stdout = capsys.readouterr().out
    assert "update=1" in stdout
    assert "eval_reward=2.000" in stdout

    shutil.rmtree(run_dir)


def test_tensorboard_core_mode_filters_noisy_training_metrics():
    run_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_artifacts" / "tb_core_filter_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    writer, log_record = build_metric_logger(
        run_dir,
        namespace="ppo/goal_nav/train",
        step_key="update",
        tensorboard_enabled=True,
        console_interval=100,
        tensorboard_metric_mode="core",
    )
    assert writer is not None
    log_record(
        {
            "update": 1,
            "mean_rollout_reward": 1.5,
            "policy_loss": 0.25,
            "eval_success_rate": 0.8,
            "eval_goal_nav_success_rate": 0.7,
            "eval_coverage_repeated_coverage_ratio": 0.12,
            "eval_coverage_demand_revisit_excess": 3.0,
            "debug_dense_component": 123.0,
            "eval_goal_nav_dense_reward_component": 456.0,
            "eval_goal_nav_reward_task_progress_reward": 789.0,
        }
    )
    writer.close()

    tags = _scalar_tags(run_dir)
    assert "ppo/goal_nav/train/mean_rollout_reward" in tags
    assert "ppo/goal_nav/train/policy_loss" in tags
    assert "ppo/goal_nav/train/eval_success_rate" in tags
    assert "ppo/goal_nav/train/eval_goal_nav_success_rate" in tags
    assert "ppo/goal_nav/train/eval_coverage_repeated_coverage_ratio" in tags
    assert "ppo/goal_nav/train/eval_coverage_demand_revisit_excess" in tags
    assert "ppo/goal_nav/train/debug_dense_component" not in tags
    assert "ppo/goal_nav/train/eval_goal_nav_dense_reward_component" not in tags
    assert "ppo/goal_nav/train/eval_goal_nav_reward_task_progress_reward" not in tags

    shutil.rmtree(run_dir)


def test_tensorboard_all_mode_keeps_all_numeric_training_metrics():
    run_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_artifacts" / "tb_all_mode_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    writer, log_record = build_metric_logger(
        run_dir,
        namespace="ppo/train",
        step_key="update",
        tensorboard_enabled=True,
        console_interval=100,
        tensorboard_metric_mode="all",
    )
    assert writer is not None
    log_record({"update": 1, "mean_rollout_reward": 1.5, "debug_dense_component": 123.0})
    writer.close()

    tags = _scalar_tags(run_dir)
    assert "ppo/train/mean_rollout_reward" in tags
    assert "ppo/train/debug_dense_component" in tags

    shutil.rmtree(run_dir)


def test_tensorboard_final_eval_core_mode_filters_single_point_noise():
    run_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_artifacts" / "tb_final_eval_filter_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    writer, _ = build_metric_logger(
        run_dir,
        namespace="ppo/train",
        step_key="update",
        tensorboard_enabled=True,
        console_interval=100,
        tensorboard_metric_mode="core",
    )
    assert writer is not None
    log_scalar_metrics(
        writer,
        "ppo/goal_nav/final_eval/N4/goal_nav",
        10,
        {
            "return_mean": 11.0,
            "success_rate_mean": 0.9,
            "collision_rate_mean": 0.01,
            "goal_coverage_ratio_mean": 0.98,
            "dense_reward_component_mean": 99.0,
        },
        tensorboard_metric_mode="core",
        metric_phase="final_eval",
    )
    writer.close()

    tags = _scalar_tags(run_dir)
    assert "ppo/goal_nav/final_eval/N4/goal_nav/return_mean" in tags
    assert "ppo/goal_nav/final_eval/N4/goal_nav/success_rate_mean" in tags
    assert "ppo/goal_nav/final_eval/N4/goal_nav/collision_rate_mean" in tags
    assert "ppo/goal_nav/final_eval/N4/goal_nav/goal_coverage_ratio_mean" in tags
    assert "ppo/goal_nav/final_eval/N4/goal_nav/dense_reward_component_mean" not in tags

    shutil.rmtree(run_dir)

from __future__ import annotations

from pathlib import Path

from scripts.formal_mappo_specialists.run_formal_specialists import (
    TASK_SPECS,
    build_train_command,
    task_output_dir,
)


def test_formal_commands_use_random_train_and_eval(tmp_path: Path):
    for task, spec in TASK_SPECS.items():
        output_dir = task_output_dir(tmp_path, "20260607_1200", task)
        command = build_train_command(
            task,
            output_dir,
            num_envs=16,
            rollout_steps=256,
            eval_interval=100,
            eval_episodes=20,
            record_eval_episodes=2,
            record_interval=5,
            env_backend="process",
        )
        joined = " ".join(command)
        assert "--train-randomization random" in joined
        assert "--eval-randomization random" in joined
        assert "--num_envs 16" in joined
        assert "--rollout_steps 256" in joined
        assert "--env_backend process" in joined
        assert f"--total_updates {spec['total_updates']}" in joined
        assert f"--output-dir {output_dir}" in joined
        assert command[command.index("--run_name") + 1] == "s0"


def test_formal_task_output_layout(tmp_path: Path):
    assert task_output_dir(tmp_path, "20260607_1200", "area_coverage") == (
        tmp_path / "20260607_1200" / "area_coverage" / "s0"
    )

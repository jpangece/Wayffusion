# Swarm30 v1

`swarm30_v1` is the frozen six-task scaling benchmark for 4, 10, 20, and 30 UAVs.

Tracks:

- `standardized_specialists`: one shared architecture and training budget per task.
- `tuned_specialists`: task-specific policy settings, reported as an upper-bound track.
- `generalist`: one six-task policy with 12,000 updates, matching the expected 2,000 updates per task.

The benchmark keeps sensing, communication, speed, and waypoint scales physically fixed while map area, grid resolution, task load, targets, and horizon grow with swarm size. It uses Direct MAPPO, the real vendored OpenAI MPE core, randomized training, deterministic fixed/random evaluation, TensorBoard, checkpoints, and media recording.

Start the queued run:

```bash
bash scripts/benchmarks/swarm30/launch_tmux.sh
```

The queue waits for the active Phase31 workers to finish before using GPUs 0-3.

Check status:

```bash
bash scripts/benchmarks/swarm30/status.sh
```

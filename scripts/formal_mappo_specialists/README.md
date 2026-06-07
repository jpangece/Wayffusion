# Formal Direct MAPPO Specialists

One-command formal training:

```bash
bash scripts/formal_mappo_specialists/launch_tmux.sh
```

Defaults:

- four tasks run concurrently, one task per GPU;
- 16 environments per task;
- each environment runs in its own process by default;
- 256 rollout steps;
- randomized training and randomized evaluation;
- area, belief, and priority run 2000 updates;
- connectivity runs 2500 updates;
- evaluation every 100 updates with 20 episodes;
- two GIFs are recorded every fifth evaluation;
- SMTP settings load from `.secrets/wayffusion_mail.env`.

Lifecycle emails are sent for suite start, each task start, each task completion or
abnormal termination, and final suite completion or failure.

Check status:

```bash
bash scripts/formal_mappo_specialists/status.sh
```

Attach to the session printed by the launcher:

```bash
tmux attach -t <session>
```

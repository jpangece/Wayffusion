# Waypoint MPE-like Dynamics Refactor

## Why Move Beyond Kinematic Point

The previous waypoint execution model moved each UAV geometrically toward a waypoint by at most `max_speed * decision_dt`. That was useful for smoke tests, but it hid several issues important for MARL and later real waypoint experiments:

- velocity was not a persistent state with continuous integration;
- control effort was not represented;
- damping, mass, contact forces, wind/noise, and low-level tracking behavior could not be ablated;
- policies could learn against an unrealistically sharp waypoint-to-position transition.

The new default keeps the external action interface as final waypoint, but inserts environment-side control and dynamics:

```text
final waypoint -> SafetyLayer -> ActionAdapter -> DynamicsBackend -> reward/metrics
```

## Boundary Between Adapter And Backend

`ActionAdapter` is part of environment transition dynamics, not policy. It converts already-decided final waypoints into low-level control commands.

`DynamicsBackend` integrates physical state. It updates position, velocity, path length, trajectory, battery, communication graph, and coverage footprints.

Neither layer samples policy actions or generates policy candidates.

## WaypointVelocityTracker

Default adapter: `WaypointVelocityTracker`.

For UAV `i`:

```text
e_i = w_i - p_i
d_i = ||e_i||
dir_i = e_i / (d_i + eps)
v_des_i = max_speed * min(1, d_i / slowdown_radius) * dir_i
```

If `d_i < acceptance_radius` and `hover_when_arrived=true`:

```text
v_des_i = 0
```

Velocity tracking:

```text
a_raw_i = velocity_gain * (v_des_i - v_i)
||a_raw_i|| <= max_accel
```

Optional smoothing:

```text
a_cmd_i = (1 - control_smoothing) * previous_control_i
          + control_smoothing * a_raw_i
```

`control_smoothing=1.0` means no smoothing.

## WaypointPDTracker

Debug/ablation adapter: `WaypointPDTracker`.

```text
a_cmd_i = kp * (w_i - p_i) - kd * v_i
||a_cmd_i|| <= max_accel
```

It is useful for checking whether the velocity-tracking controller itself is affecting learning or validation behavior.

## MPEParticleBackend

Default backend: `MPEParticleBackend`.

It implements MPE-like particle dynamics for Wayffusion tasks. It does not replace the task suite with OpenAI/PettingZoo MPE tasks.

For each physics substep:

```text
v_i <- (1 - damping) * v_i + (control_i + env_force_i) / mass_i * physics_dt
v_i <- clip_norm(v_i, max_speed)
p_i <- p_i + v_i * physics_dt
```

The backend supports:

- multiple substeps per decision step;
- max-speed clipping;
- mass and damping;
- optional collision/contact forces;
- optional wind/action/position noise;
- geofence projection;
- no-fly projection;
- path-length accumulation;
- trajectory updates;
- communication graph and coverage footprint updates.

## KinematicPointBackend

Debug/ablation backend: `KinematicPointBackend`.

It preserves the old bounded geometric waypoint step:

```text
p_next = p + dir_to_waypoint * min(distance_to_waypoint, max_speed * decision_dt)
```

Use it only for debugging or ablation against the default particle dynamics.

## Config Switches

Default:

```yaml
action_interface: waypoint
action_adapter:
  name: waypoint_velocity_tracker
dynamics_backend:
  name: mpe_particle
```

Kinematic debug:

```bash
python scripts/check/validate_waypoint_mappo.py \
  --tasks area_coverage \
  --policy-class direct_waypoint \
  --dynamics-backend kinematic_point \
  --num_agents 3 \
  --total_updates 1 \
  --eval_episodes 1 \
  --headless
```

PD adapter debug:

```yaml
action_adapter:
  name: waypoint_pd_tracker
```

## Ablation Matrix

Useful comparisons:

- `mpe_particle` vs `kinematic_point`;
- `waypoint_velocity_tracker` vs `waypoint_pd_tracker`;
- `candidate_selection_waypoint` vs `direct_waypoint`;
- deterministic dynamics vs noise-enabled dynamics;
- collision force enabled vs disabled.

Keep the environment action interface fixed as final waypoint for these ablations.

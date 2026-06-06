# Third-Party MPE Core Source

Wayffusion's waypoint MARL environment requires real MPE particle dynamics. The
preferred dependency is `mpe2`, followed by PettingZoo MPE internals. The local
validation environment did not have either package installed, so Wayffusion also
vendors the upstream OpenAI MPE core as a deterministic fallback.

## Upstream

- Repository: `https://github.com/openai/multiagent-particle-envs`
- Vendored file: `multiagent/core.py`
- Local path: `third_party/openai_mpe/core.py`
- Upstream README snapshot: `third_party/openai_mpe/UPSTREAM_README.md`

## License Status

The upstream repository did not expose a standalone `LICENSE`, `LICENSE.md`, or
`COPYING` file at vendoring time. The checked raw URLs returned 404. This status
is recorded in `third_party/openai_mpe/LICENSE`.

Before external redistribution, verify the upstream licensing status and replace
the placeholder license notice if a definitive license is available.

## Wayffusion Wrapper Boundary

Do not put Wayffusion-specific task, reward, observation, safety, or rendering
logic into `third_party/openai_mpe/core.py`.

The wrapper lives at:

```text
envs/waypoint/control/dynamics_backends.py::MPECoreBackend
```

The wrapper only:

- creates MPE `World` / `Agent` objects;
- synchronizes Wayffusion `UAVState` into MPE `Agent.state`;
- writes adapter output to `agent.action.u`;
- calls MPE `World.step()`;
- synchronizes MPE state back to Wayffusion.

# Third-Party MPE Core Source

Wayffusion's waypoint MARL environment requires real MPE particle dynamics. The
default source is the vendored local OpenAI MPE core in
`third_party/openai_mpe/core.py`. Installed package sources such as `mpe2` are
manual alternatives, not automatic fallbacks.

## Upstream

- Repository: `https://github.com/openai/multiagent-particle-envs`
- Vendored file: `multiagent/core.py`
- Local path: `third_party/openai_mpe/core.py`
- Upstream README snapshot: `third_party/openai_mpe/UPSTREAM_README.md`

## Runtime Source Selection

`MPECoreBackend` does not automatically fall back across sources. The source is
selected explicitly through `dynamics_backend.source`.

Supported values:

- `third_party_openai_mpe`: imports `third_party.openai_mpe.core`; default.
- `mpe2`: imports `mpe2._mpe_utils.core`; requires installing `mpe2`.
- `pettingzoo_mpe`: imports `pettingzoo.mpe._mpe_utils.core`; only works for PettingZoo versions exposing that internal module.

The source actually used by a run is reported in validation/training metrics as
`mpe_source`. If a configured source cannot be imported, backend construction
fails and the user must edit the env config manually.

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

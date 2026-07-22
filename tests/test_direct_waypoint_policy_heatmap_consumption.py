from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from policies import build_policy
from policies.direct_waypoint_policy import DirectWaypointPolicy


class _Space:
    def __init__(self, shape):
        self.shape = tuple(shape)


def _observation_space(*, channels: int = 2, grid_size: int = 11, include_heatmap: bool = True):
    space = {
        "task_field": _Space((5, grid_size, grid_size)),
        "all_uav_states": _Space((3, 8)),
        "task_id": _Space((6,)),
        "global_info": _Space((8,)),
        "agent_mask": _Space((3,)),
    }
    if include_heatmap:
        space["task_element_heatmap"] = _Space((channels, grid_size, grid_size))
    return space


def _observations(
    *,
    batch_size: int = 2,
    channels: int = 2,
    grid_size: int = 11,
) -> dict[str, torch.Tensor]:
    task_id = torch.zeros((batch_size, 6), dtype=torch.float32)
    task_id[:, 2] = 1.0
    return {
        "task_field": torch.zeros((batch_size, 5, grid_size, grid_size), dtype=torch.float32),
        "all_uav_states": torch.full((batch_size, 3, 8), 0.25, dtype=torch.float32),
        "task_id": task_id,
        "global_info": torch.zeros((batch_size, 8), dtype=torch.float32),
        "agent_mask": torch.ones((batch_size, 3), dtype=torch.float32),
        "task_element_heatmap": torch.zeros(
            (batch_size, channels, grid_size, grid_size),
            dtype=torch.float32,
        ),
    }


def _enabled_policy(*, channels: int = 2, grid_size: int = 11) -> DirectWaypointPolicy:
    torch.manual_seed(101)
    return DirectWaypointPolicy(
        _observation_space(channels=channels, grid_size=grid_size),
        cnn_channels=[8, 16],
        agent_hidden_dim=16,
        joint_hidden_dim=24,
        use_attention=False,
        use_task_element_heatmap=True,
    )


def _spatially_distinct_heatmaps(
    *,
    batch_size: int,
    channels: int,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.zeros((batch_size, channels, grid_size, grid_size), dtype=torch.float32)
    second = torch.zeros_like(first)
    first[:, :, 1, 1] = 1.0
    first[:, :, 1, 2] = 1.0
    second[:, :, grid_size // 2, grid_size // 2] = 1.0
    second[:, :, grid_size - 2, grid_size - 2] = 1.0
    assert torch.equal(first.sum(dim=(-2, -1)), second.sum(dim=(-2, -1)))
    return first, second


def test_build_policy_exposes_explicit_heatmap_opt_in():
    policy = build_policy(
        {
            "policy_class": "direct_waypoint",
            "cnn_channels": [8, 16],
            "agent_hidden_dim": 16,
            "joint_hidden_dim": 24,
            "use_attention": False,
            "use_task_element_heatmap": True,
        },
        _observation_space(),
        None,
    )

    assert policy.use_task_element_heatmap is True
    assert isinstance(policy.heatmap_encoder, torch.nn.Module)


def test_disabled_mode_preserves_structure_outputs_and_strict_checkpoints():
    torch.manual_seed(37)
    baseline = DirectWaypointPolicy(
        _observation_space(include_heatmap=False),
        cnn_channels=[8, 16],
        agent_hidden_dim=16,
        joint_hidden_dim=24,
        use_attention=False,
    )
    torch.manual_seed(37)
    disabled = build_policy(
        {
            "policy_class": "direct_waypoint",
            "cnn_channels": [8, 16],
            "agent_hidden_dim": 16,
            "joint_hidden_dim": 24,
            "use_attention": False,
            "use_task_element_heatmap": False,
        },
        _observation_space(),
        None,
    )

    baseline_state = baseline.state_dict()
    disabled_state = disabled.state_dict()
    assert baseline_state.keys() == disabled_state.keys()
    assert {
        key: tuple(value.shape) for key, value in baseline_state.items()
    } == {
        key: tuple(value.shape) for key, value in disabled_state.items()
    }
    assert not hasattr(disabled, "heatmap_encoder")
    assert all("heatmap" not in name for name, _ in disabled.named_parameters())
    disabled.load_state_dict(baseline_state, strict=True)
    baseline.load_state_dict(disabled.state_dict(), strict=True)

    obs_without_heatmap = _observations()
    obs_without_heatmap.pop("task_element_heatmap")
    obs_with_heatmap = deepcopy(obs_without_heatmap)
    obs_with_heatmap["task_element_heatmap"] = torch.rand((2, 2, 11, 11))
    baseline_delta, baseline_value, _ = disabled(obs_without_heatmap)
    heatmap_delta, heatmap_value, _ = disabled(obs_with_heatmap)
    assert torch.equal(baseline_delta, heatmap_delta)
    assert torch.equal(baseline_value, heatmap_value)


def test_enabled_policy_uses_separate_spatial_encoder_without_changing_task_field_encoder():
    torch.manual_seed(19)
    baseline = DirectWaypointPolicy(
        _observation_space(include_heatmap=False),
        cnn_channels=[8, 16],
        agent_hidden_dim=16,
        joint_hidden_dim=24,
        use_attention=False,
    )
    policy = _enabled_policy()

    assert isinstance(policy.heatmap_encoder, torch.nn.Module)
    assert any(isinstance(module, torch.nn.Conv2d) for module in policy.heatmap_encoder.modules())
    assert any(
        isinstance(module, torch.nn.AdaptiveAvgPool2d)
        for module in policy.heatmap_encoder.modules()
    )
    assert policy.field_encoder[0].in_channels == baseline.field_encoder[0].in_channels == 5
    heatmap_parameter_ids = {id(parameter) for parameter in policy.heatmap_encoder.parameters()}
    assert heatmap_parameter_ids
    assert heatmap_parameter_ids <= {id(parameter) for parameter in policy.parameters()}


@pytest.mark.parametrize("batch_size", [1, 4])
def test_enabled_policy_preserves_batch_actor_and_critic_shapes(batch_size):
    policy = _enabled_policy()
    observations = _observations(batch_size=batch_size)

    delta_mean, value, agent_mask = policy(observations)

    assert delta_mean.shape == (batch_size, 3, 2)
    assert value.shape == (batch_size,)
    assert agent_mask.shape == (batch_size, 3)
    encoded = policy.heatmap_encoder(observations["task_element_heatmap"])
    assert encoded.shape[0] == batch_size


def test_equal_mass_spatial_arrangements_change_actor_and_critic_outputs():
    policy = _enabled_policy()
    first_heatmap, second_heatmap = _spatially_distinct_heatmaps(
        batch_size=2,
        channels=2,
        grid_size=11,
    )
    first_obs = _observations()
    second_obs = deepcopy(first_obs)
    first_obs["task_element_heatmap"] = first_heatmap
    second_obs["task_element_heatmap"] = second_heatmap

    first_delta, first_value, _ = policy(first_obs)
    second_delta, second_value, _ = policy(second_obs)

    assert first_delta.shape == second_delta.shape == (2, 3, 2)
    assert first_value.shape == second_value.shape == (2,)
    assert not torch.allclose(first_delta, second_delta)
    assert not torch.allclose(first_value, second_value)


def _assert_finite_heatmap_gradients(policy: DirectWaypointPolicy) -> None:
    gradients = [parameter.grad for parameter in policy.heatmap_encoder.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_heatmap_encoder_has_actor_and_critic_gradient_paths():
    policy = _enabled_policy()
    observations = _observations()
    first, _ = _spatially_distinct_heatmaps(batch_size=2, channels=2, grid_size=11)
    observations["task_element_heatmap"] = first

    delta_mean, _, _ = policy(observations)
    delta_mean.sum().backward()
    _assert_finite_heatmap_gradients(policy)

    policy.zero_grad(set_to_none=True)
    _, value, _ = policy(observations)
    value.sum().backward()
    _assert_finite_heatmap_gradients(policy)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (None, "task_element_heatmap"),
        (torch.zeros((2, 11, 11)), "task_element_heatmap"),
        (torch.zeros((2, 1, 11, 11)), "task_element_heatmap"),
        (torch.zeros((2, 2, 10, 11)), "task_element_heatmap"),
        (torch.zeros((2, 2, 11, 10)), "task_element_heatmap"),
    ],
)
def test_enabled_policy_rejects_missing_or_malformed_heatmap(replacement, message):
    policy = _enabled_policy()
    observations = _observations()
    if replacement is None:
        observations.pop("task_element_heatmap")
    else:
        observations["task_element_heatmap"] = replacement

    with pytest.raises((ValueError, KeyError), match=message):
        policy(observations)


def test_enabled_policy_requires_heatmap_in_observation_space():
    with pytest.raises((ValueError, KeyError), match="task_element_heatmap"):
        DirectWaypointPolicy(
            _observation_space(include_heatmap=False),
            cnn_channels=[8, 16],
            agent_hidden_dim=16,
            joint_hidden_dim=24,
            use_attention=False,
            use_task_element_heatmap=True,
        )


def test_enabled_architecture_is_not_strictly_compatible_with_disabled_checkpoint():
    torch.manual_seed(41)
    disabled = DirectWaypointPolicy(
        _observation_space(include_heatmap=False),
        cnn_channels=[8, 16],
        agent_hidden_dim=16,
        joint_hidden_dim=24,
        use_attention=False,
    )
    enabled = _enabled_policy()

    with pytest.raises(RuntimeError):
        enabled.load_state_dict(disabled.state_dict(), strict=True)

"""Tests for the object-aware baseline extension hooks in the coupled solver."""

from __future__ import annotations

import numpy as np
import torch

from core4d_dual_retarget.coordination_eval import coordination_metrics, flatten_metrics
from core4d_dual_retarget.retarget.coupled import (
    CoupledRetargetConfig,
    gaussian_smoothing_matrix,
    joint_carry_activation,
    torch_soft_contact_indicator,
    torch_soft_contact_indicator_surface_aware,
)
from tests.test_coordination_eval import _signals


def _unit_cube_hull() -> torch.Tensor:
    # Planes a.x + b <= 0 inside the cube [-0.5, 0.5]^3.
    eqs = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            eqs.append(np.concatenate([normal, [-0.5]]))
    return torch.tensor(np.array(eqs), dtype=torch.float64)


def test_defaults_keep_frozen_solver() -> None:
    config = CoupledRetargetConfig()
    assert config.solve_robots == (1, 2)
    assert config.joint_contact_continuity_weight == 0.0
    # Visual-refinement options must be off so EXT-JC w10 / v41 are reproduced.
    assert config.joint_contact_surface_aware is False
    assert config.joint_contact_softmin_beta_m == 0.0
    assert config.joint_contact_edge_taper_frames == 0
    assert config.hand_cartesian_jerk_weight == 0.0
    assert config.reference_trust_weight == 0.0
    assert config.smoothed_hand_barrier_weight == 0.0


def test_gaussian_smoothing_matrix_matches_recipe() -> None:
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(0)
    values = rng.normal(size=(40, 5))
    for sigma in (1.0, 0.35):
        expected = gaussian_filter1d(values, sigma=sigma, axis=0, mode="nearest")
        assert np.allclose(gaussian_smoothing_matrix(40, sigma) @ values, expected, atol=1e-12)
    assert np.array_equal(gaussian_smoothing_matrix(7, 0.0), np.eye(7))


def test_surface_aware_indicator_has_no_gradient_inside_hull() -> None:
    hull = _unit_cube_hull()
    deep = torch.tensor([[[0.40, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    shallow = torch.tensor([[[0.49, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    kwargs = {"tau_m": 0.04, "beta_m": 0.01}
    # Original indicator keeps increasing with depth -> rewards penetration.
    assert float(torch_soft_contact_indicator(deep, hull, **kwargs)) > float(
        torch_soft_contact_indicator(shallow, hull, **kwargs)
    )
    s_deep = torch_soft_contact_indicator_surface_aware(deep, hull, **kwargs)
    s_shallow = torch_soft_contact_indicator_surface_aware(shallow, hull, **kwargs)
    # Saturated at sigmoid(tau/beta) up to the softplus tail (eps = 2 mm).
    assert abs(float(s_deep) - float(s_shallow)) < 1e-4
    s_deep.sum().backward()
    assert deep.grad is not None and float(deep.grad.abs().max()) < 1e-9
    # Outside the hull it still attracts (positive gradient towards the surface).
    far = torch.tensor([[[0.60, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    s_far = torch_soft_contact_indicator_surface_aware(far, hull, **kwargs)
    s_far.sum().backward()
    assert far.grad is not None and float(far.grad[0, 0, 0]) < 0
    assert float(s_far) < float(s_shallow)


def test_surface_aware_product_still_couples_both_robots() -> None:
    hull = _unit_cube_hull()
    x1 = torch.tensor([[[0.56, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)

    def grad_x1(x2_value: float) -> torch.Tensor:
        x2 = torch.tensor([[[x2_value, 0.0, 0.0]]], dtype=torch.float64)
        s1 = torch_soft_contact_indicator_surface_aware(x1, hull, tau_m=0.04, beta_m=0.01)
        s2 = torch_soft_contact_indicator_surface_aware(x2, hull, tau_m=0.04, beta_m=0.01)
        (g,) = torch.autograd.grad((1.0 - s1 * s2).sum(), x1)
        return g

    assert float((grad_x1(0.56) - grad_x1(0.75)).abs().max()) > 1e-6


def test_joint_carry_activation_taper() -> None:
    mask = np.zeros(20, dtype=bool)
    mask[5:15] = True
    hard = joint_carry_activation(mask, 0)
    assert np.array_equal(hard, mask.astype(float))
    soft = joint_carry_activation(mask, 3)
    assert soft[4] == 0.0 and soft[15] == 0.0
    assert np.allclose(soft[5:8], [0.25, 0.5, 0.75])
    assert np.allclose(soft[12:15], [0.75, 0.5, 0.25])
    assert np.all(soft[8:12] == 1.0)
    # Two windows, one shorter than the taper.
    mask2 = np.zeros(12, dtype=bool)
    mask2[1:3] = True
    mask2[6:12] = True
    soft2 = joint_carry_activation(mask2, 2)
    assert np.allclose(soft2[1:3], [1 / 3, 1 / 3])
    assert np.allclose(soft2[6:12], [1 / 3, 2 / 3, 1, 1, 2 / 3, 1 / 3])


def test_soft_contact_indicator_monotone_and_differentiable() -> None:
    hull = _unit_cube_hull()
    near = torch.tensor([[[0.52, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    far = torch.tensor([[[0.80, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    s_near = torch_soft_contact_indicator(near, hull, tau_m=0.04, beta_m=0.01)
    s_far = torch_soft_contact_indicator(far, hull, tau_m=0.04, beta_m=0.01)
    assert float(s_near) > 0.5 > float(s_far)
    s_near.sum().backward()
    assert near.grad is not None and float(near.grad.abs().sum()) > 0


def test_joint_contact_product_couples_both_robots() -> None:
    """d/dx1 of (1 - s1*s2) must depend on x2; the surrogate must not."""
    hull = _unit_cube_hull()
    x1 = torch.tensor([[[0.53, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)

    def grad_x1(x2_value: float, surrogate: bool) -> torch.Tensor:
        x2 = torch.tensor([[[x2_value, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
        s1 = torch_soft_contact_indicator(x1, hull, tau_m=0.04, beta_m=0.01)
        s2 = torch_soft_contact_indicator(x2, hull, tau_m=0.04, beta_m=0.01)
        product = (s1 + s2) / 2 if surrogate else s1 * s2
        loss = (1.0 - product).sum()
        (g,) = torch.autograd.grad(loss, x1)
        return g

    coupled_a = grad_x1(0.52, surrogate=False)
    coupled_b = grad_x1(0.70, surrogate=False)
    assert float((coupled_a - coupled_b).abs().max()) > 1e-6
    sep_a = grad_x1(0.52, surrogate=True)
    sep_b = grad_x1(0.70, surrogate=True)
    assert torch.allclose(sep_a, sep_b)


def test_new_coordination_fields() -> None:
    frames = 30
    expected = {
        1: np.r_[np.zeros(5, bool), np.ones(20, bool), np.zeros(5, bool)],
        2: np.r_[np.zeros(10, bool), np.ones(15, bool), np.zeros(5, bool)],
    }
    predicted = {
        1: np.r_[np.zeros(6, bool), np.ones(19, bool), np.zeros(5, bool)],
        2: np.r_[np.zeros(12, bool), np.ones(13, bool), np.zeros(5, bool)],
    }
    penetration = {1: np.zeros(frames), 2: np.zeros(frames)}
    penetration[2][15:17] = 1.0  # two invalid frames split the valid span
    metrics = coordination_metrics(_signals(expected, predicted, penetration=penetration))
    role = metrics["timing"]["role_transition"]
    assert role["robot1_onset_error_frames"] == 1
    assert role["robot2_onset_error_frames"] == 2
    assert role["grasp_order_preserved"] is True
    assert role["onset_lag_error_frames"] == 1
    valid = metrics["geometry_valid"]["tau_0.5cm"]
    assert valid["longest_valid_joint_run_frames"] == 8  # frames 17..24
    row = flatten_metrics(metrics)
    assert row["longest_valid_joint_run_frames_tau_0.5cm"] == 8
    assert row["grasp_order_preserved"] is True

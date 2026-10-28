"""Controlled, cumulative presets for the Join paper ablation.

These presets are deliberately separate from the historical CLI recipes.  They
all use the same independent-IK initialization and common optimizer/fidelity
settings so that J1--J4 differ only by the named method components.
"""

from __future__ import annotations

from dataclasses import asdict

from .coupled import CoupledRetargetConfig

PAPER_VARIANTS = ("J0", "J1", "J2", "J3", "J4")

PAPER_VARIANT_LABELS = {
    "J0": "Independent IK",
    "J1": "+ Object Contact",
    "J2": "+ Phase-Aware Coupling",
    "J3": "+ Geometry / Gait",
    "J4": "Full v41 Components",
}


def _common_coupled(iterations: int) -> dict[str, float | int | bool]:
    """Optimizer/fidelity settings held fixed across J1--J4."""
    return {
        "iterations": iterations,
        "learning_rate": 0.010,
        "pose_fidelity_weight": 25.0,
        "root_fidelity_weight": 35.0,
        "velocity_weight": 2.0,
        "acceleration_weight": 16.0,
        "jerk_weight": 40.0,
        "contact_weight": 90.0,
        "contact_transition_phase_weight": 1.0,
        "slip_boost_start_phase": 6,
        "slip_boost_factor": 1.0,
        "putdown_slip_boost_factor": 1.0,
        "inter_robot_weight": 0.0,
    }


def _phase_aware() -> dict[str, float | int | bool]:
    return {
        "contact_transition_phase_weight": 1.5,
        "inter_robot_weight": 8.0,
        "inter_robot_clearance": 0.12,
    }


def _geometry_gait() -> dict[str, float | int | bool]:
    """v41 geometry/gait stack, excluding its final approach/idle-foot terms."""
    return {
        "com_support_weight": 40.0,
        "foot_height_weight": 50.0,
        "foot_slip_weight": 400.0,
        "torso_upright_weight": 18.0,
        "com_center_weight": 20.0,
        "object_clearance_weight": 200.0,
        "object_clearance_margin_m": 0.04,
        "hand_clearance_margin_m": 0.0,
        "hand_surface_clearance_margin_m": 0.0,
        "hand_volume_clearance_weight": 1200.0,
        "hand_volume_clearance_margin_m": 0.008,
        "torso_volume_clearance_weight": 200.0,
        "torso_volume_clearance_margin_m": 0.0,
        "palm_offset_m": 0.052,
        "planted_root_slip_weight": 400.0,
        "foot_volume_clearance_weight": 250.0,
        "foot_volume_clearance_margin_m": 0.02,
        "foot_ground_weight": 400.0,
        "foot_slide_weight": 1200.0,
        "foot_slide_hard": True,
        "hard_plant_weight": 800.0,
        "plant_height_m": 0.02,
        "source_slow_xy_m_s": 0.10,
        "plant_require_contact": True,
        "plant_slow_only": True,
        "plant_gait_support": True,
        "object_hull_clearance_weight": 800.0,
        "hull_orient_only": False,
        "hull_mask_aabb": False,
        "hull_use_contact_mask": False,
        "hull_slack_m": 0.0,
        "hull_clearance_margin_m": 0.008,
        "hull_include_feet": False,
        "leg_volume_clearance_weight": 400.0,
        "leg_volume_clearance_margin_m": 0.0,
        "leg_hull_weight": 1200.0,
        "arm_hull_weight": 120.0,
        "swing_lift_weight": 80.0,
        "swing_height_m": 0.045,
        "swing_knee_weight": 0.0,
        "gait_knee_fidelity_weight": 90.0,
        "gait_knee_bias_rad": 0.08,
        "gait_knee_min_rad": 0.30,
        "gait_knee_max_rad": 1.10,
        "grasp_up_frac": 0.4,
        "plant_exclusive": False,
        "palmar_contact": True,
        "palm_facing_weight": 50.0,
        "finger_up_weight": 60.0,
        "plant_stop_grad": False,
        "stance_lock_weight": 1200.0,
        "putdown_dual_plant": True,
        "putdown_phase_min": 4,
        "putdown_phase_max": 4,
        "putdown_tail_frames": 18,
        "squat_stance_width_m": 0.24,
        "putdown_root_lock_weight": 0.0,
        "knee_symmetry_weight": 30.0,
        "squat_knee_min_weight": 60.0,
        "squat_knee_min_rad": 0.8,
        "both_soles_weight": 200.0,
        "stance_width_weight": 20.0,
        "leg_laterality_weight": 250.0,
        "leg_laterality_ankle_m": 0.02,
        "leg_laterality_knee_m": 0.04,
        "self_leg_clearance_weight": 400.0,
        "self_leg_clearance_m": 0.06,
        "hip_yaw_limit_weight": 150.0,
        "hip_yaw_max_rad": 0.80,
        "hip_roll_limit_weight": 80.0,
        "hip_roll_max_rad": 0.45,
        "swing_outward_weight": 250.0,
        "swing_outward_slack_m": 0.04,
        "root_speed_cap_weight": 0.0,
        "foot_speed_cap_weight": 0.0,
        "approach_ground_weight": 100.0,
        # J3 starts the anti-slip boost only at the detected approach phase.
        "slip_boost_start_phase": 2,
        "slip_boost_factor": 5.0,
        "putdown_slip_boost_factor": 10.0,
    }


def paper_join_config(variant: str, *, iterations: int = 600) -> CoupledRetargetConfig:
    """Return one controlled paper configuration for J1--J4."""
    key = variant.upper()
    if key not in PAPER_VARIANTS[1:]:
        raise ValueError(f"Coupled paper variant must be J1--J4, got {variant!r}")
    values = _common_coupled(iterations)
    if key in {"J2", "J3", "J4"}:
        values.update(_phase_aware())
    if key in {"J3", "J4"}:
        values.update(_geometry_gait())
    if key == "J4":
        values.update(
            {
                "source_approach_root_weight": 240.0,
                "swing_lift_min_root_m_s": 0.15,
                "slip_boost_start_phase": 1,
            }
        )
    return CoupledRetargetConfig(**values)


def paper_join_config_dict(variant: str, *, iterations: int = 600) -> dict[str, object]:
    """Serializable resolved configuration used for provenance and tests."""
    if variant.upper() == "J0":
        return {
            "variant": "J0",
            "label": PAPER_VARIANT_LABELS["J0"],
            "implementation": "IndependentRetargetConfig",
        }
    config = asdict(paper_join_config(variant, iterations=iterations))
    return {
        "variant": variant.upper(),
        "label": PAPER_VARIANT_LABELS[variant.upper()],
        "implementation": "CoupledRetargetConfig",
        "config": config,
    }

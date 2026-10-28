"""Open-loop joint-PD feasibility check for retargeted dual-G1 motion."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .controller import (
    com_balance_joint_deltas,
    contact_consistent_joint_torques,
    gravity_bias_zero_velocity,
    lift_roots_out_of_penetration,
    level_soles_by_root_tilt,
    plant_hovering_foot,
    project_com_onto_support,
    balance_joint_offsets,
    freeze_leg_targets,
    blend_arm_targets,
    blend_leg_targets,
    min_sole_center_z,
    planted_foot_sides,
    com_support_xy,
    apply_swing_sole_hold,
    apply_swing_com_hold,
    override_joint_targets,
    project_target_joints,
    imu_ankle_residuals,
    hip_pitch_target_bias,
    hip_roll_target_bias,
    ankle_roll_target_bias,
    is_leg_joint,
    pelvis_imu_yaw,
    peak_foot_xy_drift,
    max_foot_xy_speed,
    slip_hip_roll_bias,
    shorten_stance_split,
    subtree_com_wrench_torques,
    world_xy_to_yaw_frame,
    yaw_from_xmat,
)
from .coordinates import poses_core4d_to_robot


@dataclass(frozen=True)
class PDTrackingConfig:
    timestep: float = 0.002
    source_fps: float = 15.0
    leg_kp: float = 80.0
    waist_kp: float = 60.0
    arm_kp: float = 30.0
    wrist_kp: float = 12.0
    damping_ratio: float = 1.0
    fall_height: float = 0.45
    foot_contact_height: float = 0.08
    hold_first_frame_seconds: float = 0.0
    pin_base: bool = False
    gravity_compensation: bool = False
    com_feedback: bool = False
    com_kp: float = 2500.0
    com_kd: float = 200.0
    com_z_kp: float = 1500.0
    hip_com_gain: float = 0.4
    time_scale: float = 1.0
    com_mode: str = "jacobian"
    weld_feet: bool = False
    contact_compensation: bool = False
    contact_source: str = "virtual"
    control_mode: str = "torque"
    lift_feet: bool = False
    project_com: bool = False
    online_com: bool = False
    freeze_legs: bool = False
    project_target: bool = False
    freeze_waist: bool = False
    freeze_ankles: bool = False
    freeze_stance_foot: bool = False
    freeze_stance_latch: bool = True
    freeze_swing_sagittal: bool = False
    hold_swing_sole: bool = False
    hold_swing_com: bool = False
    stance_foot_margin: float = 0.008
    settle_seconds: float = 0.0
    com_integral: bool = False
    com_ki: float = 2.0
    residual_gains: tuple[float, float, float, float] | None = None
    residual_joints: str = "ankle"
    hip_pitch_bias: float = 0.0
    noslip_iterations: int | None = None
    contact_friction: float | None = None
    stagger_split: float | None = None
    cone: str | None = None
    slip_clutch: bool = False
    slip_speed: float = 0.05
    slip_delay: float = 0.4
    slip_robots: str = "both"
    slip_gain: float = 0.0
    solver_iterations: int | None = None
    impratio: float | None = None
    sole_condim: int | None = None
    r1_sole_condim: int | None = None
    slip_kp_scale: float = 1.0
    r2_hip_roll_bias: float = 0.0
    r2_com_roll_gain: float = 0.0
    r2_com_roll_kd: float = 0.0
    r2_com_ankle_gain: float = 0.0
    r2_com_ki: float = 0.0
    r2_com_pitch_gain: float = 0.0
    r2_com_y_ref: float = 0.0
    r2_com_x_ref: float = 0.0
    r2_late_y_ref: float | None = None
    r2_com_support: str = "midpoint"
    r2_com_ref_latch: bool = False
    r2_com_ref_delay: float = 0.0
    r2_arm_kp: float | None = None
    r2_waist_roll_bias: float = 0.0
    r2_waist_latch: bool = False
    r2_waist_latch_delay: float = 1.0
    r2_waist_roll_kp: float | None = None
    r2_level_soles: bool = False
    r2_project_com: bool = False
    ankle_kp: float | None = None
    r2_ankle_kp: float | None = None
    r2_leg_kp: float | None = None
    r2_ankle_pitch_bias: float = 0.0
    wrist_kp_override: float | None = None
    r2_pose_latch: bool = False
    r2_pose_latch_delay: float = 2.0
    r2_pose_latch_legs: bool = False
    r2_target_follow: float = 0.0
    r2_plant_foot: bool = False
    r2_late_delay: float = 4.0
    r2_late_ankle_pitch: float = 0.0
    r2_late_ankle_delay: float | None = None
    r2_late_ankle_both: bool = False
    r2_late_roll_gain: float | None = None
    r2_late_pitch_gain: float | None = None
    r2_late_pitch_delay: float | None = None
    r2_late_hip_pitch: float = 0.0
    r2_late2_hip_pitch: float = 0.0
    r2_late2_delay: float | None = None
    r2_late3_hip_pitch: float = 0.0
    r2_late3_delay: float | None = None
    r2_late_knee: float = 0.0
    r1_late_delay: float = 4.0
    r1_late_hip_pitch: float = 0.0
    r1_late2_hip_pitch: float = 0.0
    r1_late2_delay: float | None = None
    r1_late3_hip_pitch: float = 0.0
    r1_late3_delay: float | None = None
    r1_late_pitch_gain: float | None = None
    r1_late_pitch_delay: float | None = None
    arm_ease_seconds: float = 0.0
    arm_scale: float = 1.0
    leg_ease_seconds: float = 0.0
    leg_scale: float = 1.0
    freeze_object: bool = False
    r2_arm_com_comp: float = 0.0
    com_roll_both: bool = False


def _joint_kp(name: str, config: PDTrackingConfig) -> float:
    if any(token in name for token in ("hip", "knee", "ankle")):
        return config.leg_kp
    if "waist" in name:
        return config.waist_kp
    if "wrist" in name:
        return config.wrist_kp
    return config.arm_kp


def _load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: payload[name] for name in payload.files}


def _set_initial_state(
    model: object,
    data: object,
    trajectories: dict[int, dict[str, np.ndarray]],
) -> None:
    import mujoco

    for person, prefix in ((1, "r1_"), (2, "r2_")):
        trajectory = trajectories[person]
        root_joint = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            prefix + "floating_base_joint",
        )
        address = model.jnt_qposadr[root_joint]
        yaw = float(trajectory["root_yaw"][0])
        data.qpos[address : address + 3] = trajectory["root_translation"][0]
        data.qpos[address + 3 : address + 7] = [
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        ]
        for name, value in zip(
            trajectory["joint_names"],
            trajectory["joint_positions"][0],
            strict=True,
        ):
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                prefix + str(name),
            )
            data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def _sync_frame0_from_state(
    model: object,
    data: object,
    trajectories: dict[int, dict[str, np.ndarray]],
) -> None:
    import mujoco

    for person, prefix in ((1, "r1_"), (2, "r2_")):
        trajectory = trajectories[person]
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint"
        )
        address = int(model.jnt_qposadr[joint_id])
        trajectory["root_translation"][0] = np.array(
            data.qpos[address : address + 3], dtype=np.float64
        )
        for index, name in enumerate(trajectory["joint_names"]):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, prefix + str(name)
            )
            trajectory["joint_positions"][0, index] = data.qpos[
                int(model.jnt_qposadr[joint_id])
            ]


def simulate_pd_tracking(
    scene_xml: Path,
    object_poses_core4d: np.ndarray,
    robot1_path: Path,
    robot2_path: Path,
    config: PDTrackingConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Track joint trajectories with torque PD and unactuated floating bases."""
    if config is None:
        config = PDTrackingConfig()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    model.opt.timestep = config.timestep
    if config.control_mode == "position":
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if config.noslip_iterations is not None:
        model.opt.noslip_iterations = int(config.noslip_iterations)
    if config.cone == "pyramidal":
        model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    elif config.cone == "elliptic":
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    if config.solver_iterations is not None:
        model.opt.iterations = int(config.solver_iterations)
    if config.impratio is not None:
        model.opt.impratio = float(config.impratio)
    if config.control_mode == "position":
        for actuator_id in range(int(model.nu)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
            if "waist" in name:
                model.actuator_gainprm[actuator_id, 0] = float(config.waist_kp)
            if config.ankle_kp is not None and "ankle" in name:
                model.actuator_gainprm[actuator_id, 0] = float(config.ankle_kp)
            if (
                config.r2_leg_kp is not None
                and name.startswith("r2_")
                and any(token in name for token in ("hip", "knee", "ankle"))
            ):
                model.actuator_gainprm[actuator_id, 0] = float(config.r2_leg_kp)
            if (
                config.r2_ankle_kp is not None
                and name.startswith("r2_")
                and "ankle" in name
            ):
                model.actuator_gainprm[actuator_id, 0] = float(config.r2_ankle_kp)
            if config.wrist_kp_override is not None and "wrist" in name:
                model.actuator_gainprm[actuator_id, 0] = float(config.wrist_kp_override)
            if config.r2_waist_roll_kp is not None and name.startswith("r2_") and "waist_roll" in name:
                model.actuator_gainprm[actuator_id, 0] = float(config.r2_waist_roll_kp)
            if config.r2_arm_kp is not None and name.startswith("r2_"):
                if not any(token in name for token in ("hip", "knee", "ankle", "waist")):
                    kp = float(config.r2_arm_kp)
                    if "wrist" in name:
                        kp = max(kp * 0.4, 12.0)
                    model.actuator_gainprm[actuator_id, 0] = kp
    if config.sole_condim is not None:
        dim = int(config.sole_condim)
        for geom_id in range(int(model.ngeom)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name == "floor" or name.endswith("_sole"):
                model.geom_condim[geom_id] = dim
    if config.r1_sole_condim is not None:
        dim = int(config.r1_sole_condim)
        for geom_id in range(int(model.ngeom)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name == "floor" or (
                name.startswith("r1_") and name.endswith("_sole")
            ):
                model.geom_condim[geom_id] = dim
    if config.contact_friction is not None:
        mu = float(config.contact_friction)
        for geom_id in range(int(model.ngeom)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name == "floor" or name.endswith("_sole"):
                model.geom_friction[geom_id] = (mu, 0.4, 0.01)
    data = mujoco.MjData(model)
    trajectories = {
        1: _load_trajectory(robot1_path),
        2: _load_trajectory(robot2_path),
    }
    frame_count = min(
        len(object_poses_core4d),
        len(trajectories[1]["joint_positions"]),
        len(trajectories[2]["joint_positions"]),
    )
    object_poses = poses_core4d_to_robot(object_poses_core4d[:frame_count])
    _set_initial_state(model, data, trajectories)
    if config.lift_feet:
        lift_roots_out_of_penetration(model, data)
    if config.stagger_split is not None:
        for prefix in ("r1_", "r2_"):
            shorten_stance_split(
                model,
                data,
                prefix,
                target_split=float(config.stagger_split),
                axis="lateral",
                iterations=10,
            )
        lift_roots_out_of_penetration(model, data)
    if config.r2_level_soles:
        level_soles_by_root_tilt(model, data, "r2_")
        lift_roots_out_of_penetration(model, data)
    if config.r2_plant_foot:
        plant_hovering_foot(model, data, "r2_")
        lift_roots_out_of_penetration(model, data)
    if config.project_com:
        for prefix in ("r1_", "r2_"):
            project_com_onto_support(model, data, prefix)
        lift_roots_out_of_penetration(model, data)
    if config.r2_project_com:
        project_com_onto_support(model, data, "r2_")
        lift_roots_out_of_penetration(model, data)
    if (
        config.lift_feet
        or config.project_com
        or config.stagger_split is not None
        or config.r2_level_soles
        or config.r2_project_com
        or config.r2_plant_foot
    ):
        _sync_frame0_from_state(model, data, trajectories)
    if config.weld_feet:
        model.qpos0[:] = data.qpos
        mujoco.mj_forward(model, data)

    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shared_object")
    object_mocap_id = model.body_mocapid[object_body]
    free_joints: dict[int, tuple[int, int]] = {}
    for person, prefix in ((1, "r1_"), (2, "r2_")):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            prefix + "floating_base_joint",
        )
        free_joints[person] = (model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id])
    joint_metadata: dict[int, list[tuple[str, int, int, int, float, float]]] = {}
    effort_limits: dict[int, np.ndarray] = {}
    pelvis_ids: dict[int, int] = {}
    for person, prefix in ((1, "r1_"), (2, "r2_")):
        metadata = []
        person_effort_limits = []
        pelvis_ids[person] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis"
        )
        for name in trajectories[person]["joint_names"]:
            joint_name = str(name)
            full_name = prefix + joint_name
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full_name)
            kp = _joint_kp(joint_name, config)
            kd = 2 * config.damping_ratio * np.sqrt(kp)
            metadata.append(
                (
                    joint_name,
                    model.jnt_qposadr[joint_id],
                    model.jnt_dofadr[joint_id],
                    actuator_id,
                    kp,
                    kd,
                )
            )
            person_effort_limits.append(
                np.max(np.abs(model.jnt_actfrcrange[joint_id]))
            )
        joint_metadata[person] = metadata
        effort_limits[person] = np.asarray(person_effort_limits)

    original_gainprm = np.array(model.actuator_gainprm, copy=True)

    desired_com_z = {
        person: float(data.subtree_com[pelvis_ids[person]][2]) for person in (1, 2)
    }
    time_to_fall: dict[int, float | None] = {1: None, 2: None}
    contact_ncon_sum = {1: 0, 2: 0}
    contact_ncon_active = {1: 0, 2: 0}
    contact_steps = 0
    com_integrals = {1: np.zeros(2, dtype=np.float64), 2: np.zeros(2, dtype=np.float64)}
    slip_latch: dict[int, np.ndarray | None] = {1: None, 2: None}
    slip_hits = {1: 0, 2: 0}
    slip_latch_time: dict[int, float | None] = {1: None, 2: None}
    r2_waist_latch_values: dict[str, float] = {}
    r2_pose_latch_values: dict[str, float] = {}
    stance_sides: dict[int, tuple[str, ...] | None] = {1: None, 2: None}
    swing_sole_z: dict[int, dict[str, float]] = {1: {}, 2: {}}
    swing_com_xy: dict[int, np.ndarray | None] = {1: None, 2: None}
    com_refs: dict[int, np.ndarray | None] = {1: None, 2: None}
    prev_foot_xy: dict[int, np.ndarray | None] = {1: None, 2: None}

    records: dict[str, list[np.ndarray]] = {
        "time": [],
        "desired_joint_positions": [],
        "actual_joint_positions": [],
        "desired_root_translation": [],
        "actual_root_translation": [],
        "actual_root_quaternion": [],
        "pelvis_up_dot": [],
        "foot_positions": [],
        "palm_positions": [],
        "desired_contact_anchors": [],
        "contact_mask": [],
        "control": [],
        "actuator_torque": [],
        "com_support_error_xy": [],
    }
    next_record = 0
    motion_duration = (frame_count - 1) / config.source_fps * max(config.time_scale, 1e-6)
    if config.hold_first_frame_seconds > 0:
        duration = config.hold_first_frame_seconds
    else:
        duration = config.settle_seconds + motion_duration
    while data.time <= duration + config.timestep / 2:
        if config.hold_first_frame_seconds > 0:
            phase = 0.0
        elif data.time < config.settle_seconds:
            phase = 0.0
        else:
            phase = min(
                (data.time - config.settle_seconds)
                / max(config.time_scale, 1e-6)
                * config.source_fps,
                frame_count - 1,
            )
        lower_frame = int(np.floor(phase))
        upper_frame = min(lower_frame + 1, frame_count - 1)
        alpha = phase - lower_frame
        desired_joints = []
        desired_roots = []
        controls = []
        com_errors = []

        if config.pin_base:
            for person in (1, 2):
                trajectory = trajectories[person]
                qpos_adr, dof_adr = free_joints[person]
                root = (
                    (1 - alpha) * trajectory["root_translation"][lower_frame]
                    + alpha * trajectory["root_translation"][upper_frame]
                )
                yaw = (1 - alpha) * float(trajectory["root_yaw"][lower_frame]) + alpha * float(
                    trajectory["root_yaw"][upper_frame]
                )
                data.qpos[qpos_adr : qpos_adr + 3] = root
                data.qpos[qpos_adr + 3 : qpos_adr + 7] = [
                    np.cos(yaw / 2),
                    0,
                    0,
                    np.sin(yaw / 2),
                ]
                data.qvel[dof_adr : dof_adr + 6] = 0.0

        object_frame = int(np.clip(round(phase), 0, frame_count - 1))
        if config.freeze_object or config.hold_first_frame_seconds > 0:
            object_frame = 0
        data.mocap_pos[object_mocap_id] = object_poses[object_frame, :3, 3]
        object_quaternion = Rotation.from_matrix(
            object_poses[object_frame, :3, :3]
        ).as_quat()
        data.mocap_quat[object_mocap_id] = object_quaternion[[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
        use_torque = config.control_mode != "position"
        gravity_bias = (
            gravity_bias_zero_velocity(model, data)
            if use_torque and config.gravity_compensation
            else None
        )
        contact_tau = {1: None, 2: None}
        step_contacts = {1: 0, 2: 0}
        if use_torque and config.contact_compensation:
            bias = np.asarray(data.qfrc_bias, dtype=np.float64)
            for person, prefix in ((1, "r1_"), (2, "r2_")):
                torque_comp, n_contacts = contact_consistent_joint_torques(
                    model,
                    data,
                    prefix,
                    bias,
                    source=config.contact_source,
                )
                contact_tau[person] = torque_comp
                step_contacts[person] = n_contacts
            contact_steps += 1
            for person in (1, 2):
                contact_ncon_sum[person] += step_contacts[person]
                contact_ncon_active[person] += int(step_contacts[person] > 0)

        online_offsets = {1: {}, 2: {}}
        if not use_torque and config.online_com:
            for person, prefix in ((1, "r1_"), (2, "r2_")):
                online_offsets[person] = balance_joint_offsets(model, data, prefix)

        for person, prefix in ((1, "r1_"), (2, "r2_")):
            trajectory = trajectories[person]
            names = [str(name) for name in trajectory["joint_names"]]
            target = (
                (1 - alpha) * trajectory["joint_positions"][lower_frame]
                + alpha * trajectory["joint_positions"][upper_frame]
            )
            arm_weight = float(config.arm_scale)
            if config.arm_ease_seconds > 1e-12 and config.hold_first_frame_seconds <= 0:
                motion_time = float(data.time) - float(config.settle_seconds)
                arm_weight *= float(
                    np.clip(motion_time / config.arm_ease_seconds, 0.0, 1.0)
                )
            if abs(arm_weight - 1.0) > 1e-12:
                target = blend_arm_targets(
                    names,
                    target,
                    trajectory["joint_positions"][0],
                    arm_weight,
                )
            if not config.freeze_legs:
                leg_weight = float(config.leg_scale)
                if (
                    config.leg_ease_seconds > 1e-12
                    and config.hold_first_frame_seconds <= 0
                ):
                    motion_time = float(data.time) - float(config.settle_seconds)
                    leg_weight *= float(
                        np.clip(motion_time / config.leg_ease_seconds, 0.0, 1.0)
                    )
                if abs(leg_weight - 1.0) > 1e-12:
                    target = blend_leg_targets(
                        names,
                        target,
                        trajectory["joint_positions"][0],
                        leg_weight,
                    )
            freeze_sides: tuple[str, ...] | None = None
            if config.freeze_stance_foot and not config.freeze_legs:
                left_z = min_sole_center_z(model, data, prefix, "left")
                right_z = min_sole_center_z(model, data, prefix, "right")
                sides = planted_foot_sides(
                    left_z, right_z, config.stance_foot_margin
                )
                if config.freeze_stance_latch:
                    if stance_sides[person] is None:
                        stance_sides[person] = sides
                        swing_sole_z[person] = {
                            side: min_sole_center_z(model, data, prefix, side)
                            for side in ("left", "right")
                            if side not in sides
                        }
                        swing_com_xy[person] = np.array(
                            data.subtree_com[pelvis_ids[person]][:2],
                            dtype=np.float64,
                        )
                    freeze_sides = stance_sides[person]
                else:
                    freeze_sides = sides
                    if not swing_sole_z[person]:
                        swing_sole_z[person] = {
                            side: min_sole_center_z(model, data, prefix, side)
                            for side in ("left", "right")
                            if side not in sides
                        }
                    if swing_com_xy[person] is None:
                        swing_com_xy[person] = np.array(
                            data.subtree_com[pelvis_ids[person]][:2],
                            dtype=np.float64,
                        )
            if (
                config.freeze_legs
                or config.freeze_waist
                or config.freeze_ankles
                or freeze_sides
            ):
                target = freeze_leg_targets(
                    names,
                    target,
                    trajectory["joint_positions"][0],
                    freeze_legs=config.freeze_legs,
                    freeze_waist=config.freeze_waist,
                    freeze_ankles=config.freeze_ankles,
                    freeze_sides=freeze_sides,
                    freeze_swing_sagittal=config.freeze_swing_sagittal,
                )
            if (
                config.hold_swing_com
                and freeze_sides
                and not config.freeze_legs
            ):
                swing_sides = tuple(
                    side for side in ("left", "right") if side not in freeze_sides
                )
                qpos_by_name = {
                    name: int(qpos) for name, qpos, *_ in joint_metadata[person]
                }
                target = apply_swing_com_hold(
                    model,
                    data,
                    prefix,
                    names,
                    target,
                    qpos_by_name,
                    swing_sides,
                    swing_com_xy[person],
                    swing_sole_z[person],
                )
            elif (
                config.hold_swing_sole
                and freeze_sides
                and not config.freeze_legs
            ):
                swing_sides = tuple(
                    side for side in ("left", "right") if side not in freeze_sides
                )
                qpos_by_name = {
                    name: int(qpos) for name, qpos, *_ in joint_metadata[person]
                }
                target = apply_swing_sole_hold(
                    model,
                    data,
                    prefix,
                    names,
                    target,
                    qpos_by_name,
                    swing_sides,
                    swing_sole_z[person],
                )
            if person == 2 and abs(config.r2_waist_roll_bias) > 1e-12:
                waist_bias: dict[str, float] = {}
                for index, name in enumerate(names):
                    if str(name) == "waist_roll_joint":
                        waist_bias[str(name)] = float(
                            target[index] + config.r2_waist_roll_bias
                        )
                target = override_joint_targets(names, target, waist_bias)
            if person == 2 and abs(config.r2_ankle_pitch_bias) > 1e-12:
                pitch_bias: dict[str, float] = {}
                for index, name in enumerate(names):
                    if str(name) == "left_ankle_pitch_joint":
                        pitch_bias[str(name)] = float(
                            target[index] + config.r2_ankle_pitch_bias
                        )
                target = override_joint_targets(names, target, pitch_bias)
            if person == 2 and config.r2_waist_latch:
                if (
                    not r2_waist_latch_values
                    and float(data.time) > config.r2_waist_latch_delay
                ):
                    for name, qpos, *_ in joint_metadata[person]:
                        if "waist_roll" in name or "waist_pitch" in name:
                            r2_waist_latch_values[name] = float(data.qpos[qpos])
                if r2_waist_latch_values:
                    target = override_joint_targets(
                        names, target, r2_waist_latch_values
                    )
            if person == 2 and config.r2_pose_latch:
                if (
                    not r2_pose_latch_values
                    and float(data.time) > config.r2_pose_latch_delay
                ):
                    for name, qpos, *_ in joint_metadata[person]:
                        if config.r2_pose_latch_legs and not (
                            is_leg_joint(name) or "waist" in name
                        ):
                            continue
                        r2_pose_latch_values[name] = float(data.qpos[qpos])
                if r2_pose_latch_values:
                    target = override_joint_targets(
                        names, target, r2_pose_latch_values
                    )
            if person == 2 and config.r2_target_follow > 1e-12:
                blend = float(np.clip(config.r2_target_follow, 0.0, 1.0))
                actual_by_name = {
                    name: float(data.qpos[qpos])
                    for name, qpos, *_ in joint_metadata[person]
                }
                followed = np.asarray(target, dtype=np.float64).copy()
                for index, name in enumerate(names):
                    if name not in actual_by_name:
                        continue
                    if not (is_leg_joint(name) or "waist" in name):
                        continue
                    followed[index] = (1.0 - blend) * followed[index] + blend * actual_by_name[name]
                target = followed
            left_foot = data.xpos[
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, prefix + "left_ankle_roll_link"
                )
            ][:2]
            right_foot = data.xpos[
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, prefix + "right_ankle_roll_link"
                )
            ][:2]
            if config.slip_clutch and (
                config.slip_robots == "both" or str(person) == str(config.slip_robots)
            ):
                feet_xy = np.vstack((left_foot, right_foot))
                if (
                    prev_foot_xy[person] is not None
                    and slip_latch[person] is None
                    and float(data.time) > config.slip_delay
                ):
                    speed = max_foot_xy_speed(
                        feet_xy, prev_foot_xy[person], config.timestep
                    )
                    if speed > config.slip_speed:
                        slip_hits[person] += 1
                    else:
                        slip_hits[person] = 0
                    if slip_hits[person] >= 3:
                        slip_latch[person] = np.asarray(
                            [data.qpos[item[1]] for item in joint_metadata[person]],
                            dtype=np.float64,
                        )
                        slip_latch_time[person] = float(data.time)
                if slip_latch[person] is not None:
                    target = freeze_leg_targets(names, target, slip_latch[person])
            if abs(config.slip_kp_scale - 1.0) > 1e-12:
                scale = 1.0
                if (
                    prev_foot_xy[person] is not None
                    and float(data.time) > config.slip_delay
                ):
                    speed = max_foot_xy_speed(
                        np.vstack((left_foot, right_foot)),
                        prev_foot_xy[person],
                        config.timestep,
                    )
                    if speed > config.slip_speed:
                        scale = float(config.slip_kp_scale)
                for name, _, _, actuator, _, _ in joint_metadata[person]:
                    if is_leg_joint(name):
                        model.actuator_gainprm[actuator, 0] = (
                            original_gainprm[actuator, 0] * scale
                        )
            if not use_torque and config.project_target:
                target = project_target_joints(model, data, prefix, names, target)
            desired_joints.append(target)
            desired_roots.append(
                (1 - alpha) * trajectory["root_translation"][lower_frame]
                + alpha * trajectory["root_translation"][upper_frame]
            )
            extras: dict[str, float] = {}
            com_tau = None
            pelvis = pelvis_ids[person]
            yaw = yaw_from_xmat(data.xmat[pelvis])
            extras.update(
                hip_pitch_target_bias(
                    [item[0] for item in joint_metadata[person]],
                    config.hip_pitch_bias,
                )
            )
            if person == 2:
                for key, value in hip_roll_target_bias(
                    [item[0] for item in joint_metadata[person]],
                    config.r2_hip_roll_bias,
                ).items():
                    extras[key] = extras.get(key, 0.0) + value
            if abs(config.slip_gain) > 1e-12 and prev_foot_xy[person] is not None:
                left_dir = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
                dt = max(config.timestep, 1e-9)
                left_speed = float(
                    np.dot((np.asarray(left_foot) - prev_foot_xy[person][0]) / dt, left_dir)
                )
                right_speed = float(
                    np.dot((np.asarray(right_foot) - prev_foot_xy[person][1]) / dt, left_dir)
                )
                extras.update(
                    slip_hip_roll_bias(
                        [item[0] for item in joint_metadata[person]],
                        left_speed,
                        right_speed,
                        config.slip_gain,
                    )
                )
            support = com_support_xy(
                left_foot,
                right_foot,
                mode=(config.r2_com_support if person == 2 else "midpoint"),
                planted_sides=stance_sides[person] if person == 2 else None,
                left_z=(
                    min_sole_center_z(model, data, prefix, "left")
                    if person == 2 and config.r2_com_support == "planted"
                    else None
                ),
                right_z=(
                    min_sole_center_z(model, data, prefix, "right")
                    if person == 2 and config.r2_com_support == "planted"
                    else None
                ),
                margin=config.stance_foot_margin,
            )
            com_xy = data.subtree_com[pelvis][:2]
            error_world = com_xy - support
            error_yaw = world_xy_to_yaw_frame(error_world, yaw)
            velocity_yaw = world_xy_to_yaw_frame(data.subtree_linvel[pelvis][:2], yaw)
            if (
                person == 2
                and config.r2_com_ref_latch
                and com_refs[person] is None
                and float(data.time) >= config.r2_com_ref_delay
            ):
                com_refs[person] = np.asarray(error_yaw, dtype=np.float64).copy()
            x_ref = config.r2_com_x_ref
            y_ref = config.r2_com_y_ref
            if person == 2 and com_refs[person] is not None:
                x_ref = float(com_refs[person][0])
                y_ref = float(com_refs[person][1])
            if (
                person == 2
                and config.r2_late_y_ref is not None
                and float(data.time) > config.r2_late_delay
            ):
                y_ref = float(config.r2_late_y_ref)
            com_errors.append(error_yaw)
            if (
                (person == 2 or config.com_roll_both)
                and (
                abs(config.r2_com_roll_gain) > 1e-12
                or abs(config.r2_com_roll_kd) > 1e-12
                or abs(config.r2_com_ankle_gain) > 1e-12
                or abs(config.r2_com_ki) > 1e-12
                or abs(config.r2_com_pitch_gain) > 1e-12
            )
            ):
                names_now = [item[0] for item in joint_metadata[person]]
                roll_gain = config.r2_com_roll_gain
                if (
                    config.r2_late_roll_gain is not None
                    and float(data.time) > config.r2_late_delay
                ):
                    roll_gain = float(config.r2_late_roll_gain)
                hip_cmd = float(
                    np.clip(
                        -roll_gain * (error_yaw[1] - y_ref)
                        - config.r2_com_roll_kd * velocity_yaw[1]
                        - config.r2_com_ki * com_integrals[person][1],
                        -0.08,
                        0.08,
                    )
                )
                ankle_cmd = float(
                    np.clip(
                        -config.r2_com_ankle_gain * error_yaw[1]
                        - config.r2_com_roll_kd * velocity_yaw[1]
                        - config.r2_com_ki * com_integrals[person][1],
                        -0.08,
                        0.08,
                    )
                )
                for key, value in hip_roll_target_bias(names_now, hip_cmd).items():
                    extras[key] = extras.get(key, 0.0) + value
                for key, value in ankle_roll_target_bias(names_now, ankle_cmd).items():
                    extras[key] = extras.get(key, 0.0) + value
                pitch_gain = config.r2_com_pitch_gain
                pitch_delay = (
                    config.r2_late_delay
                    if config.r2_late_pitch_delay is None
                    else float(config.r2_late_pitch_delay)
                )
                if (
                    config.r2_late_pitch_gain is not None
                    and float(data.time) > pitch_delay
                ):
                    pitch_gain = float(config.r2_late_pitch_gain)
                pitch_cmd = float(
                    np.clip(
                        -pitch_gain * (error_yaw[0] - x_ref)
                        - config.r2_com_roll_kd * velocity_yaw[0],
                        -0.08,
                        0.08,
                    )
                )
                for key, value in hip_pitch_target_bias(names_now, pitch_cmd).items():
                    extras[key] = extras.get(key, 0.0) + value
            if (
                person == 2
                and abs(config.r2_late_ankle_pitch) > 1e-12
                and float(data.time)
                > (
                    config.r2_late_delay
                    if config.r2_late_ankle_delay is None
                    else float(config.r2_late_ankle_delay)
                )
            ):
                extras["left_ankle_pitch_joint"] = extras.get(
                    "left_ankle_pitch_joint", 0.0
                ) + float(config.r2_late_ankle_pitch)
                if config.r2_late_ankle_both:
                    extras["right_ankle_pitch_joint"] = extras.get(
                        "right_ankle_pitch_joint", 0.0
                    ) + float(config.r2_late_ankle_pitch)
            if (
                person == 2
                and abs(config.r2_late_hip_pitch) > 1e-12
                and float(data.time) > config.r2_late_delay
            ):
                for key, value in hip_pitch_target_bias(
                    names, float(config.r2_late_hip_pitch)
                ).items():
                    extras[key] = extras.get(key, 0.0) + value
            if (
                person == 2
                and abs(config.r2_late2_hip_pitch) > 1e-12
                and config.r2_late2_delay is not None
                and float(data.time) > float(config.r2_late2_delay)
            ):
                for key, value in hip_pitch_target_bias(
                    names, float(config.r2_late2_hip_pitch)
                ).items():
                    extras[key] = extras.get(key, 0.0) + value
            if (
                person == 2
                and abs(config.r2_late3_hip_pitch) > 1e-12
                and config.r2_late3_delay is not None
                and float(data.time) > float(config.r2_late3_delay)
            ):
                for key, value in hip_pitch_target_bias(
                    names, float(config.r2_late3_hip_pitch)
                ).items():
                    extras[key] = extras.get(key, 0.0) + value
            if (
                person == 1
                and config.r1_late_pitch_gain is not None
                and float(data.time)
                > (
                    config.r1_late_delay
                    if config.r1_late_pitch_delay is None
                    else float(config.r1_late_pitch_delay)
                )
            ):
                pitch_cmd = float(
                    np.clip(
                        -float(config.r1_late_pitch_gain) * error_yaw[0]
                        - config.r2_com_roll_kd * velocity_yaw[0],
                        -0.08,
                        0.08,
                    )
                )
                for key, value in hip_pitch_target_bias(names, pitch_cmd).items():
                    extras[key] = extras.get(key, 0.0) + value
            if (
                person == 1
                and abs(config.r1_late_hip_pitch) > 1e-12
                and float(data.time) > config.r1_late_delay
            ):
                for key, value in hip_pitch_target_bias(
                    names, float(config.r1_late_hip_pitch)
                ).items():
                    extras[key] = extras.get(key, 0.0) + value
            if (
                person == 1
                and abs(config.r1_late2_hip_pitch) > 1e-12
                and config.r1_late2_delay is not None
                and float(data.time) > float(config.r1_late2_delay)
            ):
                for key, value in hip_pitch_target_bias(
                    names, float(config.r1_late2_hip_pitch)
                ).items():
                    extras[key] = extras.get(key, 0.0) + value
            if (
                person == 1
                and abs(config.r1_late3_hip_pitch) > 1e-12
                and config.r1_late3_delay is not None
                and float(data.time) > float(config.r1_late3_delay)
            ):
                for key, value in hip_pitch_target_bias(
                    names, float(config.r1_late3_hip_pitch)
                ).items():
                    extras[key] = extras.get(key, 0.0) + value
            if (
                person == 2
                and abs(config.r2_late_knee) > 1e-12
                and float(data.time) > config.r2_late_delay
            ):
                extras["left_knee_joint"] = extras.get(
                    "left_knee_joint", 0.0
                ) + float(config.r2_late_knee)
            if person == 2 and abs(config.r2_arm_com_comp) > 1e-12:
                frame0 = trajectory["joint_positions"][0]
                shoulder = 0.0
                for index, name in enumerate(names):
                    if "shoulder_pitch" in str(name):
                        shoulder += float(target[index] - frame0[index])
                for key, value in hip_pitch_target_bias(
                    names,
                    float(np.clip(config.r2_arm_com_comp * shoulder, -0.08, 0.08)),
                ).items():
                    extras[key] = extras.get(key, 0.0) + value
            if config.com_feedback and (
                config.com_mode == "heuristic" or not use_torque
            ):
                feedback = com_balance_joint_deltas(
                    [item[0] for item in joint_metadata[person]],
                    error_yaw,
                    velocity_yaw,
                    config.com_kp,
                    config.com_kd,
                    config.hip_com_gain,
                )
                if not use_torque:
                    feedback = {
                        key: float(np.clip(value, -0.4, 0.4))
                        for key, value in feedback.items()
                    }
                for key, value in feedback.items():
                    extras[key] = extras.get(key, 0.0) + value
            elif use_torque and config.com_feedback:
                com_tau = subtree_com_wrench_torques(
                    model,
                    data,
                    pelvis,
                    support,
                    desired_com_z[person],
                    config.com_kp,
                    config.com_kd,
                    config.com_z_kp,
                )
            if config.com_integral or (
                person == 2 and abs(config.r2_com_ki) > 1e-12
            ):
                com_integrals[person] += error_yaw * config.timestep
                com_integrals[person] *= max(0.0, 1.0 - 0.2 * config.timestep)
                com_integrals[person] = np.clip(com_integrals[person], -0.35, 0.35)
            if config.com_integral:
                integral_deltas = com_balance_joint_deltas(
                    [item[0] for item in joint_metadata[person]],
                    com_integrals[person],
                    np.zeros(2),
                    config.com_ki,
                    0.0,
                    0.25,
                )
                for key, value in integral_deltas.items():
                    extras[key] = extras.get(key, 0.0) + float(np.clip(value, -0.18, 0.18))
            if config.residual_gains is not None:
                tilt, rates = pelvis_imu_yaw(data, pelvis, yaw)
                imu_extras = imu_ankle_residuals(
                    [item[0] for item in joint_metadata[person]],
                    tilt,
                    rates,
                    config.residual_gains,
                    apply_to=config.residual_joints,
                )
                for key, value in imu_extras.items():
                    extras[key] = extras.get(key, 0.0) + value
            person_controls = []
            for target_value, (name, qpos, qvel, actuator, kp, kd) in zip(
                target,
                joint_metadata[person],
                strict=True,
            ):
                if not use_torque:
                    data.ctrl[actuator] = (
                        float(target_value)
                        + extras.get(name, 0.0)
                        + online_offsets[person].get(name, 0.0)
                    )
                    person_controls.append(float(data.ctrl[actuator]))
                    continue
                torque = kp * (target_value - data.qpos[qpos]) - kd * data.qvel[qvel]
                if gravity_bias is not None:
                    torque += float(gravity_bias[qvel])
                if contact_tau[person] is not None:
                    torque += float(contact_tau[person][qvel])
                torque += extras.get(name, 0.0)
                if com_tau is not None:
                    torque += float(com_tau[qvel])
                data.ctrl[actuator] = torque
                person_controls.append(torque)
            controls.append(person_controls)
            prev_foot_xy[person] = np.vstack((left_foot, right_foot)).astype(np.float64)

        mujoco.mj_step(model, data)
        for person in (1, 2):
            if time_to_fall[person] is not None:
                continue
            height = float(data.xpos[pelvis_ids[person]][2])
            up_dot = float(data.xmat[pelvis_ids[person]].reshape(3, 3)[2, 2])
            if height < config.fall_height or up_dot < 0.5:
                time_to_fall[person] = float(data.time)

        if data.time + config.timestep / 2 >= next_record / config.source_fps:
            actual_joints = []
            actual_roots = []
            actual_root_quaternions = []
            pelvis_up = []
            feet = []
            palms = []
            actuator_torques = []
            for person, prefix in ((1, "r1_"), (2, "r2_")):
                actual_joints.append(
                    [data.qpos[item[1]] for item in joint_metadata[person]]
                )
                pelvis_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    prefix + "pelvis",
                )
                actual_roots.append(data.xpos[pelvis_id].copy())
                actual_root_quaternions.append(data.xquat[pelvis_id].copy())
                pelvis_up.append(float(data.xmat[pelvis_id].reshape(3, 3)[2, 2]))
                actuator_torques.append(
                    [data.qfrc_actuator[item[2]] for item in joint_metadata[person]]
                )
                feet.append(
                    [
                        data.xpos[
                            mujoco.mj_name2id(
                                model,
                                mujoco.mjtObj.mjOBJ_BODY,
                                prefix + side + "_ankle_roll_link",
                            )
                        ].copy()
                        for side in ("left", "right")
                    ]
                )
                palms.append(
                    [
                        data.xpos[
                            mujoco.mj_name2id(
                                model,
                                mujoco.mjtObj.mjOBJ_BODY,
                                prefix + side + "_wrist_yaw_link",
                            )
                        ].copy()
                        for side in ("left", "right")
                    ]
                )
            desired_anchors = [
                (1 - alpha) * trajectories[person]["contact_anchors"][lower_frame]
                + alpha * trajectories[person]["contact_anchors"][upper_frame]
                for person in (1, 2)
            ]
            contact_masks = [
                trajectories[person]["contact_mask"][round(phase)]
                for person in (1, 2)
            ]
            records["time"].append(np.asarray(data.time))
            records["desired_joint_positions"].append(np.asarray(desired_joints))
            records["actual_joint_positions"].append(np.asarray(actual_joints))
            records["desired_root_translation"].append(np.asarray(desired_roots))
            records["actual_root_translation"].append(np.asarray(actual_roots))
            records["actual_root_quaternion"].append(
                np.asarray(actual_root_quaternions)
            )
            records["pelvis_up_dot"].append(np.asarray(pelvis_up))
            records["foot_positions"].append(np.asarray(feet))
            records["palm_positions"].append(np.asarray(palms))
            records["desired_contact_anchors"].append(np.asarray(desired_anchors))
            records["contact_mask"].append(np.asarray(contact_masks))
            records["control"].append(np.asarray(controls))
            records["actuator_torque"].append(np.asarray(actuator_torques))
            records["com_support_error_xy"].append(np.asarray(com_errors))
            next_record += 1
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qacc).all():
            break

    arrays = {name: np.asarray(values) for name, values in records.items()}
    joint_error = arrays["actual_joint_positions"] - arrays["desired_joint_positions"]
    root_error = arrays["actual_root_translation"] - arrays["desired_root_translation"]
    foot_positions = arrays["foot_positions"]
    foot_velocity = np.diff(foot_positions[..., :2], axis=0) * config.source_fps
    foot_contact = foot_positions[:-1, ..., 2] < config.foot_contact_height
    foot_drift = peak_foot_xy_drift(foot_positions)
    metrics: dict[str, object] = {
        "config": asdict(config),
        "simulated_frames": len(arrays["time"]),
        "expected_frames": round(duration * config.source_fps) + 1,
        "numerically_stable": bool(np.isfinite(arrays["actual_joint_positions"]).all()),
        "robots": {},
    }
    robot_metrics: dict[str, object] = {}
    for index, person in enumerate((1, 2)):
        in_contact = foot_contact[:, index]
        slip_values = np.linalg.norm(foot_velocity[:, index], axis=-1)[in_contact]
        root_height = arrays["actual_root_translation"][:, index, 2]
        up_dot = arrays["pelvis_up_dot"][:, index]
        actual_torque = arrays["actuator_torque"][:, index]
        contact_mask = arrays["contact_mask"][:, index].astype(bool)
        palm_distance = np.linalg.norm(
            arrays["palm_positions"][:, index]
            - arrays["desired_contact_anchors"][:, index],
            axis=-1,
        )
        expected_contact_distances = palm_distance[contact_mask]
        robot_metrics[str(person)] = {
            "joint_tracking_rmse_rad": float(np.sqrt(np.mean(joint_error[:, index] ** 2))),
            "root_tracking_rmse_m": float(np.sqrt(np.mean(root_error[:, index] ** 2))),
            "minimum_pelvis_height_m": float(np.min(root_height)),
            "fallen_frame_ratio": float(
                np.mean((root_height < config.fall_height) | (up_dot < 0.5))
            ),
            "foot_slip_mean_m_s": float(np.mean(slip_values)) if len(slip_values) else None,
            "foot_slip_max_m_s": float(np.max(slip_values)) if len(slip_values) else None,
            "raw_pd_command_rms": float(
                np.sqrt(np.mean(arrays["control"][:, index] ** 2))
            ),
            "actuator_torque_rms_nm": float(
                np.sqrt(np.mean(actual_torque**2))
            ),
            "torque_saturation_ratio": float(
                np.mean(np.abs(actual_torque) >= 0.99 * effort_limits[person][None])
            ),
            "contact_anchor_error_mean_m": (
                float(np.mean(expected_contact_distances))
                if len(expected_contact_distances)
                else None
            ),
            "contact_retention_at_8cm": (
                float(np.mean(expected_contact_distances < 0.08))
                if len(expected_contact_distances)
                else None
            ),
            "com_support_error_mean_m": float(
                np.mean(np.linalg.norm(arrays["com_support_error_xy"][:, index], axis=-1))
            ),
            "peak_foot_xy_drift_m": [float(value) for value in foot_drift[index]],
            "slip_clutch_latch_s": slip_latch_time[person],
            "time_to_fall_s": time_to_fall[person],
            "sim_contact_mean": (
                float(contact_ncon_sum[person] / contact_steps) if contact_steps else 0.0
            ),
            "sim_contact_active_ratio": (
                float(contact_ncon_active[person] / contact_steps) if contact_steps else 0.0
            ),
        }
    metrics["robots"] = robot_metrics
    return arrays, metrics


def save_pd_tracking(
    arrays: dict[str, np.ndarray],
    metrics: dict[str, object],
    output_npz: Path,
    output_json: Path,
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def inverse_dynamics_feasibility(
    scene_xml: Path,
    robot1_path: Path,
    robot2_path: Path,
    *,
    source_fps: float = 15.0,
    upsample: int = 1,
    acceleration: str = "full",
) -> dict[str, object]:
    """Unconstrained ID: fraction of joint-frames whose required torque exceeds limits."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    trajectories = {1: _load_trajectory(robot1_path), 2: _load_trajectory(robot2_path)}
    frame_count = min(
        len(trajectories[1]["joint_positions"]),
        len(trajectories[2]["joint_positions"]),
    )
    upsample = max(int(upsample), 1)
    dense_count = (frame_count - 1) * upsample + 1
    dt = 1.0 / (source_fps * upsample)
    report: dict[str, object] = {
        "source_fps": source_fps,
        "upsample": upsample,
        "id_fps": source_fps * upsample,
        "acceleration": acceleration,
        "frames": dense_count,
        "robots": {},
    }
    source_times = np.arange(frame_count, dtype=np.float64)
    dense_times = np.linspace(0.0, frame_count - 1, dense_count)

    def _resample(values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            return np.interp(dense_times, source_times, values)
        return np.stack(
            [np.interp(dense_times, source_times, values[:, index]) for index in range(values.shape[1])],
            axis=1,
        )

    for person, prefix in ((1, "r1_"), (2, "r2_")):
        trajectory = trajectories[person]
        joints = _resample(trajectory["joint_positions"][:frame_count])
        roots = _resample(trajectory["root_translation"][:frame_count])
        yaws = _resample(trajectory["root_yaw"][:frame_count])
        vel = np.gradient(joints, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        root_vel = np.gradient(roots, dt, axis=0)
        root_acc = np.gradient(root_vel, dt, axis=0)
        yaw_vel = np.gradient(yaws.astype(np.float64), dt)
        yaw_acc = np.gradient(yaw_vel, dt)
        free_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint"
        )
        qpos_adr = model.jnt_qposadr[free_id]
        dof_adr = model.jnt_dofadr[free_id]
        names = [str(name) for name in trajectory["joint_names"]]
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
            for name in names
        ]
        limits = np.asarray([np.max(np.abs(model.jnt_actfrcrange[jid])) for jid in joint_ids])
        required = np.zeros_like(joints)
        for frame in range(dense_count):
            yaw = float(yaws[frame])
            data.qpos[qpos_adr : qpos_adr + 3] = roots[frame]
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            if acceleration == "none":
                data.qvel[:] = 0.0
                data.qacc[:] = 0.0
            elif acceleration == "joints":
                data.qvel[dof_adr : dof_adr + 6] = 0.0
                data.qacc[dof_adr : dof_adr + 6] = 0.0
            else:
                data.qvel[dof_adr : dof_adr + 3] = root_vel[frame]
                data.qvel[dof_adr + 3 : dof_adr + 6] = [0.0, 0.0, yaw_vel[frame]]
                data.qacc[dof_adr : dof_adr + 3] = root_acc[frame]
                data.qacc[dof_adr + 3 : dof_adr + 6] = [0.0, 0.0, yaw_acc[frame]]
            for value, jid in zip(joints[frame], joint_ids, strict=True):
                data.qpos[model.jnt_qposadr[jid]] = value
            if acceleration != "none":
                for value, jid in zip(vel[frame], joint_ids, strict=True):
                    data.qvel[model.jnt_dofadr[jid]] = value
                for value, jid in zip(acc[frame], joint_ids, strict=True):
                    data.qacc[model.jnt_dofadr[jid]] = value
            mujoco.mj_inverse(model, data)
            required[frame] = [data.qfrc_inverse[model.jnt_dofadr[jid]] for jid in joint_ids]
        excess = np.abs(required) - limits[None]
        report["robots"][str(person)] = {
            "torque_limit_violation_ratio": float(np.mean(excess > 0)),
            "peak_required_torque_nm": float(np.max(np.abs(required))),
            "p95_required_torque_nm": float(np.percentile(np.abs(required), 95)),
            "median_required_torque_nm": float(np.median(np.abs(required))),
            "mean_required_torque_nm": float(np.mean(np.abs(required))),
            "worst_joint": names[int(np.argmax(np.max(np.abs(required), axis=0)))],
        }
    return report

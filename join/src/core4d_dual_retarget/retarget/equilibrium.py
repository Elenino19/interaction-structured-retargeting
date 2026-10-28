"""Quasi-static equilibrium post-pass: adjust root and legs so feet can push, not pull."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ..contact_id import (
    _foot_force_jacobian,
    clip_unilateral_normals,
    solve_foot_forces,
)
from ..controller import (
    hold_com_xy,
    is_leg_joint,
    joint_side,
    min_sole_center_z,
    planted_foot_sides,
)
from ..pd_tracking import _load_trajectory
from .smooth import recompute_keypoints

POSTURE_TOKENS = (
    "hip_pitch",
    "hip_roll",
    "hip_yaw",
    "knee",
    "ankle_pitch",
    "ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
)


@dataclass(frozen=True)
class EquilibriumConfig:
    max_iterations: int = 80
    root_xy_bound_m: float = 0.12
    root_z_bound_m: float = 0.08
    joint_bound_rad: float = 0.35
    base_weight: float = 1.0
    tensile_weight: float = 8.0
    torque_weight: float = 1e-4
    posture_weight: float = 4.0
    palm_weight: float = 25.0
    foot_height_weight: float = 40.0
    target_foot_z: float = 0.055
    temporal_weight: float = 0.0


# Clip-level pass: fewer iterations, stay close to the kinematic dataset,
# and penalize frame-to-frame jumps. Do not use this config for one-frame CLI.
CLIP_EQUILIBRIUM_CONFIG = EquilibriumConfig(
    max_iterations=25,
    root_xy_bound_m=0.02,
    root_z_bound_m=0.01,
    joint_bound_rad=0.18,
    tensile_weight=12.0,
    posture_weight=8.0,
    palm_weight=80.0,
    foot_height_weight=80.0,
    target_foot_z=0.055,
    temporal_weight=24.0,
)


def _posture_indices(joint_names: list[str]) -> list[int]:
    indices = []
    for index, name in enumerate(joint_names):
        if any(token in name for token in POSTURE_TOKENS):
            indices.append(index)
    if not indices:
        raise ValueError("No hip/knee/ankle/waist joints found")
    return indices


def _pack(
    roots: dict[int, np.ndarray],
    joints: dict[int, np.ndarray],
    posture: dict[int, list[int]],
) -> np.ndarray:
    chunks = []
    for person in (1, 2):
        chunks.append(roots[person])
        chunks.append(joints[person][posture[person]])
    return np.concatenate(chunks)


def _unpack(
    vector: np.ndarray,
    roots0: dict[int, np.ndarray],
    joints0: dict[int, np.ndarray],
    posture: dict[int, list[int]],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    roots = {}
    joints = {}
    offset = 0
    for person in (1, 2):
        roots[person] = np.asarray(vector[offset : offset + 3], dtype=np.float64)
        offset += 3
        updated = np.array(joints0[person], copy=True, dtype=np.float64)
        count = len(posture[person])
        updated[posture[person]] = vector[offset : offset + count]
        offset += count
        joints[person] = updated
    return roots, joints


def _set_static_pose(
    model: object,
    data: object,
    trajectories: dict[int, dict[str, np.ndarray]],
    frame: int,
    roots: dict[int, np.ndarray],
    joints: dict[int, np.ndarray],
) -> None:
    import mujoco

    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    for person, prefix in ((1, "r1_"), (2, "r2_")):
        trajectory = trajectories[person]
        free_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint"
        )
        qpos_adr = model.jnt_qposadr[free_id]
        yaw = float(trajectory["root_yaw"][frame])
        data.qpos[qpos_adr : qpos_adr + 3] = roots[person]
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = [
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2),
        ]
        for name, value in zip(trajectory["joint_names"], joints[person], strict=True):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, prefix + str(name)
            )
            data.qpos[model.jnt_qposadr[joint_id]] = value


def _robot_equilibrium_terms(
    model: object,
    data: object,
    prefix: str,
    joint_names: list[str],
) -> dict[str, float]:
    import mujoco

    residual = np.asarray(data.qfrc_inverse, dtype=np.float64)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
    dof_adr = int(model.jnt_dofadr[free_id])
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
        for name in joint_names
    ]
    joint_dofs = np.asarray([model.jnt_dofadr[jid] for jid in joint_ids])
    jacobian, _ = _foot_force_jacobian(model, data, prefix)
    base_jac = jacobian[:, dof_adr : dof_adr + 6]
    forces = solve_foot_forces(base_jac, residual[dof_adr : dof_adr + 6])
    unilateral = clip_unilateral_normals(forces)
    leftover = residual[dof_adr : dof_adr + 6] - base_jac.T @ unilateral
    tau = residual[joint_dofs] - jacobian[:, joint_dofs].T @ unilateral
    tensile = float(max(-forces[2], 0.0) + max(-forces[5], 0.0))
    palms = []
    feet_z = []
    for side in ("left", "right"):
        palms.append(
            data.xpos[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + f"{side}_wrist_yaw_link")
            ].copy()
        )
        feet_z.append(
            float(
                data.xpos[
                    mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, prefix + f"{side}_ankle_pitch_link"
                    )
                ][2]
            )
        )
    return {
        "base": float(np.mean(leftover**2)),
        "tensile": tensile,
        "torque": float(np.mean(tau**2)),
        "palms": np.asarray(palms),
        "feet_z": np.asarray(feet_z),
    }


def _frame_bounds(
    model: object,
    trajectories: dict[int, dict[str, np.ndarray]],
    config: EquilibriumConfig,
    posture: dict[int, list[int]],
    roots0: dict[int, np.ndarray],
    joints0: dict[int, np.ndarray],
) -> tuple[list[float], list[float]]:
    import mujoco

    lower: list[float] = []
    upper: list[float] = []
    for person in (1, 2):
        lower.extend(
            roots0[person] - [config.root_xy_bound_m, config.root_xy_bound_m, config.root_z_bound_m]
        )
        upper.extend(
            roots0[person] + [config.root_xy_bound_m, config.root_xy_bound_m, config.root_z_bound_m]
        )
        names = [str(name) for name in trajectories[person]["joint_names"]]
        for index in posture[person]:
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                ("r1_" if person == 1 else "r2_") + names[index],
            )
            j_lo, j_hi = model.jnt_range[joint_id]
            center = joints0[person][index]
            lower.append(max(j_lo, center - config.joint_bound_rad))
            upper.append(min(j_hi, center + config.joint_bound_rad))
    return lower, upper


def _optimize_loaded_frame(
    model: object,
    data: object,
    trajectories: dict[int, dict[str, np.ndarray]],
    frame: int,
    config: EquilibriumConfig,
    posture: dict[int, list[int]],
    *,
    x0: np.ndarray | None = None,
    prev_roots: dict[int, np.ndarray] | None = None,
    prev_joints: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, object]]:
    import mujoco

    roots0 = {
        person: np.array(trajectories[person]["root_translation"][frame], dtype=np.float64)
        for person in (1, 2)
    }
    joints0 = {
        person: np.array(trajectories[person]["joint_positions"][frame], dtype=np.float64)
        for person in (1, 2)
    }
    _set_static_pose(model, data, trajectories, frame, roots0, joints0)
    mujoco.mj_inverse(model, data)
    palms0 = {
        person: _robot_equilibrium_terms(
            model,
            data,
            "r1_" if person == 1 else "r2_",
            [str(name) for name in trajectories[person]["joint_names"]],
        )["palms"]
        for person in (1, 2)
    }
    lower, upper = _frame_bounds(model, trajectories, config, posture, roots0, joints0)
    if x0 is None:
        start = _pack(roots0, joints0, posture)
    else:
        start = np.clip(np.asarray(x0, dtype=np.float64), lower, upper)

    def objective(vector: np.ndarray) -> float:
        roots, joints = _unpack(vector, roots0, joints0, posture)
        _set_static_pose(model, data, trajectories, frame, roots, joints)
        mujoco.mj_inverse(model, data)
        cost = 0.0
        for person, prefix in ((1, "r1_"), (2, "r2_")):
            names = [str(name) for name in trajectories[person]["joint_names"]]
            terms = _robot_equilibrium_terms(model, data, prefix, names)
            cost += config.base_weight * terms["base"]
            cost += config.tensile_weight * terms["tensile"]
            cost += config.torque_weight * terms["torque"]
            cost += config.palm_weight * float(np.mean((terms["palms"] - palms0[person]) ** 2))
            cost += config.foot_height_weight * float(
                np.mean((terms["feet_z"] - config.target_foot_z) ** 2)
            )
            delta = joints[person][posture[person]] - joints0[person][posture[person]]
            cost += config.posture_weight * float(np.mean(delta**2))
            if prev_roots is not None and prev_joints is not None and config.temporal_weight > 0:
                cost += config.temporal_weight * float(
                    np.mean((roots[person] - prev_roots[person]) ** 2)
                )
                cost += config.temporal_weight * float(
                    np.mean(
                        (joints[person][posture[person]] - prev_joints[person][posture[person]]) ** 2
                    )
                )
        return float(cost)

    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper, strict=True)),
        options={"maxiter": config.max_iterations, "ftol": 1e-9},
    )
    roots, joints = _unpack(result.x, roots0, joints0, posture)
    _set_static_pose(model, data, trajectories, frame, roots, joints)
    mujoco.mj_inverse(model, data)
    diagnostics = {
        "success": bool(result.success),
        "message": str(result.message),
        "initial_cost": float(objective(start)),
        "final_cost": float(result.fun),
        "nfev": int(result.nfev),
        "frame": frame,
        "robots": {},
    }
    for person, prefix in ((1, "r1_"), (2, "r2_")):
        names = [str(name) for name in trajectories[person]["joint_names"]]
        terms = _robot_equilibrium_terms(model, data, prefix, names)
        diagnostics["robots"][str(person)] = {
            "unilateral_base_mse": terms["base"],
            "tensile_n": terms["tensile"],
            "root_delta_m": (roots[person] - roots0[person]).tolist(),
        }
    return roots, joints, diagnostics


def refine_frame_equilibrium(
    scene_xml: Path,
    robot1_path: Path,
    robot2_path: Path,
    frame: int = 0,
    config: EquilibriumConfig | None = None,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, object]]:
    """Adjust one frame's root and posture joints for unilateral stance."""
    if config is None:
        config = EquilibriumConfig()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    trajectories = {1: _load_trajectory(robot1_path), 2: _load_trajectory(robot2_path)}
    posture = {
        person: _posture_indices([str(name) for name in trajectories[person]["joint_names"]])
        for person in (1, 2)
    }
    roots, joints, diagnostics = _optimize_loaded_frame(
        model, data, trajectories, frame, config, posture
    )
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for person in (1, 2):
        payload = {name: np.array(trajectories[person][name], copy=True) for name in trajectories[person]}
        payload["joint_positions"][frame] = joints[person]
        payload["root_translation"][frame] = roots[person]
        outputs[person] = payload
    return outputs, diagnostics


def refine_clip_equilibrium(
    scene_xml: Path,
    robot1_path: Path,
    robot2_path: Path,
    config: EquilibriumConfig | None = None,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, object]]:
    """Per-frame unilateral post-pass with warm start. Does not freeze to frame 0."""
    if config is None:
        config = CLIP_EQUILIBRIUM_CONFIG
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    source = {1: _load_trajectory(robot1_path), 2: _load_trajectory(robot2_path)}
    n_total = int(source[1]["joint_positions"].shape[0])
    if source[2]["joint_positions"].shape[0] != n_total:
        raise ValueError("robot1 and robot2 frame counts differ")
    end_frame = n_total if max_frames is None else min(n_total, start_frame + max_frames)
    posture = {
        person: _posture_indices([str(name) for name in source[person]["joint_names"]])
        for person in (1, 2)
    }
    outputs = {
        person: {name: np.array(source[person][name], copy=True) for name in source[person]}
        for person in (1, 2)
    }
    prev_roots: dict[int, np.ndarray] | None = None
    prev_joints: dict[int, np.ndarray] | None = None
    prev_vector: np.ndarray | None = None
    frame_rows: list[dict[str, object]] = []
    print(f"[unilateral] config={asdict(config)} frames={start_frame}:{end_frame}", flush=True)
    for frame in range(start_frame, end_frame):
        roots, joints, frame_diag = _optimize_loaded_frame(
            model,
            data,
            source,
            frame,
            config,
            posture,
            x0=prev_vector,
            prev_roots=prev_roots,
            prev_joints=prev_joints,
        )
        for person in (1, 2):
            outputs[person]["joint_positions"][frame] = joints[person]
            outputs[person]["root_translation"][frame] = roots[person]
        prev_roots = roots
        prev_joints = joints
        prev_vector = _pack(roots, joints, posture)
        frame_rows.append(
            {
                "frame": frame,
                "success": frame_diag["success"],
                "final_cost": frame_diag["final_cost"],
                "r1_tensile_n": frame_diag["robots"]["1"]["tensile_n"],
                "r2_tensile_n": frame_diag["robots"]["2"]["tensile_n"],
                "r1_unilateral_base_mse": frame_diag["robots"]["1"]["unilateral_base_mse"],
                "r2_unilateral_base_mse": frame_diag["robots"]["2"]["unilateral_base_mse"],
            }
        )
        if frame == start_frame or (frame + 1) % 20 == 0 or frame + 1 == end_frame:
            print(
                f"[unilateral] frame {frame + 1}/{end_frame} "
                f"tensile r1={frame_diag['robots']['1']['tensile_n']:.1f} "
                f"r2={frame_diag['robots']['2']['tensile_n']:.1f} "
                f"cost={frame_diag['final_cost']:.3g}",
                flush=True,
            )
    tensile_r1 = [float(row["r1_tensile_n"]) for row in frame_rows]
    tensile_r2 = [float(row["r2_tensile_n"]) for row in frame_rows]
    diagnostics = {
        "mode": "clip",
        "config": asdict(config),
        "start_frame": start_frame,
        "frames": len(frame_rows),
        "success_ratio": float(np.mean([bool(row["success"]) for row in frame_rows])),
        "mean_tensile_n": {
            "1": float(np.mean(tensile_r1)),
            "2": float(np.mean(tensile_r2)),
        },
        "mean_unilateral_base_mse": {
            "1": float(np.mean([row["r1_unilateral_base_mse"] for row in frame_rows])),
            "2": float(np.mean([row["r2_unilateral_base_mse"] for row in frame_rows])),
        },
        "frames_detail": frame_rows,
    }
    return outputs, diagnostics


def _rollout_indices(joint_names: list[str]) -> list[int]:
    tokens = ("ankle_pitch", "ankle_roll", "hip_pitch", "hip_roll")
    return [index for index, name in enumerate(joint_names) if any(token in name for token in tokens)]


def _pd_hold_min_height(
    model: object,
    data: object,
    trajectories: dict[int, dict[str, np.ndarray]],
    frame: int,
    roots: dict[int, np.ndarray],
    joints: dict[int, np.ndarray],
    *,
    duration: float,
    timestep: float,
    leg_kp: float,
) -> np.ndarray:
    import mujoco

    mujoco.mj_resetData(model, data)
    for index in range(int(mujoco.mjtWarning.mjNWARNING)):
        data.warning[index].number = 0
    _set_static_pose(model, data, trajectories, frame, roots, joints)
    data.time = 0.0
    model.opt.timestep = timestep
    mujoco.mj_forward(model, data)
    metadata = {}
    pelvis = {}
    targets = {}
    for person, prefix in ((1, "r1_"), (2, "r2_")):
        pelvis[person] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
        targets[person] = joints[person]
        rows = []
        for name, value in zip(trajectories[person]["joint_names"], joints[person], strict=True):
            joint_name = str(name)
            full = prefix + joint_name
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
            kp = leg_kp if any(token in joint_name for token in ("hip", "knee", "ankle", "waist")) else 30.0
            kd = 2.0 * np.sqrt(kp)
            rows.append((model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id], actuator_id, kp, kd, value, float(np.max(np.abs(model.jnt_actfrcrange[joint_id])))))
        metadata[person] = rows
    mins = np.array(
        [float(data.xpos[pelvis[1]][2]), float(data.xpos[pelvis[2]][2])],
        dtype=np.float64,
    )
    while data.time < duration:
        for person in (1, 2):
            for qpos, qvel, actuator, kp, kd, target, limit in metadata[person]:
                torque = kp * (target - data.qpos[qpos]) - kd * data.qvel[qvel]
                if limit > 0:
                    torque = float(np.clip(torque, -limit, limit))
                data.ctrl[actuator] = torque
        mujoco.mj_step(model, data)
        for index, person in enumerate((1, 2)):
            height = float(data.xpos[pelvis[person]][2])
            if height < mins[index]:
                mins[index] = height
        if not np.isfinite(mins).all() or not np.isfinite(data.qpos).all():
            mins[:] = -1.0
            break
        if data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number > 0:
            mins[:] = np.minimum(mins, -1.0)
            break
        if mins.min() < 0.45:
            break
    return mins


def refine_frame_rollout(
    scene_xml: Path,
    robot1_path: Path,
    robot2_path: Path,
    frame: int = 0,
    *,
    duration: float = 0.55,
    max_iterations: int = 25,
    leg_kp: float = 80.0,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, object]]:
    """Nudge root and ankles/hips to survive a short free-base PD hold."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    trajectories = {1: _load_trajectory(robot1_path), 2: _load_trajectory(robot2_path)}
    posture = {
        person: _rollout_indices([str(name) for name in trajectories[person]["joint_names"]])
        for person in (1, 2)
    }
    roots0 = {person: np.array(trajectories[person]["root_translation"][frame], dtype=np.float64) for person in (1, 2)}
    joints0 = {person: np.array(trajectories[person]["joint_positions"][frame], dtype=np.float64) for person in (1, 2)}
    x0 = _pack(roots0, joints0, posture)
    initial_heights = _pd_hold_min_height(
        model, data, trajectories, frame, roots0, joints0,
        duration=duration, timestep=0.005, leg_kp=leg_kp,
    )

    def objective(vector: np.ndarray) -> float:
        roots, joints = _unpack(vector, roots0, joints0, posture)
        heights = _pd_hold_min_height(
            model, data, trajectories, frame, roots, joints,
            duration=duration, timestep=0.005, leg_kp=leg_kp,
        )
        fall = np.maximum(0.50 - heights, 0.0)
        stay = roots0[1] - roots[1]
        stay2 = roots0[2] - roots[2]
        return float(np.sum(fall**2) * 80.0 + np.sum((0.75 - heights) ** 2) + 1.5 * (np.sum(stay**2) + np.sum(stay2**2)))

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": max_iterations, "maxfev": max_iterations * 4, "xatol": 2e-3, "fatol": 2e-3},
    )
    roots, joints = _unpack(result.x, roots0, joints0, posture)
    final_heights = _pd_hold_min_height(
        model, data, trajectories, frame, roots, joints,
        duration=duration, timestep=0.005, leg_kp=leg_kp,
    )
    diagnostics = {
        "mode": "rollout",
        "success": bool(result.success),
        "message": str(result.message),
        "initial_min_height_m": initial_heights.tolist(),
        "final_min_height_m": final_heights.tolist(),
        "initial_cost": float(objective(x0)),
        "final_cost": float(result.fun),
        "nfev": int(result.nfev),
        "duration_s": duration,
        "frame": frame,
        "robots": {
            str(person): {"root_delta_m": (roots[person] - roots0[person]).tolist()}
            for person in (1, 2)
        },
    }
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for person in (1, 2):
        payload = {name: np.array(trajectories[person][name], copy=True) for name in trajectories[person]}
        payload["joint_positions"][frame] = joints[person]
        payload["root_translation"][frame] = roots[person]
        outputs[person] = payload
    return outputs, diagnostics


def _stance_hold_metrics(
    model: object,
    data: object,
    trajectories: dict[int, dict[str, np.ndarray]],
    frame: int,
    roots: dict[int, np.ndarray],
    joints: dict[int, dict[str, np.ndarray]] | dict[int, np.ndarray],
    *,
    duration: float,
    timestep: float,
    collapse_height: float = 0.45,
) -> dict[str, np.ndarray]:
    """Position-actuator hold; fall times, min pelvis height, final CoM error."""
    import mujoco

    from ..controller import lift_roots_out_of_penetration

    mujoco.mj_resetData(model, data)
    _set_static_pose(model, data, trajectories, frame, roots, joints)
    lift_roots_out_of_penetration(model, data)
    data.time = 0.0
    model.opt.timestep = timestep
    pelvis = {
        person: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
        for person, prefix in ((1, "r1_"), (2, "r2_"))
    }
    ankles = {
        person: (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "left_ankle_roll_link"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "right_ankle_roll_link"),
        )
        for person, prefix in ((1, "r1_"), (2, "r2_"))
    }
    actuators = {}
    for person, prefix in ((1, "r1_"), (2, "r2_")):
        rows = []
        for name, value in zip(trajectories[person]["joint_names"], joints[person], strict=True):
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + str(name)
            )
            rows.append((actuator_id, float(value)))
        actuators[person] = rows
    fall = np.array([duration, duration], dtype=np.float64)
    min_z = np.array([10.0, 10.0], dtype=np.float64)
    final_com = np.zeros(2, dtype=np.float64)
    seen = [False, False]
    while data.time < duration:
        for person in (1, 2):
            for actuator_id, target in actuators[person]:
                data.ctrl[actuator_id] = target
        mujoco.mj_step(model, data)
        for index, person in enumerate((1, 2)):
            height = float(data.xpos[pelvis[person]][2])
            up = float(data.xmat[pelvis[person]].reshape(3, 3)[2, 2])
            if data.time > 0.2:
                min_z[index] = min(min_z[index], height)
            left, right = ankles[person]
            support = 0.5 * (data.xpos[left][:2] + data.xpos[right][:2])
            com = data.subtree_com[pelvis[person]][:2]
            final_com[index] = float(np.linalg.norm(com - support))
            if seen[index]:
                continue
            if height < collapse_height or up < 0.5:
                fall[index] = float(data.time)
                seen[index] = True
        if all(seen) or not np.isfinite(data.qpos).all():
            break
    return {"fall_s": fall, "min_z_m": min_z, "final_com_m": final_com}


def _stance_hold_times(
    model: object,
    data: object,
    trajectories: dict[int, dict[str, np.ndarray]],
    frame: int,
    roots: dict[int, np.ndarray],
    joints: dict[int, dict[str, np.ndarray]] | dict[int, np.ndarray],
    *,
    duration: float,
    timestep: float,
    fall_height: float = 0.45,
) -> np.ndarray:
    """Position-actuator hold; returns first fall time per robot (duration if none)."""
    metrics = _stance_hold_metrics(
        model,
        data,
        trajectories,
        frame,
        roots,
        joints,
        duration=duration,
        timestep=timestep,
        collapse_height=fall_height,
    )
    return metrics["fall_s"]


def refine_stance_pose(
    scene_xml: Path,
    robot1_path: Path,
    robot2_path: Path,
    frame: int = 0,
    *,
    duration: float = 5.5,
    keep_pose: bool = False,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, object]]:
    """Lift, project CoM, then search split ankles for a long free-base hold."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from ..controller import lift_roots_out_of_penetration, project_com_onto_support
    from ..pd_tracking import _sync_frame0_from_state

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    trajectories = {1: _load_trajectory(robot1_path), 2: _load_trajectory(robot2_path)}
    roots = {
        person: np.array(trajectories[person]["root_translation"][frame], dtype=np.float64)
        for person in (1, 2)
    }
    joints = {
        person: np.array(trajectories[person]["joint_positions"][frame], dtype=np.float64)
        for person in (1, 2)
    }
    _set_static_pose(model, data, trajectories, frame, roots, joints)
    lift_roots_out_of_penetration(model, data)
    if not keep_pose:
        for prefix in ("r1_", "r2_"):
            project_com_onto_support(model, data, prefix)
        lift_roots_out_of_penetration(model, data)
        if frame == 0:
            _sync_frame0_from_state(model, data, trajectories)
            roots = {
                person: np.array(trajectories[person]["root_translation"][0], dtype=np.float64)
                for person in (1, 2)
            }
            joints = {
                person: np.array(trajectories[person]["joint_positions"][0], dtype=np.float64)
                for person in (1, 2)
            }
        else:
            for person, prefix in ((1, "r1_"), (2, "r2_")):
                free_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint"
                )
                address = int(model.jnt_qposadr[free_id])
                roots[person] = np.array(data.qpos[address : address + 3], dtype=np.float64)
                for index, name in enumerate(trajectories[person]["joint_names"]):
                    joint_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, prefix + str(name)
                    )
                    joints[person][index] = data.qpos[int(model.jnt_qposadr[joint_id])]

    names = {
        person: [str(name) for name in trajectories[person]["joint_names"]]
        for person in (1, 2)
    }

    def rank(metrics: dict[str, np.ndarray]) -> float:
        z = np.array(metrics["min_z_m"], dtype=np.float64)
        fall = metrics["fall_s"]
        for index in range(2):
            if fall[index] < duration - 1e-6 or z[index] < 0.70:
                z[index] = 0.12 * fall[index]
        return float(z.min()) - 0.2 * float(np.mean(metrics["final_com_m"]))

    baseline = _stance_hold_metrics(
        model, data, trajectories, frame, roots, joints, duration=duration, timestep=0.002
    )
    best_score = rank(baseline)
    best_joints = {person: np.array(joints[person], copy=True) for person in (1, 2)}
    best_metrics = baseline
    best_delta: dict[str, object] = {}
    groups = []
    for person in (1, 2):
        for token in (
            "left_ankle_pitch",
            "right_ankle_pitch",
            "left_ankle_roll",
            "right_ankle_roll",
            "left_hip_pitch",
            "right_hip_pitch",
        ):
            indices = [i for i, name in enumerate(names[person]) if name.startswith(token)]
            if indices:
                groups.append((person, token, indices))
    original = {person: np.array(best_joints[person], copy=True) for person in (1, 2)}
    for person, token, indices in groups:
        current = {key: np.array(best_joints[key], copy=True) for key in (1, 2)}
        for delta in np.linspace(-0.14, 0.12, 8):
            trial = {key: np.array(current[key], copy=True) for key in (1, 2)}
            trial[person][indices] = original[person][indices] + delta
            metrics = _stance_hold_metrics(
                model, data, trajectories, frame, roots, trial, duration=duration, timestep=0.002
            )
            score = rank(metrics)
            if score > best_score + 1e-4:
                best_score = score
                best_joints = trial
                best_metrics = metrics
                best_delta[f"r{person}_{token}"] = float(delta)
            if (
                metrics["fall_s"].min() >= duration - 1e-6
                and metrics["min_z_m"].min() >= 0.72
            ):
                break
    diagnostics = {
        "mode": "stance",
        "keep_pose": keep_pose,
        "baseline_fall_s": baseline["fall_s"].tolist(),
        "baseline_min_z_m": baseline["min_z_m"].tolist(),
        "final_fall_s": best_metrics["fall_s"].tolist(),
        "final_min_z_m": best_metrics["min_z_m"].tolist(),
        "final_com_m": best_metrics["final_com_m"].tolist(),
        "joint_delta_rad": best_delta,
        "duration_s": duration,
        "frame": frame,
        "score": best_score,
    }
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for person in (1, 2):
        payload = {name: np.array(trajectories[person][name], copy=True) for name in trajectories[person]}
        payload["joint_positions"][frame] = best_joints[person]
        payload["root_translation"][frame] = roots[person]
        outputs[person] = payload
    return outputs, diagnostics


def apply_stance_clip(
    scene_xml: Path,
    robot1_path: Path,
    robot2_path: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, object]]:
    """Lift every frame. Keep frame-0 legs; later frames only move arms/waist."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from ..controller import lift_roots_out_of_penetration, project_com_onto_support

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    trajectories = {1: _load_trajectory(robot1_path), 2: _load_trajectory(robot2_path)}
    frame_count = min(
        len(trajectories[1]["joint_positions"]),
        len(trajectories[2]["joint_positions"]),
    )
    for frame in range(frame_count):
        roots = {
            person: np.array(trajectories[person]["root_translation"][frame], dtype=np.float64)
            for person in (1, 2)
        }
        joints = {
            person: np.array(trajectories[person]["joint_positions"][frame], dtype=np.float64)
            for person in (1, 2)
        }
        _set_static_pose(model, data, trajectories, frame, roots, joints)
        lift_roots_out_of_penetration(model, data)
        for prefix in ("r1_", "r2_"):
            project_com_onto_support(model, data, prefix)
        lift_roots_out_of_penetration(model, data)
        for person, prefix in ((1, "r1_"), (2, "r2_")):
            free_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint"
            )
            address = int(model.jnt_qposadr[free_id])
            trajectories[person]["root_translation"][frame] = np.array(
                data.qpos[address : address + 3], dtype=np.float64
            )
            if frame == 0:
                continue
            stance = trajectories[person]["joint_positions"][0]
            for index, name in enumerate(trajectories[person]["joint_names"]):
                if is_leg_joint(str(name)):
                    trajectories[person]["joint_positions"][frame, index] = stance[index]
                    continue
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, prefix + str(name)
                )
                trajectories[person]["joint_positions"][frame, index] = data.qpos[
                    int(model.jnt_qposadr[joint_id])
                ]
    diagnostics = {"mode": "stance_clip", "frames": frame_count}
    return trajectories, diagnostics


def filter_swing_com_hold(
    scene_xml: Path,
    robot1_path: Path,
    robot2_path: Path,
    *,
    person: int = 2,
    margin: float = 0.008,
    max_joint_step: float = 0.08,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, object]]:
    """Rewrite the hover-leg joints so commanded CoM xy stays at frame 0.

    Planted chain, waist, arms, and both roots stay as in the source clip.
    Frame 0 is unchanged. Swing joints are rate-limited toward the clip, then
    projected. Does not write in place; the caller chooses the output directory.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    trajectories = {1: _load_trajectory(robot1_path), 2: _load_trajectory(robot2_path)}
    prefix = f"r{person}_"
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
    names = [str(name) for name in trajectories[person]["joint_names"]]
    frame_count = min(
        len(trajectories[1]["joint_positions"]),
        len(trajectories[2]["joint_positions"]),
    )
    frame0 = {
        key: np.array(trajectories[key]["joint_positions"][0], dtype=np.float64)
        for key in (1, 2)
    }
    roots0 = {
        key: np.array(trajectories[key]["root_translation"][0], dtype=np.float64)
        for key in (1, 2)
    }
    _set_static_pose(model, data, trajectories, 0, roots0, frame0)
    mujoco.mj_forward(model, data)
    com0 = np.array(data.subtree_com[pelvis][:2], dtype=np.float64)
    z_left = min_sole_center_z(model, data, prefix, "left")
    z_right = min_sole_center_z(model, data, prefix, "right")
    planted = planted_foot_sides(z_left, z_right, margin)
    swing = tuple(side for side in ("left", "right") if side not in planted)
    z_refs = {
        side: min_sole_center_z(model, data, prefix, side) for side in swing
    }
    swing_mask = np.array(
        [is_leg_joint(name) and joint_side(name) in swing for name in names]
    )
    swing_indices = np.flatnonzero(swing_mask)
    source = np.array(trajectories[person]["joint_positions"], copy=True)
    filtered = np.array(source, copy=True)
    prev_swing = np.array(frame0[person][swing_mask], copy=True)
    before_dcom: list[float] = []
    after_dcom: list[float] = []
    before_z: list[float] = []
    after_z: list[float] = []
    swing_rms_clip: list[float] = []
    peak_step = 0.0
    step_limit = float(max_joint_step)
    for frame in range(frame_count):
        clip = np.array(source[frame], dtype=np.float64)
        mixed = np.array(frame0[person], copy=True)
        if frame == 0 or not np.any(swing_mask):
            mixed[swing_mask] = clip[swing_mask]
        else:
            mixed[swing_mask] = prev_swing + np.clip(
                clip[swing_mask] - prev_swing, -step_limit, step_limit
            )
        joints = {1: np.array(frame0[1], copy=True), 2: np.array(frame0[2], copy=True)}
        joints[person] = mixed
        _set_static_pose(model, data, trajectories, 0, roots0, joints)
        mujoco.mj_forward(model, data)
        before_dcom.append(
            float(np.linalg.norm(data.subtree_com[pelvis][:2] - com0))
        )
        z_side = swing[0] if swing else "right"
        before_z.append(min_sole_center_z(model, data, prefix, z_side))
        for side in swing:
            hold_com_xy(
                model,
                data,
                prefix,
                com0,
                swing_side=side,
                z_ref=z_refs.get(side),
            )
        ik_swing = np.array(mixed[swing_mask], copy=True)
        for offset, index in enumerate(swing_indices):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, prefix + names[int(index)]
            )
            ik_swing[offset] = float(data.qpos[int(model.jnt_qposadr[joint_id])])
        if frame == 0:
            limited = ik_swing
        else:
            limited = prev_swing + np.clip(
                ik_swing - prev_swing, -step_limit, step_limit
            )
        peak_step = max(peak_step, float(np.max(np.abs(limited - prev_swing))))
        mixed[swing_mask] = limited
        joints[person] = mixed
        _set_static_pose(model, data, trajectories, 0, roots0, joints)
        mujoco.mj_forward(model, data)
        after_dcom.append(
            float(np.linalg.norm(data.subtree_com[pelvis][:2] - com0))
        )
        after_z.append(min_sole_center_z(model, data, prefix, z_side))
        updated = np.array(clip, copy=True)
        updated[swing_mask] = limited
        filtered[frame] = updated
        prev_swing = np.array(limited, copy=True)
        if np.any(swing_mask):
            swing_rms_clip.append(
                float(np.sqrt(np.mean((limited - clip[swing_mask]) ** 2)))
            )
    filtered[0] = frame0[person]
    trajectories[person]["joint_positions"] = filtered
    diagnostics: dict[str, object] = {
        "mode": "filter_swing_com_hold",
        "person": person,
        "planted_sides": list(planted),
        "swing_sides": list(swing),
        "frames": frame_count,
        "max_joint_step_rad": step_limit,
        "peak_swing_step_rad": peak_step,
        "peak_swing_speed_rad_s": float(peak_step * 15.0),
        "com_xy_ref_m": [float(com0[0]), float(com0[1])],
        "z_refs_m": {side: float(value) for side, value in z_refs.items()},
        "dcom_before_max_m": float(max(before_dcom) if before_dcom else 0.0),
        "dcom_after_max_m": float(max(after_dcom) if after_dcom else 0.0),
        "dcom_before_mean_m": float(np.mean(before_dcom) if before_dcom else 0.0),
        "dcom_after_mean_m": float(np.mean(after_dcom) if after_dcom else 0.0),
        "swing_z_before_min_m": float(min(before_z) if before_z else 0.0),
        "swing_z_after_min_m": float(min(after_z) if after_z else 0.0),
        "swing_rms_vs_clip_mean_rad": float(
            np.mean(swing_rms_clip) if swing_rms_clip else 0.0
        ),
    }
    return trajectories, diagnostics


def synthesize_hover(
    scene_xml: Path,
    robot1_path: Path,
    robot2_path: Path,
    *,
    person: int = 2,
    margin: float = 0.008,
    yaw_amp: float = 0.12,
    roll_amp: float = 0.0,
    period_s: float = 2.0,
    fps: float = 15.0,
    hold_com: bool = True,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, object]]:
    """Keep frame-0 legs and add a CoM-neutral hip-yaw/roll sine on the hover foot."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    trajectories = {1: _load_trajectory(robot1_path), 2: _load_trajectory(robot2_path)}
    prefix = f"r{person}_"
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
    names = [str(name) for name in trajectories[person]["joint_names"]]
    frame_count = min(
        len(trajectories[1]["joint_positions"]),
        len(trajectories[2]["joint_positions"]),
    )
    frame0 = {
        key: np.array(trajectories[key]["joint_positions"][0], dtype=np.float64)
        for key in (1, 2)
    }
    roots0 = {
        key: np.array(trajectories[key]["root_translation"][0], dtype=np.float64)
        for key in (1, 2)
    }
    _set_static_pose(model, data, trajectories, 0, roots0, frame0)
    mujoco.mj_forward(model, data)
    com0 = np.array(data.subtree_com[pelvis][:2], dtype=np.float64)
    planted = planted_foot_sides(
        min_sole_center_z(model, data, prefix, "left"),
        min_sole_center_z(model, data, prefix, "right"),
        margin,
    )
    swing = tuple(side for side in ("left", "right") if side not in planted)
    side = swing[0] if swing else "right"
    name_index = {name: index for index, name in enumerate(names)}
    yaw_key = f"{side}_hip_yaw_joint"
    roll_key = f"{side}_hip_roll_joint"
    source = np.array(trajectories[person]["joint_positions"], copy=True)
    filtered = np.array(source, copy=True)
    z_ref = min_sole_center_z(model, data, prefix, side)
    after_dcom: list[float] = []
    for frame in range(frame_count):
        pose = np.array(source[frame], copy=True)
        for index, name in enumerate(names):
            if is_leg_joint(name):
                pose[index] = frame0[person][index]
        phase = 2.0 * np.pi * (frame / max(fps, 1e-6)) / max(period_s, 1e-6)
        if yaw_key in name_index:
            pose[name_index[yaw_key]] += float(yaw_amp) * np.sin(phase)
        if roll_key in name_index:
            pose[name_index[roll_key]] += float(roll_amp) * np.sin(phase)
        joints = {1: np.array(frame0[1], copy=True), 2: np.array(frame0[2], copy=True)}
        joints[person] = np.array(frame0[person], copy=True)
        if yaw_key in name_index:
            joints[person][name_index[yaw_key]] = pose[name_index[yaw_key]]
        if roll_amp and roll_key in name_index:
            joints[person][name_index[roll_key]] = pose[name_index[roll_key]]
        _set_static_pose(model, data, trajectories, 0, roots0, joints)
        mujoco.mj_forward(model, data)
        if hold_com:
            hold_com_xy(
                model, data, prefix, com0, swing_side=side, z_ref=z_ref
            )
            for index, name in enumerate(names):
                if is_leg_joint(name) and joint_side(name) == side:
                    joint_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name
                    )
                    pose[index] = float(data.qpos[int(model.jnt_qposadr[joint_id])])
        after_dcom.append(
            float(np.linalg.norm(data.subtree_com[pelvis][:2] - com0))
        )
        filtered[frame] = pose
    filtered[0] = frame0[person]
    trajectories[person]["joint_positions"] = filtered
    diagnostics: dict[str, object] = {
        "mode": "synthesize_hover",
        "person": person,
        "planted_sides": list(planted),
        "swing_sides": list(swing),
        "yaw_amp_rad": float(yaw_amp),
        "roll_amp_rad": float(roll_amp),
        "period_s": float(period_s),
        "dcom_after_max_m": float(max(after_dcom) if after_dcom else 0.0),
        "dcom_after_mean_m": float(np.mean(after_dcom) if after_dcom else 0.0),
    }
    return trajectories, diagnostics


def save_equilibrium_outputs(
    outputs: dict[int, dict[str, np.ndarray]],
    diagnostics: dict[str, object],
    output_dir: Path,
    urdf_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for person, payload in outputs.items():
        names = [str(name) for name in payload["keypoint_names"]]
        payload["robot_keypoints"] = recompute_keypoints(
            urdf_path,
            payload["joint_positions"],
            payload["root_translation"],
            payload["root_yaw"],
            names,
        )
        np.savez_compressed(output_dir / f"robot{person}.npz", **payload)
    (output_dir / "equilibrium.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )

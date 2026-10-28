"""Contact-constrained inverse dynamics: foot forces explain the floating base."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .pd_tracking import _load_trajectory


def solve_foot_forces(jacobian_base: np.ndarray, residual_base: np.ndarray) -> np.ndarray:
    """Least-squares 3D forces at two feet that match a 6-DoF base residual.

    ``jacobian_base`` is 6 (fx,fy,fz × 2 feet) by 6 (base dofs), mapping
    ``tau_base = jacobian_base.T @ forces``.
    """
    mapping = np.asarray(jacobian_base, dtype=np.float64)
    residual = np.asarray(residual_base, dtype=np.float64).reshape(6)
    forces, *_ = np.linalg.lstsq(mapping.T, residual, rcond=None)
    return forces.reshape(6)


def clip_unilateral_normals(forces: np.ndarray) -> np.ndarray:
    """Zero tensile (negative) foot normals; leave tangents unchanged."""
    clipped = np.asarray(forces, dtype=np.float64).reshape(6).copy()
    clipped[2] = max(float(clipped[2]), 0.0)
    clipped[5] = max(float(clipped[5]), 0.0)
    return clipped


def friction_cone_flags(forces: np.ndarray, mu: float) -> dict[str, np.ndarray]:
    packed = np.asarray(forces, dtype=np.float64).reshape(-1, 6)
    left = packed[:, 0:3]
    right = packed[:, 3:6]
    stacked = np.concatenate([left, right], axis=0)
    normal = stacked[:, 2]
    tangent = np.linalg.norm(stacked[:, :2], axis=1)
    return {
        "negative_normal": normal < -1e-6,
        "friction_violation": (normal > 1e-6) & (tangent > mu * normal + 1e-6),
        "normal": normal,
        "tangent": tangent,
    }


def _resample(values: np.ndarray, source_times: np.ndarray, dense_times: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return np.interp(dense_times, source_times, values)
    return np.stack(
        [np.interp(dense_times, source_times, values[:, index]) for index in range(values.shape[1])],
        axis=1,
    )


def _set_robot_id_state(
    model: object,
    data: object,
    prefix: str,
    trajectory: dict[str, np.ndarray],
    frame: int,
    *,
    acceleration: str,
    joints: np.ndarray,
    roots: np.ndarray,
    yaws: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    root_vel: np.ndarray,
    root_acc: np.ndarray,
    yaw_vel: np.ndarray,
    yaw_acc: np.ndarray,
) -> None:
    import mujoco

    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
    qpos_adr = model.jnt_qposadr[free_id]
    dof_adr = model.jnt_dofadr[free_id]
    yaw = float(yaws[frame])
    data.qpos[qpos_adr : qpos_adr + 3] = roots[frame]
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    names = [str(name) for name in trajectory["joint_names"]]
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name) for name in names
    ]
    if acceleration == "none":
        data.qvel[dof_adr : dof_adr + 6] = 0.0
        data.qacc[dof_adr : dof_adr + 6] = 0.0
        for jid in joint_ids:
            data.qvel[model.jnt_dofadr[jid]] = 0.0
            data.qacc[model.jnt_dofadr[jid]] = 0.0
    elif acceleration == "joints":
        data.qvel[dof_adr : dof_adr + 6] = 0.0
        data.qacc[dof_adr : dof_adr + 6] = 0.0
        for value, jid in zip(vel[frame], joint_ids, strict=True):
            data.qvel[model.jnt_dofadr[jid]] = value
        for value, jid in zip(acc[frame], joint_ids, strict=True):
            data.qacc[model.jnt_dofadr[jid]] = value
    else:
        data.qvel[dof_adr : dof_adr + 3] = root_vel[frame]
        data.qvel[dof_adr + 3 : dof_adr + 6] = [0.0, 0.0, yaw_vel[frame]]
        data.qacc[dof_adr : dof_adr + 3] = root_acc[frame]
        data.qacc[dof_adr + 3 : dof_adr + 6] = [0.0, 0.0, yaw_acc[frame]]
        for value, jid in zip(vel[frame], joint_ids, strict=True):
            data.qvel[model.jnt_dofadr[jid]] = value
        for value, jid in zip(acc[frame], joint_ids, strict=True):
            data.qacc[model.jnt_dofadr[jid]] = value
    for value, jid in zip(joints[frame], joint_ids, strict=True):
        data.qpos[model.jnt_qposadr[jid]] = value


def _foot_force_jacobian(model: object, data: object, prefix: str) -> tuple[np.ndarray, list[int]]:
    import mujoco

    blocks = []
    body_ids = []
    for side in ("left", "right"):
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + f"{side}_ankle_roll_link"
        )
        point = np.array(data.xpos[body_id], dtype=np.float64)
        point[2] = 0.0
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, point, body_id)
        blocks.append(jacp)
        body_ids.append(body_id)
    return np.vstack(blocks), body_ids


def contact_constrained_id(
    scene_xml: Path,
    robot1_path: Path,
    robot2_path: Path,
    *,
    source_fps: float = 15.0,
    upsample: int = 1,
    acceleration: str = "none",
    mu: float = 0.8,
) -> dict[str, object]:
    """Unconstrained vs two-foot-force ID on the same kinematic trajectory."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    raw = {1: _load_trajectory(robot1_path), 2: _load_trajectory(robot2_path)}
    frame_count = min(len(raw[1]["joint_positions"]), len(raw[2]["joint_positions"]))
    upsample = max(int(upsample), 1)
    dense_count = (frame_count - 1) * upsample + 1
    dt = 1.0 / (source_fps * upsample)
    source_times = np.arange(frame_count, dtype=np.float64)
    dense_times = np.linspace(0.0, frame_count - 1, dense_count)

    prepared: dict[int, dict[str, np.ndarray]] = {}
    for person in (1, 2):
        trajectory = raw[person]
        joints = _resample(trajectory["joint_positions"][:frame_count], source_times, dense_times)
        roots = _resample(trajectory["root_translation"][:frame_count], source_times, dense_times)
        yaws = _resample(trajectory["root_yaw"][:frame_count], source_times, dense_times)
        vel = np.gradient(joints, dt, axis=0)
        prepared[person] = {
            "joints": joints,
            "roots": roots,
            "yaws": yaws,
            "vel": vel,
            "acc": np.gradient(vel, dt, axis=0),
            "root_vel": np.gradient(roots, dt, axis=0),
            "root_acc": np.gradient(np.gradient(roots, dt, axis=0), dt, axis=0),
            "yaw_vel": np.gradient(yaws.astype(np.float64), dt),
            "yaw_acc": np.gradient(np.gradient(yaws.astype(np.float64), dt), dt),
        }

    report: dict[str, object] = {
        "source_fps": source_fps,
        "upsample": upsample,
        "acceleration": acceleration,
        "mu": mu,
        "frames": dense_count,
        "robots": {},
    }
    unconstrained = {1: [], 2: []}
    contacted = {1: [], 2: []}
    forces = {1: [], 2: []}
    forces_unilateral = {1: [], 2: []}
    base_residual = {1: [], 2: []}
    base_residual_unilateral = {1: [], 2: []}

    for frame in range(dense_count):
        for person, prefix in ((1, "r1_"), (2, "r2_")):
            _set_robot_id_state(
                model,
                data,
                prefix,
                raw[person],
                frame,
                acceleration=acceleration,
                **prepared[person],
            )
        mujoco.mj_inverse(model, data)
        residual = np.array(data.qfrc_inverse, copy=True)
        for person, prefix in ((1, "r1_"), (2, "r2_")):
            names = [str(name) for name in raw[person]["joint_names"]]
            joint_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
                for name in names
            ]
            joint_dofs = np.asarray([model.jnt_dofadr[jid] for jid in joint_ids])
            free_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint"
            )
            dof_adr = int(model.jnt_dofadr[free_id])
            jacobian, _ = _foot_force_jacobian(model, data, prefix)
            base_jac = jacobian[:, dof_adr : dof_adr + 6]
            foot_f = solve_foot_forces(base_jac, residual[dof_adr : dof_adr + 6])
            foot_u = clip_unilateral_normals(foot_f)
            tau_contact = residual[joint_dofs] - jacobian[:, joint_dofs].T @ foot_f
            unconstrained[person].append(residual[joint_dofs])
            contacted[person].append(tau_contact)
            forces[person].append(foot_f)
            forces_unilateral[person].append(foot_u)
            base_residual[person].append(
                residual[dof_adr : dof_adr + 6] - base_jac.T @ foot_f
            )
            base_residual_unilateral[person].append(
                residual[dof_adr : dof_adr + 6] - base_jac.T @ foot_u
            )

    for person in (1, 2):
        names = [str(name) for name in raw[person]["joint_names"]]
        joint_ids = [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                ("r1_" if person == 1 else "r2_") + name,
            )
            for name in names
        ]
        limits = np.asarray([np.max(np.abs(model.jnt_actfrcrange[jid])) for jid in joint_ids])
        raw_tau = np.asarray(unconstrained[person])
        contact_tau = np.asarray(contacted[person])
        foot_f = np.asarray(forces[person])
        flags = friction_cone_flags(foot_f, mu)
        report["robots"][str(person)] = {
            "unconstrained": _torque_stats(raw_tau, limits, names),
            "contact": _torque_stats(contact_tau, limits, names),
            "foot_force_mean_n": float(np.mean(np.linalg.norm(foot_f.reshape(-1, 2, 3), axis=-1))),
            "foot_force_peak_n": float(np.max(np.linalg.norm(foot_f.reshape(-1, 2, 3), axis=-1))),
            "negative_normal_ratio": float(np.mean(flags["negative_normal"])),
            "friction_cone_violation_ratio": float(np.mean(flags["friction_violation"])),
            "base_residual_after_contact_rms": float(
                np.sqrt(np.mean(np.asarray(base_residual[person]) ** 2))
            ),
            "unilateral_base_residual_rms": float(
                np.sqrt(np.mean(np.asarray(base_residual_unilateral[person]) ** 2))
            ),
            "unilateral_foot_force_mean_n": float(
                np.mean(np.linalg.norm(np.asarray(forces_unilateral[person]).reshape(-1, 2, 3), axis=-1))
            ),
        }
    return report


def _torque_stats(required: np.ndarray, limits: np.ndarray, names: list[str]) -> dict[str, object]:
    excess = np.abs(required) - limits[None]
    return {
        "torque_limit_violation_ratio": float(np.mean(excess > 0)),
        "peak_required_torque_nm": float(np.max(np.abs(required))),
        "p95_required_torque_nm": float(np.percentile(np.abs(required), 95)),
        "median_required_torque_nm": float(np.median(np.abs(required))),
        "mean_required_torque_nm": float(np.mean(np.abs(required))),
        "worst_joint": names[int(np.argmax(np.max(np.abs(required), axis=0)))],
    }


def save_contact_id(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

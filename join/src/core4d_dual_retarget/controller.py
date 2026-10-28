"""Balance-layer helpers for PD tracking: yaw-frame CoM error and ankle/hip torques."""

from __future__ import annotations

import numpy as np


def yaw_from_xmat(xmat: np.ndarray) -> float:
    rotation = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def world_xy_to_yaw_frame(xy: np.ndarray, yaw: float) -> np.ndarray:
    x, y = np.asarray(xy, dtype=np.float64)[:2]
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.array([cosine * x + sine * y, -sine * x + cosine * y], dtype=np.float64)


def foot_translational_jacobian(model: object, data: object, prefix: str) -> np.ndarray:
    """Stacked 3D Jacobians of both ankle roll links at the ground (z=0)."""
    import mujoco

    blocks = []
    for side in ("left", "right"):
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + f"{side}_ankle_roll_link"
        )
        point = np.array(data.xpos[body_id], dtype=np.float64)
        point[2] = 0.0
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, point, body_id)
        blocks.append(jacp)
    return np.vstack(blocks)


def solve_foot_forces(jacobian_base: np.ndarray, residual_base: np.ndarray) -> np.ndarray:
    mapping = np.asarray(jacobian_base, dtype=np.float64)
    residual = np.asarray(residual_base, dtype=np.float64).reshape(6)
    forces, *_ = np.linalg.lstsq(mapping.T, residual, rcond=None)
    return forces.reshape(6)


def clip_unilateral_normals(forces: np.ndarray) -> np.ndarray:
    """Zero tensile (negative) world-z foot normals; leave tangents unchanged."""
    clipped = np.asarray(forces, dtype=np.float64).reshape(6).copy()
    clipped[2] = max(float(clipped[2]), 0.0)
    clipped[5] = max(float(clipped[5]), 0.0)
    return clipped


def clip_contact_frame_forces(forces: np.ndarray, mu: float) -> np.ndarray:
    """Clip stacked [fn, ft1, ft2] blocks to unilateral Coulomb friction."""
    packed = np.asarray(forces, dtype=np.float64).reshape(-1, 3).copy()
    for row in packed:
        row[0] = max(float(row[0]), 0.0)
        tangent = float(np.hypot(row[1], row[2]))
        limit = mu * float(row[0])
        if tangent > limit + 1e-9:
            row[1:] *= limit / tangent
    return packed.reshape(-1)


def _body_has_prefix(model: object, body_id: int, prefix: str) -> bool:
    import mujoco

    current = int(body_id)
    while current > 0:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, current) or ""
        if name.startswith(prefix):
            return True
        current = int(model.body_parentid[current])
    return False


def _geom_is_floor(model: object, geom_id: int) -> bool:
    import mujoco

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""
    if name == "floor":
        return True
    return int(model.geom_bodyid[geom_id]) == 0 and int(model.geom_type[geom_id]) == int(
        mujoco.mjtGeom.mjGEOM_PLANE
    )


def _geom_is_sole(model: object, geom_id: int) -> bool:
    import mujoco

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""
    return name.endswith("_sole")


def _model_has_soles(model: object) -> bool:
    import mujoco

    for geom_id in range(int(model.ngeom)):
        if _geom_is_sole(model, geom_id):
            return True
    return False


def lift_roots_out_of_penetration(
    model: object,
    data: object,
    prefixes: tuple[str, ...] = ("r1_", "r2_"),
    margin: float = 0.002,
    iterations: int = 5,
    soles_only: bool = False,
) -> dict[str, float]:
    """Raise floating bases until floor contacts are no deeper than ``margin``."""
    import mujoco

    lifts = {prefix: 0.0 for prefix in prefixes}
    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        mujoco.mj_collision(model, data)
        step = {prefix: 0.0 for prefix in prefixes}
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            dist = float(contact.dist)
            if dist >= -1e-6:
                continue
            is_floor_1 = _geom_is_floor(model, int(contact.geom1))
            is_floor_2 = _geom_is_floor(model, int(contact.geom2))
            if is_floor_1 == is_floor_2:
                continue
            robot_geom = int(contact.geom2 if is_floor_1 else contact.geom1)
            if soles_only and not _geom_is_sole(model, robot_geom):
                continue
            robot_body = int(model.geom_bodyid[robot_geom])
            for prefix in prefixes:
                if _body_has_prefix(model, robot_body, prefix):
                    step[prefix] = max(step[prefix], -dist + margin)
                    break
        if max(step.values()) < 1e-5:
            break
        for prefix, lift in step.items():
            if lift <= 0.0:
                continue
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint"
            )
            data.qpos[int(model.jnt_qposadr[joint_id]) + 2] += lift
            lifts[prefix] += lift
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return lifts


def _quat_mul_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    wa, xa, ya, za = np.asarray(first, dtype=np.float64)[:4]
    wb, xb, yb, zb = np.asarray(second, dtype=np.float64)[:4]
    return np.array(
        [
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ],
        dtype=np.float64,
    )


def min_sole_center_z(model: object, data: object, prefix: str, side: str) -> float:
    """Lowest sole geom center on one foot; ankle z if that robot has no soles."""
    import mujoco

    lowest = float("inf")
    found = False
    token = f"{prefix}{side}_"
    for geom_id in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith(token) or not name.endswith("_sole"):
            continue
        found = True
        lowest = min(lowest, float(data.geom_xpos[geom_id][2]))
    if found:
        return lowest
    body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + f"{side}_ankle_roll_link"
    )
    return float(data.xpos[body][2])


def min_sole_bottom_z(model: object, data: object, prefix: str, side: str) -> float:
    """Lowest world-z of box soles on one foot; ankle z if that robot has no soles."""
    import mujoco

    lowest = float("inf")
    found = False
    token = f"{prefix}{side}_"
    for geom_id in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith(token) or not name.endswith("_sole"):
            continue
        found = True
        height = float(data.geom_xpos[geom_id][2])
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
            height -= float(np.abs(rotation[2]) @ size)
        lowest = min(lowest, height)
    if found:
        return lowest
    body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + f"{side}_ankle_roll_link"
    )
    return float(data.xpos[body][2])


def level_soles_by_root_tilt(
    model: object,
    data: object,
    prefix: str,
    *,
    iterations: int = 8,
    gain: float = 0.85,
    max_step: float = 0.12,
    tolerance: float = 0.001,
) -> float:
    """Tilt the floating base so left/right sole bottoms share a world-z.

    Pure vertical lift cannot plant a hovering foot: the low foot's penetration
    is undone by ``lift_roots_out_of_penetration``. A small roll/pitch about
    the inter-foot axis equalizes heights, then a later lift plants both.
    """
    import mujoco

    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint"
    )
    address = int(model.jnt_qposadr[joint_id])
    left = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + "left_ankle_roll_link"
    )
    right = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + "right_ankle_roll_link"
    )
    delta_z = 0.0
    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        delta_z = min_sole_center_z(model, data, prefix, "right") - min_sole_center_z(
            model, data, prefix, "left"
        )
        if abs(delta_z) < tolerance:
            break
        span = np.asarray(data.xpos[right][:2], dtype=np.float64) - np.asarray(
            data.xpos[left][:2], dtype=np.float64
        )
        distance = float(np.hypot(span[0], span[1]))
        if distance < 0.05:
            break
        axis = np.array([-span[1], span[0], 0.0], dtype=np.float64) / distance
        angle = float(np.clip(gain * delta_z / distance, -max_step, max_step))
        half = 0.5 * angle
        delta_quat = np.array(
            [np.cos(half), *(axis * np.sin(half))],
            dtype=np.float64,
        )
        data.qpos[address + 3 : address + 7] = _quat_mul_wxyz(
            delta_quat, np.asarray(data.qpos[address + 3 : address + 7], dtype=np.float64)
        )
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return float(
        min_sole_center_z(model, data, prefix, "right")
        - min_sole_center_z(model, data, prefix, "left")
    )


def plant_hovering_foot(
    model: object,
    data: object,
    prefix: str,
    *,
    iterations: int = 12,
    gain: float = 0.7,
    max_step: float = 0.06,
    tolerance: float = 0.0015,
    target_gap: float = 0.0,
) -> float:
    """IK the higher leg down to the planted sole without rolling the root.

    Root tilt equalizes feet but throws CoM off the support. Lengthening only
    the hovering hip/knee/ankle keeps the planted foot and pelvis yaw.
    ``target_gap`` leaves that much remaining sole-center height difference.
    """
    import mujoco

    left_z = min_sole_center_z(model, data, prefix, "left")
    right_z = min_sole_center_z(model, data, prefix, "right")
    if abs(right_z - left_z) < tolerance:
        return float(right_z - left_z)
    side = "right" if right_z > left_z else "left"
    other = "left" if side == "right" else "right"
    body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + f"{side}_ankle_roll_link"
    )
    columns: list[int] = []
    addresses: list[int] = []
    ranges: list[np.ndarray] = []
    tokens = ("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")
    for joint_id in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not name.startswith(prefix) or f"{side}_" not in name:
            continue
        if "yaw" in name or not any(token in name for token in tokens):
            continue
        columns.append(int(model.jnt_dofadr[joint_id]))
        addresses.append(int(model.jnt_qposadr[joint_id]))
        ranges.append(np.asarray(model.jnt_range[joint_id], dtype=np.float64))
    if not columns:
        return float(right_z - left_z)
    jacobian = np.zeros((3, int(model.nv)))
    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        error = (
            min_sole_center_z(model, data, prefix, side)
            - min_sole_center_z(model, data, prefix, other)
            - abs(float(target_gap))
        )
        if abs(error) < tolerance:
            break
        point = np.array(data.xpos[body], dtype=np.float64)
        mujoco.mj_jac(model, data, jacobian, None, point, body)
        mapping = jacobian[2, columns]
        scale = float(mapping @ mapping)
        if scale < 1e-10:
            break
        delta = np.clip((-gain * error) * mapping / scale, -max_step, max_step)
        for address, step, joint_range in zip(addresses, delta, ranges, strict=True):
            data.qpos[address] = float(
                np.clip(data.qpos[address] + step, joint_range[0], joint_range[1])
            )
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return float(
        min_sole_center_z(model, data, prefix, "right")
        - min_sole_center_z(model, data, prefix, "left")
    )


def hold_sole_height(
    model: object,
    data: object,
    prefix: str,
    side: str,
    z_ref: float,
    *,
    iterations: int = 12,
    gain: float = 0.7,
    max_step: float = 0.06,
    tolerance: float = 0.0015,
) -> float:
    """Raise one foot so sole-center z is at least ``z_ref``. Never lower it.

    Sagittal hip/knee/ankle only. Does not zero velocities; the caller must
    restore ``qpos``/``qvel`` if this runs mid-step.
    """
    import mujoco

    body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + f"{side}_ankle_roll_link"
    )
    columns: list[int] = []
    addresses: list[int] = []
    ranges: list[np.ndarray] = []
    tokens = ("hip_pitch", "knee", "ankle_pitch")
    side_token = f"{side}_"
    for joint_id in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not name.startswith(prefix) or side_token not in name:
            continue
        if "yaw" in name or not any(token in name for token in tokens):
            continue
        columns.append(int(model.jnt_dofadr[joint_id]))
        addresses.append(int(model.jnt_qposadr[joint_id]))
        ranges.append(np.asarray(model.jnt_range[joint_id], dtype=np.float64))
    if not columns:
        return min_sole_center_z(model, data, prefix, side)
    jacobian = np.zeros((3, int(model.nv)))
    z_now = min_sole_center_z(model, data, prefix, side)
    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        z_now = min_sole_center_z(model, data, prefix, side)
        error = z_now - float(z_ref)
        if error >= -tolerance:
            break
        point = np.array(data.xpos[body], dtype=np.float64)
        mujoco.mj_jac(model, data, jacobian, None, point, body)
        mapping = jacobian[2, columns]
        scale = float(mapping @ mapping)
        if scale < 1e-10:
            break
        delta = np.clip((-gain * error) * mapping / scale, -max_step, max_step)
        for address, step, joint_range in zip(addresses, delta, ranges, strict=True):
            data.qpos[address] = float(
                np.clip(data.qpos[address] + step, joint_range[0], joint_range[1])
            )
    mujoco.mj_forward(model, data)
    return min_sole_center_z(model, data, prefix, side)


SWING_COM_TOKENS = (
    "hip_pitch",
    "hip_roll",
    "hip_yaw",
    "knee",
    "ankle_pitch",
    "ankle_roll",
)


def hold_com_xy(
    model: object,
    data: object,
    prefix: str,
    com_xy_ref: np.ndarray,
    *,
    swing_side: str,
    z_ref: float | None = None,
    iterations: int = 16,
    gain: float = 0.75,
    max_step: float = 0.05,
    xy_tolerance: float = 0.004,
    z_tolerance: float = 0.0015,
) -> tuple[np.ndarray, float]:
    """Move only ``swing_side`` so subtree CoM xy stays at ``com_xy_ref``.

    Optionally keep that sole-center z at least ``z_ref`` (never lower it).
    Does not zero velocities; the caller must restore ``qpos``/``qvel`` if
    this runs mid-step.
    """
    import mujoco

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
    ankle = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + f"{swing_side}_ankle_roll_link"
    )
    columns: list[int] = []
    addresses: list[int] = []
    ranges: list[np.ndarray] = []
    side_token = f"{swing_side}_"
    for joint_id in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not name.startswith(prefix) or side_token not in name:
            continue
        if not any(token in name for token in SWING_COM_TOKENS):
            continue
        columns.append(int(model.jnt_dofadr[joint_id]))
        addresses.append(int(model.jnt_qposadr[joint_id]))
        ranges.append(np.asarray(model.jnt_range[joint_id], dtype=np.float64))
    ref = np.asarray(com_xy_ref, dtype=np.float64)[:2]
    z_now = min_sole_center_z(model, data, prefix, swing_side)
    if not columns:
        return np.asarray(data.subtree_com[pelvis][:2], dtype=np.float64) - ref, z_now
    jacobian_com = np.zeros((3, int(model.nv)))
    jacobian_foot = np.zeros((3, int(model.nv)))
    com_error = np.zeros(2)
    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        com_error = np.asarray(data.subtree_com[pelvis][:2], dtype=np.float64) - ref
        z_now = min_sole_center_z(model, data, prefix, swing_side)
        rows = [com_error]
        mujoco.mj_jacSubtreeCom(model, data, jacobian_com, pelvis)
        mapping = [jacobian_com[:2, columns]]
        if z_ref is not None and z_now < float(z_ref) - z_tolerance:
            rows.append(np.array([z_now - float(z_ref)], dtype=np.float64))
            point = np.array(data.xpos[ankle], dtype=np.float64)
            mujoco.mj_jac(model, data, jacobian_foot, None, point, ankle)
            mapping.append(jacobian_foot[2:3, columns])
        if (
            float(np.linalg.norm(com_error)) < xy_tolerance
            and (z_ref is None or z_now >= float(z_ref) - z_tolerance)
        ):
            break
        jacobian = np.vstack(mapping)
        error = np.concatenate(rows)
        delta, *_ = np.linalg.lstsq(jacobian, -gain * error, rcond=None)
        delta = np.clip(delta, -max_step, max_step)
        for address, step, joint_range in zip(addresses, delta, ranges, strict=True):
            data.qpos[address] = float(
                np.clip(data.qpos[address] + step, joint_range[0], joint_range[1])
            )
    mujoco.mj_forward(model, data)
    com_error = np.asarray(data.subtree_com[pelvis][:2], dtype=np.float64) - ref
    return com_error, min_sole_center_z(model, data, prefix, swing_side)


def apply_swing_com_hold(
    model: object,
    data: object,
    prefix: str,
    names: list[str] | np.ndarray,
    target: np.ndarray,
    qpos_by_name: dict[str, int],
    swing_sides: tuple[str, ...],
    com_xy_ref: np.ndarray | None,
    z_refs: dict[str, float],
) -> np.ndarray:
    """Project swing-leg targets so CoM xy and sole z stay at the latched stand."""
    import mujoco

    if not swing_sides or com_xy_ref is None:
        return np.asarray(target, dtype=np.float64)
    mixed = np.asarray(target, dtype=np.float64).copy()
    saved_qpos = np.array(data.qpos, copy=True)
    saved_qvel = np.array(data.qvel, copy=True)
    for index, name in enumerate(names):
        address = qpos_by_name.get(str(name))
        if address is None:
            continue
        data.qpos[int(address)] = float(mixed[index])
    mujoco.mj_forward(model, data)
    for side in swing_sides:
        hold_com_xy(
            model,
            data,
            prefix,
            com_xy_ref,
            swing_side=side,
            z_ref=z_refs.get(side),
        )
    for index, name in enumerate(names):
        short = str(name)
        if joint_side(short) not in swing_sides or not is_leg_joint(short):
            continue
        address = qpos_by_name.get(short)
        if address is None:
            continue
        mixed[index] = float(data.qpos[int(address)])
    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    mujoco.mj_forward(model, data)
    return mixed


def apply_swing_sole_hold(
    model: object,
    data: object,
    prefix: str,
    names: list[str] | np.ndarray,
    target: np.ndarray,
    qpos_by_name: dict[str, int],
    swing_sides: tuple[str, ...],
    z_refs: dict[str, float],
) -> np.ndarray:
    """Project swing-leg targets so commanded sole z stays at or above ``z_refs``."""
    import mujoco

    if not swing_sides:
        return np.asarray(target, dtype=np.float64)
    mixed = np.asarray(target, dtype=np.float64).copy()
    saved_qpos = np.array(data.qpos, copy=True)
    saved_qvel = np.array(data.qvel, copy=True)
    for index, name in enumerate(names):
        address = qpos_by_name.get(str(name))
        if address is None:
            continue
        data.qpos[int(address)] = float(mixed[index])
    mujoco.mj_forward(model, data)
    for side in swing_sides:
        if side not in z_refs:
            continue
        hold_sole_height(model, data, prefix, side, z_refs[side])
    for index, name in enumerate(names):
        short = str(name)
        if joint_side(short) not in swing_sides or not is_leg_joint(short):
            continue
        address = qpos_by_name.get(short)
        if address is None:
            continue
        mixed[index] = float(data.qpos[int(address)])
    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    mujoco.mj_forward(model, data)
    return mixed


BALANCE_JOINT_TOKENS = (
    "hip_pitch",
    "hip_roll",
    "ankle_pitch",
    "ankle_roll",
    "waist_pitch",
    "waist_roll",
)
LEG_JOINT_TOKENS = ("hip", "knee", "ankle")


def peak_foot_xy_drift(foot_positions: np.ndarray) -> np.ndarray:
    """Peak planar foot travel. ``foot_positions`` is [T, robots, 2, 3]."""
    xy = np.asarray(foot_positions, dtype=np.float64)[..., :2]
    drift = xy - xy[:1]
    return np.max(np.linalg.norm(drift, axis=-1), axis=0)


def max_foot_xy_speed(
    current_xy: np.ndarray,
    previous_xy: np.ndarray,
    dt: float,
) -> float:
    """Peak planar speed of two feet. ``*_xy`` is [2, 2]."""
    delta = np.asarray(current_xy, dtype=np.float64) - np.asarray(
        previous_xy, dtype=np.float64
    )
    return float(np.max(np.linalg.norm(delta / max(float(dt), 1e-9), axis=-1)))


def slip_hip_roll_bias(
    joint_names: list[str] | np.ndarray,
    left_lateral_speed: float,
    right_lateral_speed: float,
    gain: float,
    clip: float = 0.08,
) -> dict[str, float]:
    """Oppose lateral foot skate with hip roll. Positive speed is further left."""
    if abs(float(gain)) < 1e-12:
        return {}
    extras: dict[str, float] = {}
    for name in joint_names:
        key = str(name)
        if "hip_roll" not in key or "yaw" in key:
            continue
        speed = left_lateral_speed if "left" in key else right_lateral_speed
        extras[key] = float(np.clip(-float(gain) * speed, -clip, clip))
    return extras


def is_leg_joint(name: str) -> bool:
    return any(token in name for token in LEG_JOINT_TOKENS)


def joint_side(name: str) -> str | None:
    short = str(name)
    if short.startswith("left_"):
        return "left"
    if short.startswith("right_"):
        return "right"
    return None


def planted_foot_sides(
    left_z: float,
    right_z: float,
    margin: float,
) -> tuple[str, ...]:
    """Sides whose soles are within ``margin`` of the lower sole."""
    sides: list[str] = []
    if float(left_z) <= float(right_z) + float(margin):
        sides.append("left")
    if float(right_z) <= float(left_z) + float(margin):
        sides.append("right")
    return tuple(sides)


def com_support_xy(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    *,
    mode: str = "midpoint",
    planted_sides: tuple[str, ...] | None = None,
    left_z: float | None = None,
    right_z: float | None = None,
    margin: float = 0.008,
) -> np.ndarray:
    """Planar support point for CoM error. ``planted`` uses the loaded foot."""
    left = np.asarray(left_xy, dtype=np.float64)[:2]
    right = np.asarray(right_xy, dtype=np.float64)[:2]
    kind = str(mode)
    if kind == "left":
        return left.copy()
    if kind == "right":
        return right.copy()
    if kind == "planted":
        sides = planted_sides
        if not sides and left_z is not None and right_z is not None:
            # Always the lower sole so a sub-margin hover cannot flicker
            # between midpoint and single-support.
            if float(left_z) <= float(right_z):
                return left.copy()
            return right.copy()
        if sides == ("left",):
            return left.copy()
        if sides == ("right",):
            return right.copy()
    return 0.5 * (left + right)


def is_arm_joint(name: str) -> bool:
    return any(token in name for token in ("shoulder", "elbow", "wrist"))


def blend_arm_targets(
    names: list[str] | np.ndarray,
    target: np.ndarray,
    frame0: np.ndarray,
    weight: float,
) -> np.ndarray:
    """weight=0 keeps frame-0 arms; weight=1 is the clip arms."""
    mixed = np.asarray(target, dtype=np.float64).copy()
    start = np.asarray(frame0, dtype=np.float64)
    amount = float(np.clip(weight, 0.0, 1.0))
    for index, name in enumerate(names):
        if is_arm_joint(str(name)):
            mixed[index] = (1.0 - amount) * start[index] + amount * mixed[index]
    return mixed


def blend_leg_targets(
    names: list[str] | np.ndarray,
    target: np.ndarray,
    frame0: np.ndarray,
    weight: float,
) -> np.ndarray:
    """weight=0 keeps frame-0 legs; weight=1 is the clip legs."""
    mixed = np.asarray(target, dtype=np.float64).copy()
    start = np.asarray(frame0, dtype=np.float64)
    amount = float(np.clip(weight, 0.0, 1.0))
    for index, name in enumerate(names):
        if is_leg_joint(str(name)):
            mixed[index] = (1.0 - amount) * start[index] + amount * mixed[index]
    return mixed


def override_joint_targets(
    names: list[str] | np.ndarray,
    target: np.ndarray,
    overrides: dict[str, float],
) -> np.ndarray:
    """Replace named joint targets. Keys are short joint names."""
    if not overrides:
        return np.asarray(target, dtype=np.float64)
    replaced = np.asarray(target, dtype=np.float64).copy()
    for index, name in enumerate(names):
        key = str(name)
        if key in overrides:
            replaced[index] = float(overrides[key])
    return replaced


def freeze_leg_targets(
    names: list[str] | np.ndarray,
    target: np.ndarray,
    stance: np.ndarray,
    *,
    freeze_legs: bool = True,
    freeze_waist: bool = False,
    freeze_ankles: bool = False,
    freeze_sides: tuple[str, ...] | None = None,
    freeze_swing_sagittal: bool = False,
) -> np.ndarray:
    """Keep selected stance joints at the standable first-frame pose."""
    frozen = np.asarray(target, dtype=np.float64).copy()
    stance_values = np.asarray(stance, dtype=np.float64)
    planted = tuple(freeze_sides) if freeze_sides else ()
    swing = tuple(side for side in ("left", "right") if side not in planted)
    for index, name in enumerate(names):
        short = str(name)
        side = joint_side(short)
        freeze_sagittal = bool(
            freeze_swing_sagittal
            and planted
            and side in swing
            and any(token in short for token in ("hip_pitch", "knee", "ankle_pitch"))
        )
        if (
            (freeze_legs and is_leg_joint(short))
            or (freeze_waist and "waist" in short)
            or (freeze_ankles and "ankle" in short)
            or (planted and is_leg_joint(short) and side in planted)
            or freeze_sagittal
        ):
            frozen[index] = stance_values[index]
    return frozen


SOLE_OFFSET = np.array([0.035, 0.0, -0.028], dtype=np.float64)


def _sole_world_point(data: object, ankle_body: int) -> np.ndarray:
    rotation = np.asarray(data.xmat[ankle_body], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(data.xpos[ankle_body], dtype=np.float64)
    return origin + rotation @ SOLE_OFFSET


def project_com_onto_support(
    model: object,
    data: object,
    prefix: str,
    *,
    iterations: int = 5,
    gain: float = 0.9,
    max_step: float = 0.3,
) -> np.ndarray:
    """Adjust stance joints so the subtree CoM sits over the ankle midpoint."""
    import mujoco

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
    left = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + "left_ankle_roll_link"
    )
    right = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + "right_ankle_roll_link"
    )
    columns: list[int] = []
    addresses: list[int] = []
    ranges: list[np.ndarray] = []
    for joint_id in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not name.startswith(prefix):
            continue
        if "yaw" in name or not any(token in name for token in BALANCE_JOINT_TOKENS):
            continue
        columns.append(int(model.jnt_dofadr[joint_id]))
        addresses.append(int(model.jnt_qposadr[joint_id]))
        ranges.append(np.asarray(model.jnt_range[joint_id], dtype=np.float64))
    error = np.zeros(2)
    if not columns:
        return error
    jacobian_com = np.zeros((3, model.nv))
    jacobian_left = np.zeros((3, model.nv))
    jacobian_right = np.zeros((3, model.nv))
    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSubtreeCom(model, data, jacobian_com, pelvis)
        mujoco.mj_jac(
            model, data, jacobian_left, None, np.array(data.xpos[left], dtype=np.float64), left
        )
        mujoco.mj_jac(
            model, data, jacobian_right, None, np.array(data.xpos[right], dtype=np.float64), right
        )
        support = 0.5 * (
            np.asarray(data.xpos[left][:2]) + np.asarray(data.xpos[right][:2])
        )
        com = np.asarray(data.subtree_com[pelvis][:2], dtype=np.float64)
        error = com - support
        if float(np.linalg.norm(error)) < 0.005:
            break
        relative = (
            jacobian_com[:2, columns]
            - 0.5 * (jacobian_left[:2, columns] + jacobian_right[:2, columns])
        )
        delta, *_ = np.linalg.lstsq(relative, -gain * error, rcond=None)
        delta = np.clip(delta, -max_step, max_step)
        for address, step, joint_range in zip(addresses, delta, ranges, strict=True):
            data.qpos[address] = float(
                np.clip(data.qpos[address] + step, joint_range[0], joint_range[1])
            )
    mujoco.mj_forward(model, data)
    return error


def shorten_stance_split(
    model: object,
    data: object,
    prefix: str,
    *,
    target_split: float = 0.24,
    axis: str = "lateral",
    iterations: int = 6,
    gain: float = 0.6,
    max_step: float = 0.08,
) -> float:
    """IK both feet toward each other along pelvis left (lateral) or forward."""
    import mujoco

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
    left = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + "left_ankle_roll_link"
    )
    right = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + "right_ankle_roll_link"
    )
    columns: list[int] = []
    addresses: list[int] = []
    ranges: list[np.ndarray] = []
    for joint_id in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not name.startswith(prefix):
            continue
        if "yaw" in name or not any(token in name for token in LEG_JOINT_TOKENS):
            continue
        columns.append(int(model.jnt_dofadr[joint_id]))
        addresses.append(int(model.jnt_qposadr[joint_id]))
        ranges.append(np.asarray(model.jnt_range[joint_id], dtype=np.float64))
    split = 0.0
    if not columns:
        return split
    jacobian_left = np.zeros((3, model.nv))
    jacobian_right = np.zeros((3, model.nv))

    def _axis(yaw: float) -> np.ndarray:
        cosine = float(np.cos(yaw))
        sine = float(np.sin(yaw))
        if axis == "sagittal":
            return np.array([cosine, sine], dtype=np.float64)
        return np.array([-sine, cosine], dtype=np.float64)

    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        direction = _axis(yaw_from_xmat(data.xmat[pelvis]))
        left_xy = np.asarray(data.xpos[left][:2], dtype=np.float64)
        right_xy = np.asarray(data.xpos[right][:2], dtype=np.float64)
        split = float(np.dot(left_xy - right_xy, direction))
        excess = abs(split) - float(target_split)
        if excess <= 0.01:
            break
        reduce = np.sign(split) * excess
        desired = np.concatenate((-0.5 * reduce * direction, 0.5 * reduce * direction))
        mujoco.mj_jac(
            model, data, jacobian_left, None, np.array(data.xpos[left], dtype=np.float64), left
        )
        mujoco.mj_jac(
            model,
            data,
            jacobian_right,
            None,
            np.array(data.xpos[right], dtype=np.float64),
            right,
        )
        stacked = np.vstack(
            (jacobian_left[:2, columns], jacobian_right[:2, columns])
        )
        delta, *_ = np.linalg.lstsq(stacked, gain * desired, rcond=None)
        delta = np.clip(delta, -max_step, max_step)
        for address, step, joint_range in zip(addresses, delta, ranges, strict=True):
            data.qpos[address] = float(
                np.clip(data.qpos[address] + step, joint_range[0], joint_range[1])
            )
    mujoco.mj_forward(model, data)
    direction = _axis(yaw_from_xmat(data.xmat[pelvis]))
    split = float(
        np.dot(
            np.asarray(data.xpos[left][:2]) - np.asarray(data.xpos[right][:2]),
            direction,
        )
    )
    return split


def shorten_sagittal_stagger(
    model: object,
    data: object,
    prefix: str,
    *,
    target_split: float = 0.22,
    iterations: int = 6,
    gain: float = 0.6,
    max_step: float = 0.08,
) -> float:
    return shorten_stance_split(
        model,
        data,
        prefix,
        target_split=target_split,
        axis="lateral",
        iterations=iterations,
        gain=gain,
        max_step=max_step,
    )


def balance_joint_offsets(
    model: object,
    data: object,
    prefix: str,
    *,
    iterations: int = 2,
    gain: float = 0.75,
    max_step: float = 0.18,
    deadband: float = 0.02,
) -> dict[str, float]:
    """CoM-support joint deltas from the current pose, without changing simulator state."""
    import mujoco

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
    left = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + "left_ankle_roll_link"
    )
    right = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + "right_ankle_roll_link"
    )
    support = 0.5 * (
        np.asarray(data.xpos[left][:2]) + np.asarray(data.xpos[right][:2])
    )
    com = np.asarray(data.subtree_com[pelvis][:2], dtype=np.float64)
    if float(np.linalg.norm(com - support)) < deadband:
        return {}
    saved_qpos = np.array(data.qpos, copy=True)
    saved_qvel = np.array(data.qvel, copy=True)
    project_com_onto_support(
        model, data, prefix, iterations=iterations, gain=gain, max_step=max_step
    )
    offsets: dict[str, float] = {}
    for joint_id in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not name.startswith(prefix):
            continue
        if "yaw" in name or not any(token in name for token in BALANCE_JOINT_TOKENS):
            continue
        address = int(model.jnt_qposadr[joint_id])
        short = name[len(prefix) :]
        offsets[short] = float(
            np.clip(data.qpos[address] - saved_qpos[address], -max_step, max_step)
        )
    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    mujoco.mj_forward(model, data)
    return offsets


def project_target_joints(
    model: object,
    data: object,
    prefix: str,
    names: list[str] | np.ndarray,
    targets: np.ndarray,
    *,
    iterations: int = 3,
    gain: float = 0.8,
    max_step: float = 0.08,
) -> np.ndarray:
    """CoM-project a desired joint vector on the current floating base, then restore state."""
    import mujoco

    saved_qpos = np.array(data.qpos, copy=True)
    saved_qvel = np.array(data.qvel, copy=True)
    result = np.asarray(targets, dtype=np.float64).copy()
    for index, name in enumerate(names):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, prefix + str(name)
        )
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(result[index])
    project_com_onto_support(
        model, data, prefix, iterations=iterations, gain=gain, max_step=max_step
    )
    for index, name in enumerate(names):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, prefix + str(name)
        )
        result[index] = float(data.qpos[int(model.jnt_qposadr[joint_id])])
    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    mujoco.mj_forward(model, data)
    return result


def simulator_contact_jacobian(
    model: object,
    data: object,
    prefix: str,
) -> np.ndarray | None:
    """Stacked contact-frame Jacobians of floor–robot contacts on one robot."""
    import mujoco

    if int(data.ncon) == 0:
        mujoco.mj_collision(model, data)
    blocks = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        is_floor_1 = _geom_is_floor(model, int(contact.geom1))
        is_floor_2 = _geom_is_floor(model, int(contact.geom2))
        if is_floor_1 == is_floor_2:
            continue
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        robot_body = body2 if is_floor_1 else body1
        if not _body_has_prefix(model, robot_body, prefix):
            continue
        robot_is_first = not is_floor_1
        sign = -1.0 if robot_is_first else 1.0
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(
            model,
            data,
            jacp,
            None,
            np.array(contact.pos, dtype=np.float64),
            robot_body,
        )
        normal = np.asarray(contact.frame[:3], dtype=np.float64)
        tangent1 = np.asarray(contact.frame[3:6], dtype=np.float64)
        tangent2 = np.asarray(contact.frame[6:9], dtype=np.float64)
        axes = np.stack([normal, tangent1, tangent2], axis=1)
        blocks.append(sign * axes.T @ jacp)
    if not blocks:
        return None
    return np.vstack(blocks)


def solve_unilateral_contact_forces(
    jacobian_base: np.ndarray,
    residual_base: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Least-squares contact-frame forces with compressive normals, then Coulomb clip."""
    from scipy.optimize import lsq_linear

    mapping = np.asarray(jacobian_base, dtype=np.float64)
    residual = np.asarray(residual_base, dtype=np.float64).reshape(6)
    nforce = mapping.shape[0]
    lower = np.full(nforce, -np.inf)
    upper = np.full(nforce, np.inf)
    lower[0::3] = 0.0
    result = lsq_linear(mapping.T, residual, bounds=(lower, upper))
    forces = np.asarray(result.x, dtype=np.float64)
    if not np.isfinite(forces).all():
        forces, *_ = np.linalg.lstsq(mapping.T, residual, rcond=None)
    return clip_contact_frame_forces(forces, mu)


def contact_consistent_joint_torques(
    model: object,
    data: object,
    prefix: str,
    bias: np.ndarray,
    *,
    source: str = "virtual",
    mu: float = 0.8,
) -> tuple[np.ndarray, int]:
    """Gravity compensation using virtual feet or simulator contacts.

    Returns joint-space bias compensation for all ``nv`` dofs and the number of
    simulator contacts used (0 means the virtual two-foot model).
    """
    import mujoco

    free_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint"
    )
    dof_adr = int(model.jnt_dofadr[free_id])
    jacobian = None
    n_contacts = 0
    if source == "sim":
        jacobian = simulator_contact_jacobian(model, data, prefix)
        if jacobian is not None:
            n_contacts = int(jacobian.shape[0] // 3)
    if jacobian is None or jacobian.size == 0:
        jacobian = foot_translational_jacobian(model, data, prefix)
        forces = clip_unilateral_normals(
            solve_foot_forces(jacobian[:, dof_adr : dof_adr + 6], bias[dof_adr : dof_adr + 6])
        )
        return np.asarray(bias, dtype=np.float64) - jacobian.T @ forces, 0
    base_jac = jacobian[:, dof_adr : dof_adr + 6]
    forces = solve_unilateral_contact_forces(
        base_jac, bias[dof_adr : dof_adr + 6], mu
    )
    return np.asarray(bias, dtype=np.float64) - jacobian.T @ forces, n_contacts


def gravity_bias_zero_velocity(model: object, data: object) -> np.ndarray:
    """Bias forces with Coriolis removed: gravity (and other configuration terms)."""
    import mujoco

    saved = np.array(data.qvel, copy=True)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    bias = np.array(data.qfrc_bias, copy=True)
    data.qvel[:] = saved
    mujoco.mj_forward(model, data)
    return bias


def subtree_com_wrench_torques(
    model: object,
    data: object,
    body_id: int,
    support_xy: np.ndarray,
    desired_z: float,
    kp_xy: float,
    kd_xy: float,
    kp_z: float,
) -> np.ndarray:
    """Joint-space torques from a Cartesian CoM PD wrench (including floating-base dofs)."""
    import mujoco

    jacobian = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, jacobian, body_id)
    com = np.asarray(data.subtree_com[body_id], dtype=np.float64)
    velocity = np.asarray(data.subtree_linvel[body_id], dtype=np.float64)
    support = np.asarray(support_xy, dtype=np.float64)[:2]
    wrench = np.array(
        [
            -kp_xy * (com[0] - support[0]) - kd_xy * velocity[0],
            -kp_xy * (com[1] - support[1]) - kd_xy * velocity[1],
            -kp_z * (com[2] - desired_z) - kd_xy * velocity[2],
        ],
        dtype=np.float64,
    )
    return jacobian.T @ wrench


def com_balance_joint_deltas(
    joint_names: list[str],
    error_yaw: np.ndarray,
    velocity_yaw: np.ndarray,
    kp: float,
    kd: float,
    hip_gain: float = 0.4,
) -> dict[str, float]:
    """Ankle strategy plus a weaker hip strategy in the pelvis yaw frame.

    Positive e_x means CoM is ahead of the support midpoint. Pitch joints use
    axis +Y; a restoring plantarflexion command is therefore negative.
    """
    error_x, error_y = np.asarray(error_yaw, dtype=np.float64)[:2]
    velocity_x, velocity_y = np.asarray(velocity_yaw, dtype=np.float64)[:2]
    pitch = float(-kp * error_x - kd * velocity_x)
    roll = float(kp * error_y + kd * velocity_y)
    extras: dict[str, float] = {}
    for name in joint_names:
        key = str(name)
        if "ankle_pitch" in key:
            extras[key] = pitch
        elif "ankle_roll" in key:
            extras[key] = roll if "left" in key else -roll
        elif hip_gain and "hip_pitch" in key:
            extras[key] = hip_gain * pitch
        elif hip_gain and "hip_roll" in key and "yaw" not in key:
            extras[key] = hip_gain * (roll if "left" in key else -roll)
    return extras


def pelvis_imu_yaw(data: object, pelvis: int, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis pitch/roll and angular rates in the yaw frame.

    Positive pitch is the torso tilting forward; positive roll is left-ear down.
    """
    rotation = np.asarray(data.xmat[pelvis], dtype=np.float64).reshape(3, 3)
    up = rotation[:, 2]
    cosine = float(np.cos(yaw))
    sine = float(np.sin(yaw))
    forward = np.array([cosine, sine], dtype=np.float64)
    left = np.array([-sine, cosine], dtype=np.float64)
    pitch = float(np.arctan2(float(np.dot(up[:2], forward)), float(up[2])))
    roll = float(np.arctan2(float(np.dot(up[:2], left)), float(up[2])))
    omega = np.asarray(data.cvel, dtype=np.float64).reshape(-1, 6)[int(pelvis), :3]
    pitch_rate = float(omega[0] * -sine + omega[1] * cosine)
    roll_rate = float(omega[0] * cosine + omega[1] * sine)
    return np.array([pitch, roll], dtype=np.float64), np.array(
        [pitch_rate, roll_rate], dtype=np.float64
    )


def imu_ankle_residuals(
    joint_names: list[str] | np.ndarray,
    tilt: np.ndarray,
    rates: np.ndarray,
    gains: tuple[float, float, float, float],
    *,
    hip_gain: float = 0.0,
    clip: float = 0.12,
    apply_to: str = "ankle",
) -> dict[str, float]:
    """Linear IMU residual. ``apply_to`` is ankle, hip, or both."""
    kp_pitch, kd_pitch, kp_roll, kd_roll = (float(value) for value in gains)
    pitch = float(-kp_pitch * tilt[0] - kd_pitch * rates[0])
    roll = float(-kp_roll * tilt[1] - kd_roll * rates[1])
    use_ankle = apply_to in ("ankle", "both")
    use_hip = apply_to in ("hip", "both")
    extras: dict[str, float] = {}
    for name in joint_names:
        key = str(name)
        if use_ankle and "ankle_pitch" in key:
            extras[key] = pitch
        elif use_ankle and "ankle_roll" in key:
            extras[key] = roll if "left" in key else -roll
        elif use_hip and "hip_pitch" in key:
            extras[key] = pitch if apply_to == "hip" else hip_gain * pitch
        elif use_hip and "hip_roll" in key and "yaw" not in key:
            value = roll if "left" in key else -roll
            extras[key] = value if apply_to == "hip" else hip_gain * value
    return {key: float(np.clip(value, -clip, clip)) for key, value in extras.items()}


def hip_pitch_target_bias(
    joint_names: list[str] | np.ndarray,
    bias: float,
) -> dict[str, float]:
    """Constant hip-pitch target offset. Positive moves the kinematic CoM forward."""
    if abs(float(bias)) < 1e-12:
        return {}
    return {
        str(name): float(bias)
        for name in joint_names
        if "hip_pitch" in str(name)
    }


def hip_roll_target_bias(
    joint_names: list[str] | np.ndarray,
    bias: float,
) -> dict[str, float]:
    """Pelvis-roll hip command. Left and right hips get opposite signs."""
    if abs(float(bias)) < 1e-12:
        return {}
    extras: dict[str, float] = {}
    for name in joint_names:
        key = str(name)
        if "hip_roll" not in key or "yaw" in key:
            continue
        extras[key] = float(bias) if "left" in key else -float(bias)
    return extras


def ankle_roll_target_bias(
    joint_names: list[str] | np.ndarray,
    bias: float,
) -> dict[str, float]:
    """Ankle-roll CoM lever. Left and right ankles get opposite signs."""
    if abs(float(bias)) < 1e-12:
        return {}
    extras: dict[str, float] = {}
    for name in joint_names:
        key = str(name)
        if "ankle_roll" not in key:
            continue
        extras[key] = float(bias) if "left" in key else -float(bias)
    return extras

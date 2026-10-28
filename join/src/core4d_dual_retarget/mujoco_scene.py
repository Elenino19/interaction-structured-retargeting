"""Build and render a dual-G1 MuJoCo scene from retargeted trajectories."""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

from .coordinates import CORE4D_TO_ROBOT, poses_core4d_to_robot


def _prefix_names(element: ET.Element, prefix: str) -> None:
    for node in element.iter():
        if "name" in node.attrib:
            node.set("name", prefix + node.get("name", ""))


def _add_stance_foot_boxes(root: ET.Element, *, stiff: bool = False) -> None:
    """Add box soles, disable official 5 mm spheres, and optionally stiffen contact."""
    contact = {
        "friction": "1.8 0.4 0.01",
        "solref": "0.004 1",
        "solimp": "0.9 0.95 0.001 0.5 2",
        "condim": "3",
    }
    for body in root.iter("body"):
        name = body.get("name", "")
        if not name.endswith("ankle_roll_link"):
            continue
        for geom in body:
            if geom.tag != "geom":
                continue
            if geom.get("type", "sphere") == "sphere" and geom.get("size") == "0.005":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        sole_attrs = {
            "name": f"{name}_sole",
            "type": "box",
            "size": "0.11 0.04 0.012",
            "pos": "0.035 0 -0.028",
            "friction": contact["friction"],
            "rgba": "0.15 0.15 0.15 1",
        }
        if stiff:
            sole_attrs.update(
                {
                    "solref": contact["solref"],
                    "solimp": contact["solimp"],
                    "condim": contact["condim"],
                }
            )
        ET.SubElement(body, "geom", **sole_attrs)
    for geom in root.iter("geom"):
        if geom.get("name") == "floor":
            geom.set("friction", contact["friction"])
            if stiff:
                geom.set("solref", contact["solref"])
                geom.set("solimp", contact["solimp"])
                geom.set("condim", contact["condim"])


def _add_prefix_sticky_floor(
    root: ET.Element, prefix: str, bit: str, *, condim: str = "6"
) -> None:
    """condim-6 plane that only collides with one robot's soles."""
    contact = {
        "friction": "1.8 0.4 0.01",
        "solref": "0.004 1",
        "solimp": "0.9 0.95 0.001 0.5 2",
    }
    worldbody = None
    for node in root.findall("worldbody"):
        worldbody = node
        break
    if worldbody is None:
        return
    ET.SubElement(
        worldbody,
        "geom",
        name=f"floor_{prefix}sticky",
        type="plane",
        size="0 0 0.05",
        friction=contact["friction"],
        solref=contact["solref"],
        solimp=contact["solimp"],
        condim=condim,
        contype=bit,
        conaffinity=bit,
        rgba="0.2 0.2 0.2 0",
    )
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if name.startswith(prefix) and name.endswith("_sole"):
            geom.set("contype", bit)
            geom.set("conaffinity", bit)


def _add_r1_sticky_floor(root: ET.Element) -> None:
    _add_prefix_sticky_floor(root, "r1_", "2")


def _set_stance_option(root: ET.Element) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        compiler = root.find("compiler")
        index = list(root).index(compiler) + 1 if compiler is not None else 0
        root.insert(index, option)
    option.set("integrator", "implicitfast")
    option.set("impratio", "20")
    option.set("cone", "elliptic")
    option.set("noslip_iterations", "3")


def _actuator_position_kp(name: str, leg_kp: float, arm_kp: float) -> float:
    if any(token in name for token in ("hip", "knee", "ankle")):
        return leg_kp
    if "waist" in name:
        return max(arm_kp, 80.0)
    if "wrist" in name:
        return max(arm_kp * 0.4, 12.0)
    return arm_kp


def _convert_motors_to_position(
    root: ET.Element, *, leg_kp: float, arm_kp: float
) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        return
    for motor in actuator:
        name = motor.get("name", "")
        motor.tag = "position"
        motor.set("kp", f"{_actuator_position_kp(name, leg_kp, arm_kp):g}")
        motor.set("dampratio", "1")


def build_dual_g1_scene_xml(
    source_xml: Path,
    object_mesh: Path,
    output_xml: Path,
    *,
    weld_feet: bool = False,
    foot_boxes: bool = False,
    position_actuators: bool = False,
    stiff_contacts: bool = False,
    leg_kp: float = 200.0,
    arm_kp: float = 40.0,
    r1_sticky_floor: bool = False,
    r2_sticky_floor: bool = False,
    r2_sticky_condim: int = 6,
) -> Path:
    """Duplicate the official G1 model with namespaced joints and one mocap object."""
    root = ET.parse(source_xml).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Official G1 MJCF has no compiler element")
    compiler.set("meshdir", str((source_xml.parent / "meshes").resolve()))

    robot_worldbody = next(
        (
            worldbody
            for worldbody in root.findall("worldbody")
            if worldbody.find("./body[@name='pelvis']") is not None
        ),
        None,
    )
    if robot_worldbody is None:
        raise ValueError("Could not locate G1 pelvis body")
    original_pelvis = robot_worldbody.find("./body[@name='pelvis']")
    if original_pelvis is None:
        raise ValueError("Could not locate G1 pelvis body")
    robot_worldbody.remove(original_pelvis)

    for prefix in ("r1_", "r2_"):
        body = copy.deepcopy(original_pelvis)
        _prefix_names(body, prefix)
        robot_worldbody.append(body)

    actuator = root.find("actuator")
    if actuator is not None:
        original_actuators = list(actuator)
        actuator.clear()
        for prefix in ("r1_", "r2_"):
            for original in original_actuators:
                motor = copy.deepcopy(original)
                if "name" in motor.attrib:
                    motor.set("name", prefix + motor.get("name", ""))
                if "joint" in motor.attrib:
                    motor.set("joint", prefix + motor.get("joint", ""))
                actuator.append(motor)

    sensor = root.find("sensor")
    if sensor is not None:
        root.remove(sensor)

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1280")
    global_visual.set("offheight", "960")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "mesh",
        name="shared_object_mesh",
        file=str(object_mesh.resolve()),
    )
    object_body = ET.SubElement(robot_worldbody, "body", name="shared_object", mocap="true")
    object_rotation = Rotation.from_matrix(CORE4D_TO_ROBOT).as_quat()
    object_quaternion = (
        object_rotation[3],
        object_rotation[0],
        object_rotation[1],
        object_rotation[2],
    )
    ET.SubElement(
        object_body,
        "geom",
        name="shared_object_geom",
        type="mesh",
        mesh="shared_object_mesh",
        quat=" ".join(f"{value:.12g}" for value in object_quaternion),
        rgba="0.25 0.75 0.30 0.85",
        contype="0",
        conaffinity="0",
    )

    if weld_feet:
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        for prefix in ("r1_", "r2_"):
            for side in ("left", "right"):
                ET.SubElement(
                    equality,
                    "weld",
                    name=f"{prefix}{side}_foot_weld",
                    body1=f"{prefix}{side}_ankle_roll_link",
                    solref="0.004 1",
                    solimp="0.9 0.95 0.001",
                )

    if foot_boxes:
        _add_stance_foot_boxes(root, stiff=stiff_contacts or position_actuators)
        if r1_sticky_floor:
            _add_prefix_sticky_floor(root, "r1_", "2")
        if r2_sticky_floor:
            _add_prefix_sticky_floor(
                root, "r2_", "4", condim=str(r2_sticky_condim)
            )
    if position_actuators:
        _set_stance_option(root)
        _convert_motors_to_position(root, leg_kp=leg_kp, arm_kp=arm_kp)
    elif stiff_contacts:
        _set_stance_option(root)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_xml, encoding="unicode")
    return output_xml


def _set_robot_qpos(
    model: object,
    data: object,
    prefix: str,
    joint_names: np.ndarray,
    joint_positions: np.ndarray,
    root_translation: np.ndarray,
    root_yaw: float,
) -> None:
    import mujoco

    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
    free_address = model.jnt_qposadr[free_id]
    data.qpos[free_address : free_address + 3] = root_translation
    data.qpos[free_address + 3 : free_address + 7] = [
        np.cos(root_yaw / 2),
        0,
        0,
        np.sin(root_yaw / 2),
    ]
    for name, position in zip(joint_names, joint_positions, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + str(name))
        data.qpos[model.jnt_qposadr[joint_id]] = position


def render_dual_g1_video(
    scene_xml: Path,
    object_poses_core4d: np.ndarray,
    robot1_path: Path,
    robot2_path: Path,
    output_path: Path,
    *,
    frame_step: int = 1,
    source_fps: float = 15.0,
    width: int = 960,
    height: int = 720,
) -> None:
    """Render actual G1 meshes and the shared object from kinematic trajectories."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    with np.load(robot1_path) as robot1, np.load(robot2_path) as robot2:
        trajectories = {
            1: {name: robot1[name] for name in robot1.files},
            2: {name: robot2[name] for name in robot2.files},
        }
    frame_count = min(
        len(object_poses_core4d),
        len(trajectories[1]["joint_positions"]),
        len(trajectories[2]["joint_positions"]),
    )
    object_poses = poses_core4d_to_robot(object_poses_core4d[:frame_count])
    object_mocap_id = model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shared_object")
    ]

    all_centers = np.concatenate(
        [
            trajectories[1]["root_translation"],
            trajectories[2]["root_translation"],
            object_poses[:, :3, 3],
        ]
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.mean(all_centers, axis=0)
    camera.distance = max(np.ptp(all_centers[:, :2], axis=0).max() * 1.8, 3.0)
    camera.azimuth = 135
    camera.elevation = -22

    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        with imageio.get_writer(
            output_path,
            fps=source_fps / frame_step,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            for frame in range(0, frame_count, frame_step):
                for person, prefix in ((1, "r1_"), (2, "r2_")):
                    trajectory = trajectories[person]
                    _set_robot_qpos(
                        model,
                        data,
                        prefix,
                        trajectory["joint_names"],
                        trajectory["joint_positions"][frame],
                        trajectory["root_translation"][frame],
                        float(trajectory["root_yaw"][frame]),
                    )
                data.mocap_pos[object_mocap_id] = object_poses[frame, :3, 3]
                quaternion_xyzw = Rotation.from_matrix(
                    object_poses[frame, :3, :3]
                ).as_quat()
                data.mocap_quat[object_mocap_id] = quaternion_xyzw[[3, 0, 1, 2]]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        renderer.close()


def render_pd_tracking_video(
    scene_xml: Path,
    object_poses_core4d: np.ndarray,
    pd_trajectory_path: Path,
    robot1_reference_path: Path,
    robot2_reference_path: Path,
    output_path: Path,
    *,
    source_fps: float = 15.0,
    width: int = 960,
    height: int = 720,
) -> None:
    """Render recorded free-base PD states, including falls and foot slip."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    with np.load(pd_trajectory_path) as payload:
        actual_joints = payload["actual_joint_positions"]
        actual_roots = payload["actual_root_translation"]
        actual_quaternions = payload["actual_root_quaternion"]
    references = {}
    for person, path in ((1, robot1_reference_path), (2, robot2_reference_path)):
        with np.load(path) as payload:
            references[person] = payload["joint_names"]

    frame_count = min(len(actual_joints), len(object_poses_core4d))
    object_poses = poses_core4d_to_robot(object_poses_core4d[:frame_count])
    object_mocap_id = model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shared_object")
    ]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.mean(object_poses[:, :3, 3], axis=0)
    camera.distance = 3.5
    camera.azimuth = 135
    camera.elevation = -22

    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        with imageio.get_writer(
            output_path,
            fps=source_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            for frame in range(frame_count):
                for person, prefix in ((1, "r1_"), (2, "r2_")):
                    free_id = mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        prefix + "floating_base_joint",
                    )
                    free_address = model.jnt_qposadr[free_id]
                    data.qpos[free_address : free_address + 3] = actual_roots[frame, person - 1]
                    data.qpos[free_address + 3 : free_address + 7] = actual_quaternions[
                        frame, person - 1
                    ]
                    for name, position in zip(
                        references[person],
                        actual_joints[frame, person - 1],
                        strict=True,
                    ):
                        joint_id = mujoco.mj_name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            prefix + str(name),
                        )
                        data.qpos[model.jnt_qposadr[joint_id]] = position

                data.mocap_pos[object_mocap_id] = object_poses[frame, :3, 3]
                quaternion = Rotation.from_matrix(object_poses[frame, :3, :3]).as_quat()
                data.mocap_quat[object_mocap_id] = quaternion[[3, 0, 1, 2]]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        renderer.close()

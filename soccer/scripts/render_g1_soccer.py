#!/usr/bin/env python3
"""Render two G1s, a mocap ball, and five static cones."""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import copy
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

from isr_assets import ensure_join_on_path, g1_urdf, g1_xml
ensure_join_on_path()

from core4d_dual_retarget.coordinates import CORE4D_TO_ROBOT, points_core4d_to_robot
from core4d_dual_retarget.mujoco_scene import _prefix_names, _set_robot_qpos


def archive_export(path: Path, tag: str = "") -> Path | None:
    """Copy an existing export aside before rewriting it."""
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    dest = archive_dir / f"{path.stem}_{stamp}{suffix}{path.suffix}"
    shutil.copy2(path, dest)
    print("archived", path, "->", dest)
    return dest


def _hide_collision_geoms(body: ET.Element) -> None:
    """Official G1 MJCF draws collision cylinders/meshes on top of the visual mesh."""
    for geom in body.iter("geom"):
        if geom.get("group", "0") == "1":
            continue
        geom.set("rgba", "0 0 0 0")
        geom.set("group", "3")


def build_scene(source_xml: Path, cones_xyz_yup: np.ndarray, output_xml: Path) -> Path:
    root = ET.parse(source_xml).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((source_xml.parent / "meshes").resolve()))
    robot_worldbody = next(
        worldbody
        for worldbody in root.findall("worldbody")
        if worldbody.find("./body[@name='pelvis']") is not None
    )
    original_pelvis = robot_worldbody.find("./body[@name='pelvis']")
    robot_worldbody.remove(original_pelvis)
    for prefix in ("r1_", "r2_"):
        body = copy.deepcopy(original_pelvis)
        _prefix_names(body, prefix)
        _hide_collision_geoms(body)
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

    ball = ET.SubElement(robot_worldbody, "body", name="ball", mocap="true")
    ET.SubElement(
        ball,
        "geom",
        name="ball_geom",
        type="sphere",
        size="0.11",
        rgba="0.95 0.95 0.95 1",
        contype="0",
        conaffinity="0",
    )
    for index, position in enumerate(
        (
            (0.096, 0.0, 0.0),
            (-0.096, 0.0, 0.0),
            (0.0, 0.096, 0.0),
            (0.0, -0.096, 0.0),
            (0.0, 0.0, 0.096),
            (0.0, 0.0, -0.096),
        )
    ):
        ET.SubElement(
            ball,
            "geom",
            name=f"ball_spot_{index}",
            type="sphere",
            size="0.019",
            pos=" ".join(f"{value:.3f}" for value in position),
            rgba="0.04 0.04 0.04 1",
            contype="0",
            conaffinity="0",
        )
    cones_robot = points_core4d_to_robot(cones_xyz_yup)
    for i, pos in enumerate(cones_robot):
        body = ET.SubElement(
            robot_worldbody,
            "body",
            name=f"cone_{i}",
            pos=f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"cone_{i}_geom",
            type="cylinder",
            size="0.06 0.05",
            rgba="1.0 0.35 0.08 1",
            contype="0",
            conaffinity="0",
        )

    for material in root.iter("material"):
        if material.get("name") == "groundplane":
            material.set("reflectance", "0.04")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1280")
    global_visual.set("offheight", "960")

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_xml, encoding="unicode")
    return output_xml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--g1-dir",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/g1"),
    )
    parser.add_argument(
        "--ball-cones",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/ball_cones.npz"),
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=g1_xml(),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/g1_ball_cones_preview.mp4"),
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--frame-step", type=int, default=1)
    args = parser.parse_args()
    guard_release_writes(args, args.g1_dir / 'soccer_scene.xml')

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    env = np.load(args.ball_cones)
    scene = args.g1_dir / "soccer_scene.xml"
    build_scene(args.xml, env["cone_xyz_yup"], scene)
    ball_robot = points_core4d_to_robot(env["ball_xyz_yup"])

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    robots = {}
    for person, name in ((1, "robot_a.npz"), (2, "robot_b.npz")):
        with np.load(args.g1_dir / name) as payload:
            robots[person] = {key: payload[key] for key in payload.files}

    frame_count = min(
        len(ball_robot),
        len(robots[1]["joint_positions"]),
        len(robots[2]["joint_positions"]),
    )
    ball_quaternions = np.zeros((frame_count, 4), dtype=np.float64)
    orientation = Rotation.identity()
    ball_quaternions[0] = [1.0, 0.0, 0.0, 0.0]
    for frame in range(1, frame_count):
        displacement = ball_robot[frame, :2] - ball_robot[frame - 1, :2]
        distance = float(np.linalg.norm(displacement))
        if distance > 1e-8:
            axis = np.array(
                [-displacement[1], displacement[0], 0.0], dtype=np.float64
            ) / distance
            orientation = Rotation.from_rotvec(
                axis * (distance / 0.11)
            ) * orientation
        quaternion_xyzw = orientation.as_quat()
        ball_quaternions[frame] = quaternion_xyzw[[3, 0, 1, 2]]
    ball_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")]
    robot_centers = np.concatenate(
        [robots[1]["root_translation"], robots[2]["root_translation"]]
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.mean(robot_centers, axis=0)
    camera.distance = 4.4
    # Slightly oblique so cone gaps remain visible instead of collapsing in
    # the side-on projection.
    camera.azimuth = 75
    camera.elevation = -24

    args.out.parent.mkdir(parents=True, exist_ok=True)
    archive_export(args.out, tag="pre_render")
    renderer = mujoco.Renderer(model, height=720, width=960)
    try:
        with imageio.get_writer(
            args.out,
            fps=args.fps / args.frame_step,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            for frame in range(0, frame_count, args.frame_step):
                for person, prefix in ((1, "r1_"), (2, "r2_")):
                    traj = robots[person]
                    _set_robot_qpos(
                        model,
                        data,
                        prefix,
                        traj["joint_names"],
                        traj["joint_positions"][frame],
                        traj["root_translation"][frame],
                        float(traj["root_yaw"][frame]),
                    )
                data.mocap_pos[ball_id] = ball_robot[frame]
                data.mocap_quat[ball_id] = ball_quaternions[frame]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        renderer.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()

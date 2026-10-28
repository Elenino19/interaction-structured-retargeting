"""Small, reproducible figures used during data auditing."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from .audit import load_npz_mapping
from .contact import PHASE_NAMES, resolve_object_model
from .coordinates import points_core4d_to_robot

SMPLX_BODY_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
)

ROBOT_KEYPOINT_EDGES = (
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("pelvis", "neck"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)


def _display_coordinates(points: np.ndarray) -> np.ndarray:
    """Map CORE4D's y-up coordinates to Matplotlib's z-up display."""
    return points[..., [0, 2, 1]]


def render_source_sequence_gif(
    sequence_root: Path,
    sequence_id: str,
    object_models_root: Path,
    output_path: Path,
    *,
    frame_step: int = 2,
    fps: float = 7.5,
) -> None:
    """Render a lightweight skeleton/object preview without SMPL-X model files."""
    sequence_dir = sequence_root.joinpath(*sequence_id.split("/"))
    person1 = load_npz_mapping(sequence_dir / "person1_poses.npz")
    person2 = load_npz_mapping(sequence_dir / "person2_poses.npz")
    object_pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not object_pose_path.exists():
        object_pose_path = sequence_dir / "smooth_objposes.npy"
    object_poses = np.load(object_pose_path)
    mesh = trimesh.load_mesh(
        resolve_object_model(sequence_dir, object_models_root),
        process=False,
    )
    object_vertices = np.asarray(mesh.vertices)
    if len(object_vertices) > 800:
        sample_indices = np.linspace(0, len(object_vertices) - 1, 800).astype(int)
        object_vertices = object_vertices[sample_indices]

    joints1 = np.asarray(person1["joints"])[:, :22]
    joints2 = np.asarray(person2["joints"])[:, :22]
    trajectory_points = np.concatenate(
        [joints1.reshape(-1, 3), joints2.reshape(-1, 3), object_poses[:, :3, 3]],
        axis=0,
    )
    displayed = _display_coordinates(trajectory_points)
    low = np.nanpercentile(displayed, 1, axis=0) - 0.25
    high = np.nanpercentile(displayed, 99, axis=0) + 0.25
    center = (low + high) / 2
    radius = max((high - low).max() / 2, 0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, mode="I", duration=1.0 / fps, loop=0) as writer:
        for frame in range(0, len(object_poses), frame_step):
            figure = plt.figure(figsize=(7, 6), layout="constrained")
            axis = figure.add_subplot(111, projection="3d")
            for joints, color, label in (
                (joints1[frame], "tab:blue", "Person 1"),
                (joints2[frame], "tab:orange", "Person 2"),
            ):
                points = _display_coordinates(joints)
                for child, parent in enumerate(SMPLX_BODY_PARENTS):
                    if parent >= 0:
                        segment = points[[parent, child]]
                        axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color)
                axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=8, color=color)
                axis.plot([], [], [], color=color, label=label)

            pose = object_poses[frame]
            object_world = object_vertices @ pose[:3, :3].T + pose[:3, 3]
            object_display = _display_coordinates(object_world)
            axis.scatter(
                object_display[:, 0],
                object_display[:, 1],
                object_display[:, 2],
                s=2,
                color="tab:green",
                alpha=0.45,
                label="Box",
            )
            axis.set(
                xlim=(center[0] - radius, center[0] + radius),
                ylim=(center[1] - radius, center[1] + radius),
                zlim=(center[2] - radius, center[2] + radius),
                xlabel="World X (m)",
                ylabel="World Z (m)",
                zlabel="Height Y (m)",
                title=f"{sequence_id} · frame {frame}/{len(object_poses) - 1}",
            )
            axis.view_init(elev=22, azim=-65)
            axis.legend(loc="upper right")
            figure.canvas.draw()
            image = np.asarray(figure.canvas.buffer_rgba())[..., :3]
            writer.append_data(image)
            plt.close(figure)


def plot_contact_timeline(
    analysis_npz: Path,
    output_path: Path,
    *,
    fps: float = 15.0,
) -> None:
    """Plot both participants' minimum fingertip-to-object distances."""
    with np.load(analysis_npz) as data:
        person1 = data["person1_fingertip_distances"].min(axis=1)
        person2 = data["person2_fingertip_distances"].min(axis=1)
        phases = data["phase"]

    time = np.arange(len(phases)) / fps
    figure, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
    colors = ("#eeeeee", "#d9d9d9", "#c6dbef", "#fdd0a2", "#c7e9c0", "#eeeeee")
    for phase_index, phase_name in enumerate(PHASE_NAMES):
        mask = phases == phase_index
        if not np.any(mask):
            continue
        indices = np.flatnonzero(mask)
        axis.axvspan(
            indices[0] / fps,
            (indices[-1] + 1) / fps,
            color=colors[phase_index],
            alpha=0.45,
            label=phase_name,
        )
    axis.plot(time, person1 * 100, linewidth=2, label="Person 1 fingertips")
    axis.plot(time, person2 * 100, linewidth=2, label="Person 2 fingertips")
    axis.axhline(5.0, color="black", linestyle="--", linewidth=1, label="5 cm enter threshold")
    axis.axhline(7.0, color="black", linestyle=":", linewidth=1, label="7 cm exit threshold")
    axis.set_title("CORE4D join: minimum fingertip-to-object distance")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Distance (cm)")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axis.legend(unique.values(), unique.keys(), ncol=2, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def render_independent_retarget_gif(
    sequence_root: Path,
    sequence_id: str,
    object_models_root: Path,
    result_root: Path,
    output_path: Path,
    *,
    frame_step: int = 2,
    fps: float = 7.5,
    method_label: str = "Independent baseline",
) -> None:
    """Render both independent G1 keypoint trajectories with the shared box."""
    sequence_dir = sequence_root.joinpath(*sequence_id.split("/"))
    object_pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not object_pose_path.exists():
        object_pose_path = sequence_dir / "smooth_objposes.npy"
    object_poses = np.load(object_pose_path)
    mesh = trimesh.load_mesh(
        resolve_object_model(sequence_dir, object_models_root),
        process=False,
    )
    object_vertices = np.asarray(mesh.vertices)
    if len(object_vertices) > 800:
        indices = np.linspace(0, len(object_vertices) - 1, 800).astype(int)
        object_vertices = object_vertices[indices]

    result_dir = result_root / sequence_id.replace("/", "_")
    robots: dict[int, tuple[np.ndarray, list[str]]] = {}
    for person in (1, 2):
        with np.load(result_dir / f"robot{person}.npz") as result:
            robots[person] = (
                result["robot_keypoints"],
                [str(name) for name in result["keypoint_names"]],
            )

    all_robot_points = np.concatenate([value[0].reshape(-1, 3) for value in robots.values()])
    object_centers = points_core4d_to_robot(object_poses[:, :3, 3])
    all_points = np.concatenate([all_robot_points, object_centers], axis=0)
    low = np.nanpercentile(all_points, 1, axis=0) - 0.25
    high = np.nanpercentile(all_points, 99, axis=0) + 0.25
    center = (low + high) / 2
    radius = max((high - low).max() / 2, 0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, mode="I", duration=1.0 / fps, loop=0) as writer:
        for frame in range(0, len(object_poses), frame_step):
            figure = plt.figure(figsize=(7, 6), layout="constrained")
            axis = figure.add_subplot(111, projection="3d")
            for person, color in ((1, "tab:blue"), (2, "tab:orange")):
                points, names = robots[person]
                current = points[frame]
                name_to_index = {name: index for index, name in enumerate(names)}
                for parent, child in ROBOT_KEYPOINT_EDGES:
                    segment = current[[name_to_index[parent], name_to_index[child]]]
                    axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color)
                axis.scatter(current[:, 0], current[:, 1], current[:, 2], s=10, color=color)
                axis.plot([], [], [], color=color, label=f"Robot {person}")

            pose = object_poses[frame]
            object_world_core = object_vertices @ pose[:3, :3].T + pose[:3, 3]
            object_world = points_core4d_to_robot(object_world_core)
            axis.scatter(
                object_world[:, 0],
                object_world[:, 1],
                object_world[:, 2],
                s=2,
                color="tab:green",
                alpha=0.45,
                label="Box",
            )
            axis.set(
                xlim=(center[0] - radius, center[0] + radius),
                ylim=(center[1] - radius, center[1] + radius),
                zlim=(center[2] - radius, center[2] + radius),
                xlabel="Robot world X (m)",
                ylabel="Robot world Y (m)",
                zlabel="Height Z (m)",
                title=f"{method_label} · frame {frame}/{len(object_poses) - 1}",
            )
            axis.view_init(elev=22, azim=-65)
            axis.legend(loc="upper right")
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[..., :3])
            plt.close(figure)


def plot_metric_comparison(summary_path: Path, output_path: Path) -> None:
    """Render aggregate paired-method means and standard deviations."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    methods = summary["methods"]
    method_names = list(methods)
    panels = (
        ("contact_f1", "Contact F1", None),
        ("contact_onset_error_frames", "|Onset error| (frames)", None),
        (
            "mean_palm_distance_during_expected_contact_cm",
            "Palm distance during contact (cm)",
            None,
        ),
        ("semantic_keypoint_error_cm", "Semantic keypoint error (cm)", None),
        ("joint_jerk_rms_rad_s3", "Joint jerk RMS (rad/s³)", None),
        ("com_support_excess_mean_m", "CoM support excess (m)", None),
        ("ankle_height_mae_m", "Ankle height MAE (m)", None),
    )
    available = [panel for panel in panels if all(panel[0] in methods[name] for name in method_names)]
    figure, axes = plt.subplots(1, len(available), figsize=(3.2 * len(available), 4), layout="constrained")
    if len(available) == 1:
        axes = [axes]
    colors = ("#7f8c8d", "#2878b5", "#d35400")[: len(method_names)]
    labels = tuple(name.replace("_", " ").title() for name in method_names)
    for axis, (metric, title, ylim) in zip(axes, available, strict=True):
        means = [methods[method][metric]["mean"] or 0.0 for method in method_names]
        deviations = [methods[method][metric]["std"] or 0.0 for method in method_names]
        axis.bar(labels, means, yerr=deviations, color=colors, capsize=4)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.2)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.tick_params(axis="x", rotation=25)
    figure.suptitle(
        f"join + box + obs0 ({summary['paired_sequence_count']} paired sequences)",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

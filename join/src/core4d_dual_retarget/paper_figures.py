"""Publication figures for the controlled Join paper evaluation."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from .audit import load_npz_mapping
from .contact import (
    PHASE_NAMES,
    first_stable_contact,
    hysteresis_contact,
    resolve_object_model,
)
from .metrics import (
    robot_hand_volume_distances_to_object,
    rubber_hand_world_volume_points,
)

QUALITATIVE_FRAMES = (55, 68, 77, 100, 150)
QUALITATIVE_MOMENTS = (
    "Single carry",
    "Joiner approach",
    "Contact transition",
    "Joint carry",
    "Post / put-down",
)


def _video_frames(path: Path, frame_indices: tuple[int, ...]) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        return [reader.get_data(index) for index in frame_indices]
    finally:
        reader.close()


def plot_join_qualitative_comparison(
    source_media: Path,
    method_videos: dict[str, Path],
    output_path: Path,
    *,
    source_frame_step: int = 1,
    source_label: str = "Human source",
) -> None:
    """Compose source and method videos at five matched source frames."""
    row_labels = [source_label, *method_videos.keys()]
    source_indices = tuple(frame // source_frame_step for frame in QUALITATIVE_FRAMES)
    image_rows = [_video_frames(source_media, source_indices)]
    image_rows.extend(
        _video_frames(path, QUALITATIVE_FRAMES) for path in method_videos.values()
    )
    figure, axes = plt.subplots(
        len(image_rows),
        len(QUALITATIVE_FRAMES),
        figsize=(7.16, 5.7),
        layout="constrained",
    )
    for row, images in enumerate(image_rows):
        for column, image in enumerate(images):
            axis = axes[row, column]
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(row_labels[row], fontsize=8, fontweight="bold")
            if row == 0:
                axis.set_title(
                    f"{QUALITATIVE_MOMENTS[column]}\nf{QUALITATIVE_FRAMES[column]}",
                    fontsize=7,
                )
    # Restrained callouts: describe experiment provenance, not hidden geometry.
    for row, label in enumerate(row_labels[1:], start=1):
        if "Independent" in label:
            callout = "Independent IK:\nno interaction loss"
        elif "Frozen v41" in label:
            callout = "Frozen v41:\nwarm-started final"
        elif "J4" in label:
            callout = "Cold-start J4:\nfull constraints"
        else:
            continue
        axes[row, 1].text(
            0.02,
            0.96,
            callout,
            transform=axes[row, 1].transAxes,
            va="top",
            fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_join_phase_contact_timeline(
    sequence_root: Path,
    sequence_id: str,
    object_models_root: Path,
    contact_analysis_path: Path,
    result_root: Path,
    output_path: Path,
    *,
    method_label: str = "Robot result",
    fps: float = 15.0,
    contact_enter_m: float = 0.04,
    contact_exit_m: float = 0.055,
    urdf_path: Path = Path(
        "third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
    ),
) -> None:
    """Plot source expected and robot predicted contacts over six Join phases."""
    sequence_dir = sequence_root.joinpath(*sequence_id.split("/"))
    object_pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not object_pose_path.exists():
        object_pose_path = sequence_dir / "smooth_objposes.npy"
    object_poses = np.load(object_pose_path)
    object_mesh = trimesh.load_mesh(
        resolve_object_model(sequence_dir, object_models_root), process=False
    )
    with np.load(contact_analysis_path) as contact:
        phases = np.asarray(contact["phase"])
        expected = {
            1: np.asarray(contact["person1_contact"], dtype=bool),
            2: np.asarray(contact["person2_contact"], dtype=bool),
        }

    predicted: dict[int, np.ndarray] = {}
    distances: dict[int, np.ndarray] = {}
    stem = sequence_id.replace("/", "_")
    for person in (1, 2):
        result_path = result_root / stem / f"robot{person}.npz"
        if not result_path.exists():
            result_path = result_root / f"robot{person}.npz"
        result = load_npz_mapping(result_path)
        hand_world = rubber_hand_world_volume_points(
            urdf_path,
            np.asarray(result["joint_positions"]),
            np.asarray(result["root_translation"]),
            np.asarray(result["root_yaw"]),
        )
        hand_distances = robot_hand_volume_distances_to_object(
            hand_world, object_poses, np.asarray(object_mesh.vertices)
        )
        distances[person] = hand_distances.min(axis=1)
        predicted[person] = hysteresis_contact(
            distances[person],
            enter_threshold=contact_enter_m,
            exit_threshold=contact_exit_m,
        )

    time = np.arange(len(phases), dtype=np.float64) / fps
    colors = ("#eeeeee", "#d9d9d9", "#c6dbef", "#fdd0a2", "#c7e9c0", "#eeeeee")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.16, 3.9),
        sharex=True,
        gridspec_kw={"height_ratios": (1.1, 1.0)},
        layout="constrained",
    )
    figure.suptitle(f"Join contact timeline — {method_label}", fontsize=10)
    for axis in axes:
        for phase_index, phase_name in enumerate(PHASE_NAMES):
            indices = np.flatnonzero(phases == phase_index)
            if len(indices):
                axis.axvspan(
                    indices[0] / fps,
                    (indices[-1] + 1) / fps,
                    color=colors[phase_index],
                    alpha=0.45,
                    label=phase_name,
                )

    axes[0].plot(time, distances[1] * 100, label="Robot 1 palm-box", linewidth=1.5)
    axes[0].plot(time, distances[2] * 100, label="Robot 2 palm-box", linewidth=1.5)
    axes[0].axhline(contact_enter_m * 100, color="black", linestyle="--", linewidth=0.8)
    axes[0].axhline(contact_exit_m * 100, color="black", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Distance [cm]")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.2)

    lanes = (
        (1, expected[1], 3.0, "R1 expected"),
        (1, predicted[1], 2.2, "R1 predicted"),
        (2, expected[2], 1.0, "R2 expected"),
        (2, predicted[2], 0.2, "R2 predicted"),
    )
    for _, signal, y, label in lanes:
        axes[1].step(time, y + signal.astype(float) * 0.55, where="post", label=label)
    for person, color in ((1, "tab:blue"), (2, "tab:orange")):
        for signal, linestyle, name in (
            (expected[person], "--", "expected"),
            (predicted[person], "-", "predicted"),
        ):
            try:
                onset = first_stable_contact(signal)
            except ValueError:
                continue
            axes[1].axvline(
                onset / fps,
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                label=f"R{person} {name} onset f{onset}",
            )
    axes[1].set_yticks([0.2, 1.0, 2.2, 3.0])
    axes[1].set_yticklabels(
        ["R2 predicted", "R2 expected", "R1 predicted", "R1 expected"],
        fontsize=7,
    )
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Contact state")
    axes[1].grid(alpha=0.2)

    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=False))
        axis.legend(unique.values(), unique.keys(), fontsize=5.8, ncol=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

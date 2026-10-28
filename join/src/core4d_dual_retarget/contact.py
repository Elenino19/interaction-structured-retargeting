"""Object-centric fingertip distances and phase detection for CORE4D join."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .audit import load_npz_mapping

SMPLX_FINGERTIP_VERTEX_IDS = {
    "left": np.array([5361, 4933, 5058, 5169, 5286], dtype=np.int64),
    "right": np.array([8079, 7669, 7794, 7905, 8022], dtype=np.int64),
}

PHASE_NAMES = (
    "pre_interaction",
    "single_carry",
    "approach",
    "contact_transition",
    "joint_carry",
    "post_interaction",
)


@dataclass(frozen=True)
class JoinAnalysis:
    sequence_id: str
    frame_count: int
    join_frame: int
    approach_start_frame: int
    incumbent_person: int
    joining_person: int
    incumbent_contact_start_frame: int
    joint_carry_end_frame: int
    joining_contact_ratio_before_join: float
    joining_contact_ratio_after_join: float
    object_model: str


def resolve_object_model(sequence_dir: Path, object_models_root: Path) -> Path:
    """Resolve the upstream absolute model path to the downloaded local asset."""
    metadata = json.loads((sequence_dir / "object_metadata.json").read_text(encoding="utf-8"))
    source_path = Path(str(metadata["obj_model_path"]))
    basename = source_path.name
    matches = sorted(object_models_root.rglob(basename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one local object model named {basename!r}, found {len(matches)}"
        )
    return matches[0]


def fingertips_in_object_space(
    vertices: np.ndarray,
    object_poses: np.ndarray,
) -> np.ndarray:
    """Return ten SMPL-X fingertips in the moving object's canonical frame."""
    fingertip_ids = np.concatenate(
        [SMPLX_FINGERTIP_VERTEX_IDS["left"], SMPLX_FINGERTIP_VERTEX_IDS["right"]]
    )
    world_points = vertices[:, fingertip_ids]
    rotations = object_poses[:, :3, :3]
    translations = object_poses[:, :3, 3]
    return np.einsum("tki,tij->tkj", world_points - translations[:, None], rotations)


def fingertip_distances_to_object(
    vertices: np.ndarray,
    object_poses: np.ndarray,
    object_vertices: np.ndarray,
) -> np.ndarray:
    """Approximate surface distances using the official object's mesh vertices."""
    local_fingertips = fingertips_in_object_space(vertices, object_poses)
    tree = cKDTree(object_vertices)
    distances, _ = tree.query(local_fingertips.reshape(-1, 3), k=1)
    return distances.reshape(local_fingertips.shape[:2])


def hysteresis_contact(
    distances: np.ndarray,
    *,
    enter_threshold: float = 0.05,
    exit_threshold: float = 0.07,
) -> np.ndarray:
    """Convert per-frame minimum distances to a stable binary contact signal."""
    contact = np.zeros(distances.shape[0], dtype=bool)
    active = False
    for index, distance in enumerate(distances):
        if active:
            active = bool(distance < exit_threshold)
        else:
            active = bool(distance < enter_threshold)
        contact[index] = active
    return contact


def first_stable_contact(contact: np.ndarray, *, min_frames: int = 5) -> int:
    """Find the first contact run lasting at least ``min_frames``."""
    if min_frames < 1:
        raise ValueError("min_frames must be positive")
    run_start: int | None = None
    for index, active in enumerate(contact):
        if active and run_start is None:
            run_start = index
        elif not active:
            if run_start is not None and index - run_start >= min_frames:
                return run_start
            run_start = None
    if run_start is not None and len(contact) - run_start >= min_frames:
        return run_start
    raise ValueError("No stable contact interval found")


def stable_contact_end(contact: np.ndarray, start: int, *, min_frames: int = 5) -> int:
    """Return the first frame of a sustained non-contact interval after ``start``."""
    inactive_start: int | None = None
    for index in range(start, len(contact)):
        if not contact[index] and inactive_start is None:
            inactive_start = index
        elif contact[index]:
            inactive_start = None
        if inactive_start is not None and index - inactive_start + 1 >= min_frames:
            return inactive_start
    return len(contact)


def join_phase_labels(
    incumbent_contact: np.ndarray,
    joining_contact: np.ndarray,
    *,
    approach_frames: int = 15,
    transition_radius: int = 2,
    min_contact_frames: int = 5,
) -> tuple[np.ndarray, int, int, int]:
    """Segment a full clip while preserving the four core join phases."""
    if incumbent_contact.shape != joining_contact.shape:
        raise ValueError("Contact signals must have matching shapes")
    incumbent_start = first_stable_contact(incumbent_contact, min_frames=min_contact_frames)
    join_frame = first_stable_contact(joining_contact, min_frames=min_contact_frames)
    approach_start = max(incumbent_start, join_frame - approach_frames)
    transition_start = max(approach_start, join_frame - transition_radius)
    transition_end = min(len(joining_contact), join_frame + transition_radius + 1)
    joint_end = min(
        stable_contact_end(incumbent_contact, transition_end, min_frames=min_contact_frames),
        stable_contact_end(joining_contact, transition_end, min_frames=min_contact_frames),
    )

    phases = np.zeros(len(joining_contact), dtype=np.int8)
    phases[incumbent_start:approach_start] = 1
    phases[approach_start:transition_start] = 2
    phases[transition_start:transition_end] = 3
    phases[transition_end:joint_end] = 4
    phases[joint_end:] = 5
    return phases, join_frame, approach_start, joint_end


def analyze_join_sequence(
    sequence_root: Path,
    sequence_id: str,
    object_models_root: Path,
) -> tuple[JoinAnalysis, dict[str, np.ndarray]]:
    """Compute fingertip contact signals and join phases for one sequence."""
    sequence_dir = sequence_root.joinpath(*sequence_id.split("/"))
    person1 = load_npz_mapping(sequence_dir / "person1_poses.npz")
    person2 = load_npz_mapping(sequence_dir / "person2_poses.npz")
    object_pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not object_pose_path.exists():
        object_pose_path = sequence_dir / "smooth_objposes.npy"
    object_poses = np.load(object_pose_path)
    object_model = resolve_object_model(sequence_dir, object_models_root)
    mesh = trimesh.load_mesh(object_model, process=False)

    person1_distances = fingertip_distances_to_object(
        np.asarray(person1["vertices"]),
        object_poses,
        np.asarray(mesh.vertices),
    )
    person2_distances = fingertip_distances_to_object(
        np.asarray(person2["vertices"]),
        object_poses,
        np.asarray(mesh.vertices),
    )
    person1_min = person1_distances.min(axis=1)
    person2_min = person2_distances.min(axis=1)
    person1_contact = hysteresis_contact(person1_min)
    person2_contact = hysteresis_contact(person2_min)
    contact_signals = {1: person1_contact, 2: person2_contact}
    contact_starts = {
        person: first_stable_contact(contact)
        for person, contact in contact_signals.items()
    }
    incumbent_person = min(contact_starts, key=contact_starts.get)
    joining_person = max(contact_starts, key=contact_starts.get)
    incumbent_contact = contact_signals[incumbent_person]
    joining_contact = contact_signals[joining_person]
    phases, join_frame, approach_start, joint_end = join_phase_labels(
        incumbent_contact,
        joining_contact,
    )

    analysis = JoinAnalysis(
        sequence_id=sequence_id,
        frame_count=len(phases),
        join_frame=join_frame,
        approach_start_frame=approach_start,
        incumbent_person=incumbent_person,
        joining_person=joining_person,
        incumbent_contact_start_frame=contact_starts[incumbent_person],
        joint_carry_end_frame=joint_end,
        joining_contact_ratio_before_join=float(joining_contact[:join_frame].mean()),
        joining_contact_ratio_after_join=float(joining_contact[join_frame:].mean()),
        object_model=str(object_model),
    )
    arrays = {
        "person1_fingertip_distances": person1_distances,
        "person2_fingertip_distances": person2_distances,
        "person1_contact": person1_contact,
        "person2_contact": person2_contact,
        "phase": phases,
    }
    return analysis, arrays

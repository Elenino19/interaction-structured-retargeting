"""Structural validation for selectively downloaded CORE4D sequences."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArraySummary:
    shape: tuple[int, ...]
    dtype: str
    finite: bool | None


@dataclass(frozen=True)
class SequenceAudit:
    sequence_id: str
    valid: bool
    frame_counts: dict[str, int]
    object_rotation_max_orthogonality_error: float
    object_rotation_max_determinant_error: float
    arrays: dict[str, dict[str, ArraySummary]]
    warnings: list[str]
    errors: list[str]


def load_npz_mapping(path: Path) -> dict[str, Any]:
    """Load either a normal NPZ or CORE4D's object-wrapped ``arr_0`` format."""
    with np.load(path, allow_pickle=True) as archive:
        if archive.files == ["arr_0"]:
            value = archive["arr_0"]
            if value.shape == () and isinstance(value.item(), dict):
                return value.item()
        return {name: archive[name] for name in archive.files}


def summarize_mapping(mapping: dict[str, Any]) -> dict[str, ArraySummary]:
    summaries: dict[str, ArraySummary] = {}
    for key, value in mapping.items():
        array = np.asarray(value)
        finite: bool | None
        if np.issubdtype(array.dtype, np.number):
            finite = bool(np.isfinite(array).all())
        else:
            finite = None
        summaries[key] = ArraySummary(
            shape=tuple(int(size) for size in array.shape),
            dtype=str(array.dtype),
            finite=finite,
        )
    return summaries


def _first_frame_count(mapping: dict[str, Any]) -> int:
    preferred = ("joints", "vertices", "body_pose", "global_orient", "transl", "poses")
    for key in preferred:
        if key in mapping:
            value = np.asarray(mapping[key])
            if value.ndim > 0:
                return int(value.shape[0])
    for value in mapping.values():
        array = np.asarray(value)
        if array.ndim > 0:
            return int(array.shape[0])
    return 0


def audit_sequence(sequence_root: Path, sequence_id: str) -> SequenceAudit:
    """Validate frame alignment, finite arrays, and object rotations."""
    root = sequence_root.joinpath(*sequence_id.split("/"))
    required = {
        "person1": root / "person1_poses.npz",
        "person2": root / "person2_poses.npz",
        "object": root / "smooth_objposes.npy",
        "metadata": root / "object_metadata.json",
    }
    warnings: list[str] = []
    errors = [f"Missing {name}: {path}" for name, path in required.items() if not path.exists()]
    if errors:
        return SequenceAudit(sequence_id, False, {}, float("inf"), float("inf"), {}, warnings, errors)

    person1 = load_npz_mapping(required["person1"])
    person2 = load_npz_mapping(required["person2"])
    object_poses = np.load(required["object"])
    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        errors.append("object_metadata.json is not a JSON object")

    arrays = {
        "person1": summarize_mapping(person1),
        "person2": summarize_mapping(person2),
    }
    frame_counts = {
        "person1": _first_frame_count(person1),
        "person2": _first_frame_count(person2),
        "object": int(object_poses.shape[0]) if object_poses.ndim > 0 else 0,
    }
    if len(set(frame_counts.values())) != 1:
        errors.append(f"Frame count mismatch: {frame_counts}")
    if object_poses.ndim != 3 or object_poses.shape[1:] != (4, 4):
        errors.append(f"Expected object poses shaped (T, 4, 4), got {object_poses.shape}")
        orthogonality_error = determinant_error = float("inf")
    else:
        rotations = object_poses[:, :3, :3]
        identity = np.eye(3, dtype=rotations.dtype)
        orthogonality_error = float(
            np.linalg.norm(np.swapaxes(rotations, 1, 2) @ rotations - identity, axis=(1, 2)).max()
        )
        determinant_error = float(np.abs(np.linalg.det(rotations) - 1.0).max())
        if orthogonality_error > 1e-3 or determinant_error > 1e-3:
            warnings.append(
                "Object trajectory contains rotations outside SO(3): "
                f"orth={orthogonality_error:.3g}, det={determinant_error:.3g}"
            )
        if orthogonality_error > 5e-2 or determinant_error > 5e-2:
            errors.append("Object rotation corruption exceeds the automatic-repair threshold")

    for person, summary in arrays.items():
        non_finite = [name for name, value in summary.items() if value.finite is False]
        if non_finite:
            errors.append(f"{person} contains non-finite arrays: {non_finite}")

    return SequenceAudit(
        sequence_id=sequence_id,
        valid=not errors,
        frame_counts=frame_counts,
        object_rotation_max_orthogonality_error=orthogonality_error,
        object_rotation_max_determinant_error=determinant_error,
        arrays=arrays,
        warnings=warnings,
        errors=errors,
    )


def write_audit(audit: SequenceAudit, path: Path) -> None:
    """Serialize an audit report with array summaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(audit), indent=2) + "\n", encoding="utf-8")


def project_rotations_to_so3(object_poses: np.ndarray) -> np.ndarray:
    """Project every 3x3 rotation block to its nearest proper rotation."""
    if object_poses.ndim != 3 or object_poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected object poses shaped (T, 4, 4), got {object_poses.shape}")
    repaired = object_poses.copy()
    rotations = repaired[:, :3, :3]
    u, _, vh = np.linalg.svd(rotations)
    projected = u @ vh
    negative = np.linalg.det(projected) < 0
    if np.any(negative):
        u[negative, :, -1] *= -1
        projected[negative] = u[negative] @ vh[negative]
    repaired[:, :3, :3] = projected
    repaired[:, 3, :] = np.array([0, 0, 0, 1], dtype=repaired.dtype)
    return repaired

"""Post-process consistency repair for the frozen smoothing recipe.

The raw EXT-JC solve already keeps the rubber-hand samples outside the object
convex hull (the same samples and hull as ``metrics.hull_inside_max_cm``).
The frozen ``smooth-trajectories --clearance-recipe`` then Gaussian-smooths
the joints while the object pose is left unchanged, and that is what moves
the hands into the object.  The sole-lift does not.

This repair leaves the coupled objective, its weight, the raw solve and the
frozen smoothing recipe untouched.  After smoothing it blends only the
waist and arm joints back toward the raw solve, per frame, by the smallest
amount that restores the raw clearance, and ramps that blend over a few
frames so the correction itself is not a one-frame pop.  Legs, root and the
sole lift stay as the recipe produced them.

Usage:
    .venv/bin/python scripts/ext_jc_postprocess.py --sequence 20231108/020 \\
        --raw-root <raw> --smooth-root <smooth> --out-root <out>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
import trimesh
from scipy.spatial import ConvexHull

from core4d_dual_retarget.contact import resolve_object_model
from core4d_dual_retarget.coordinates import points_robot_to_core4d
from core4d_dual_retarget.metrics import points_in_object_frame
from core4d_dual_retarget.retarget.balance import (
    LEFT_HAND_VOLUME_OFFSETS_M,
    RIGHT_HAND_VOLUME_OFFSETS_M,
    numpy_apply_root_yaw,
    numpy_transform_offsets,
)
from core4d_dual_retarget.retarget.smooth import recompute_keypoints

URDF = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf")
HAND_TOKENS = ("waist", "shoulder", "elbow", "wrist")
ALPHAS = np.linspace(0.0, 1.0, 21)
TOL_CM = 0.05
RAMP_FRAMES = 3


def _object(sequence_id: str) -> tuple[np.ndarray, np.ndarray]:
    seq_dir = Path("data/core4d/motions").joinpath(*sequence_id.split("/"))
    pose_path = seq_dir / "smooth_objposes_so3.npy"
    if not pose_path.exists():
        pose_path = seq_dir / "smooth_objposes.npy"
    mesh = trimesh.load_mesh(
        resolve_object_model(seq_dir, Path("data/core4d/object_models")), process=False
    )
    equations = ConvexHull(np.asarray(mesh.vertices, dtype=np.float64)).equations
    return np.load(pose_path), equations


def _depth_cm(
    chain: pk.Chain,
    joints: np.ndarray,
    root: np.ndarray,
    yaw: np.ndarray,
    object_poses: np.ndarray,
    equations: np.ndarray,
) -> np.ndarray:
    transforms = chain.forward_kinematics(torch.tensor(joints, dtype=torch.float64))
    local = numpy_transform_offsets(
        transforms["left_rubber_hand"].get_matrix().detach().cpu().numpy(),
        np.asarray(LEFT_HAND_VOLUME_OFFSETS_M, dtype=np.float64),
    )
    right = numpy_transform_offsets(
        transforms["right_rubber_hand"].get_matrix().detach().cpu().numpy(),
        np.asarray(RIGHT_HAND_VOLUME_OFFSETS_M, dtype=np.float64),
    )
    world = numpy_apply_root_yaw(np.concatenate([local, right], axis=1), root, yaw)
    obj = points_in_object_frame(points_robot_to_core4d(world), object_poses)
    values = obj @ equations[:, :3].T + equations[:, 3]
    return np.maximum(-values.max(axis=-1), 0.0).max(axis=1) * 100


def _ramp(alpha: np.ndarray, width: int) -> np.ndarray:
    """Keep each frame's required blend and fade it linearly onto its neighbours."""
    out = np.array(alpha, copy=True)
    for shift in range(1, width + 1):
        scale = 1.0 - shift / (width + 1)
        out[shift:] = np.maximum(out[shift:], alpha[:-shift] * scale)
        out[:-shift] = np.maximum(out[:-shift], alpha[shift:] * scale)
    return out


def repair_robot(
    raw_path: Path,
    smooth_path: Path,
    out_path: Path,
    chain: pk.Chain,
    object_poses: np.ndarray,
    equations: np.ndarray,
) -> dict[str, object]:
    with np.load(raw_path) as raw, np.load(smooth_path) as smooth:
        raw_q = np.asarray(raw["joint_positions"], dtype=np.float64)
        data = {name: smooth[name] for name in smooth.files}
    smooth_q = np.asarray(data["joint_positions"], dtype=np.float64)
    root = np.asarray(data["root_translation"], dtype=np.float64)
    yaw = np.asarray(data["root_yaw"], dtype=np.float64)
    names = [str(n) for n in data["joint_names"]]
    mask = np.array([any(tok in name for tok in HAND_TOKENS) for name in names])

    before = _depth_cm(chain, smooth_q, root, yaw, object_poses, equations)
    grid = []
    for alpha in ALPHAS:
        blended = np.array(smooth_q, copy=True)
        blended[:, mask] = (1.0 - alpha) * smooth_q[:, mask] + alpha * raw_q[:, mask]
        grid.append(_depth_cm(chain, blended, root, yaw, object_poses, equations))
    grid_arr = np.stack(grid)
    required = np.ones(len(smooth_q), dtype=np.float64)
    for frame in range(len(smooth_q)):
        clear = np.flatnonzero(grid_arr[:, frame] <= TOL_CM)
        required[frame] = float(ALPHAS[clear[0]]) if len(clear) else 1.0
    alpha = _ramp(required, RAMP_FRAMES)
    repaired = np.array(smooth_q, copy=True)
    repaired[:, mask] = (1.0 - alpha)[:, None] * smooth_q[:, mask] + alpha[:, None] * raw_q[:, mask]
    after = _depth_cm(chain, repaired, root, yaw, object_poses, equations)

    data["joint_positions"] = repaired
    data["robot_keypoints"] = recompute_keypoints(
        URDF, repaired, root, yaw, [str(n) for n in data["keypoint_names"]]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    return {
        "frames": len(smooth_q),
        "before_max_cm": round(float(before.max()), 3),
        "after_max_cm": round(float(after.max()), 3),
        "frames_before_gt_0p5cm": int((before > 0.5).sum()),
        "frames_after_gt_0p5cm": int((after > 0.5).sum()),
        "frames_blended": int((alpha > 1e-6).sum()),
        "alpha_max": round(float(alpha.max()), 3),
        "alpha_mean_on_blended": round(float(alpha[alpha > 1e-6].mean()) if (alpha > 1e-6).any() else 0.0, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--smooth-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    stem = args.sequence.replace("/", "_")
    poses, equations = _object(args.sequence)
    chain = pk.build_chain_from_urdf(URDF.read_text()).to(dtype=torch.float64)
    report: dict[str, object] = {"sequence": args.sequence, "tolerance_cm": TOL_CM, "ramp_frames": RAMP_FRAMES}
    for person in (1, 2):
        report[f"robot{person}"] = repair_robot(
            args.raw_root / stem / f"robot{person}.npz",
            args.smooth_root / stem / f"robot{person}.npz",
            args.out_root / stem / f"robot{person}.npz",
            chain,
            poses,
            equations,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

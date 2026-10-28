#!/usr/bin/env python3
"""FK GVHMR global SMPL-X, put feet on y=0, separate players on x."""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import os
import sys
from pathlib import Path
import sys as _isr_sys
_isr_sys.path.insert(0, str(Path(__file__).resolve().parent))
from isr_assets import ensure_join_on_path, g1_urdf, g1_xml, gvhmr_root, yolo_weights

import numpy as np
import torch

def gvhmr_result_paths() -> dict[str, Path]:
    import os

    root = Path(os.environ.get("SOCCER_GVHMR_RESULTS", "data/gvhmr/touch_ladder_demo"))
    return {
        "player_a": root / "player_a/touch_ladder_demo_player_a/hmr4d_results.pt",
        "player_b": root / "player_b/touch_ladder_demo_player_b/hmr4d_results.pt",
    }
GAP_M = 2.8
L_SHOULDER, R_SHOULDER = 16, 17


def mean_forward_xz(joints: np.ndarray) -> np.ndarray:
    lateral = joints[:, L_SHOULDER] - joints[:, R_SHOULDER]
    forward = np.cross(lateral, np.array([0.0, 1.0, 0.0], dtype=joints.dtype))
    vec = forward[:, [0, 2]].mean(axis=0)
    norm = np.linalg.norm(vec)
    return vec / max(float(norm), 1e-6)


def rotate_y(joints: np.ndarray, theta: float, pivot_xyz: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(theta), np.sin(theta)
    centered = joints - pivot_xyz
    rotated = centered.copy()
    rotated[..., 0] = cosine * centered[..., 0] + sine * centered[..., 2]
    rotated[..., 2] = -sine * centered[..., 0] + cosine * centered[..., 2]
    return rotated + pivot_xyz


def face_direction(joints: np.ndarray, target_xz: np.ndarray) -> np.ndarray:
    current = mean_forward_xz(joints)
    current_angle = np.arctan2(current[0], current[1])
    target_angle = np.arctan2(target_xz[0], target_xz[1])
    pivot = np.mean(joints[:, 0], axis=0)
    aligned = rotate_y(joints, float(target_angle - current_angle), pivot)
    aligned[..., 0] -= np.mean(aligned[:, 0, 0])
    aligned[..., 2] -= np.mean(aligned[:, 0, 2])
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gvhmr-root", type=Path, default=gvhmr_root())
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/smplx_joints.npz"),
    )
    args = parser.parse_args()
    guard_release_writes(args)
    out_path = args.out.resolve()

    sys.path.insert(0, str(args.gvhmr_root))
    os.chdir(args.gvhmr_root)
    from hmr4d.utils.smplx_utils import make_smplx

    model = make_smplx("supermotion")
    joints = {}
    for prefix, path in gvhmr_result_paths().items():
        pred = torch.load(path, map_location="cpu")
        params = {key: value.float() for key, value in pred["smpl_params_global"].items()}
        with torch.no_grad():
            out = model(**params)
        j = out.joints.cpu().numpy()
        j[..., 1] -= np.min(j[:, [7, 8], 1])
        joints[prefix] = j
        print(prefix, j.shape, "pelvis_y", float(np.median(j[:, 0, 1])))

    # Two independent GVHMR worlds only got an x-shift before. Rotate each
    # skeleton so they face across the ladder, then place them. Do not add π
    # later in G1 IK — that crosses the hips.
    joints["player_a"] = face_direction(joints["player_a"], np.array([1.0, 0.0]))
    joints["player_b"] = face_direction(joints["player_b"], np.array([-1.0, 0.0]))
    joints["player_a"][..., 0] -= 0.5 * GAP_M
    joints["player_b"][..., 0] += 0.5 * GAP_M
    print("A fwd", mean_forward_xz(joints["player_a"]), "B fwd", mean_forward_xz(joints["player_b"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        player_a_joints=joints["player_a"],
        player_b_joints=joints["player_b"],
        fps=24.0,
        gap_m=GAP_M,
        label_type="gvhmr_pseudo_label_not_mocap",
    )
    print("wrote", out_path)


if __name__ == "__main__":
    main()

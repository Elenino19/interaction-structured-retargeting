#!/usr/bin/env python3
"""Pack two person-relative GVHMR tracks into a local demo export.

This is not a metric two-person reconstruction. Each player stays in the
GVHMR global frame; they are only shifted onto a shared ground plane and
separated along x by a visual ladder gap so a later retarget script has
two bodies in one array. Do not treat the result as mocap.
"""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

LADDER_GAP_M = 2.8


def load_global(path: Path) -> dict[str, np.ndarray]:
    data = torch.load(path, map_location="cpu")
    params = data["smpl_params_global"]
    return {key: value.detach().cpu().numpy() for key, value in params.items()}


def place_on_shared_ground(
    player_a: dict[str, np.ndarray],
    player_b: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    a = {k: v.copy() for k, v in player_a.items()}
    b = {k: v.copy() for k, v in player_b.items()}
    a["transl"] -= a["transl"][:1]
    b["transl"] -= b["transl"][:1]
    floor = min(float(a["transl"][:, 1].min()), float(b["transl"][:, 1].min()))
    a["transl"][:, 1] -= floor
    b["transl"][:, 1] -= floor
    a["transl"][:, 0] -= 0.5 * LADDER_GAP_M
    b["transl"][:, 0] += 0.5 * LADDER_GAP_M
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/gvhmr/touch_ladder_demo"),
    )
    parser.add_argument(
        "--prep",
        type=Path,
        default=Path("data/clips/touch_ladder_demo_prep.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/exports/touch_ladder_demo"),
    )
    args = parser.parse_args()
    guard_release_writes(args)

    path_a = args.root / "player_a/touch_ladder_demo_player_a/hmr4d_results.pt"
    path_b = args.root / "player_b/touch_ladder_demo_player_b/hmr4d_results.pt"
    incam_a = args.root / "player_a/touch_ladder_demo_player_a/1_incam.mp4"
    incam_b = args.root / "player_b/touch_ladder_demo_player_b/1_incam.mp4"
    if not path_a.exists() or not path_b.exists():
        raise SystemExit("GVHMR results missing; run run_gvhmr_pair.py first.")

    aligned_a, aligned_b = place_on_shared_ground(load_global(path_a), load_global(path_b))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "cooperative_motion.npz",
        fps=24.0,
        n_frames=aligned_a["body_pose"].shape[0],
        ladder_gap_m=LADDER_GAP_M,
        player_a_body_pose=aligned_a["body_pose"],
        player_a_global_orient=aligned_a["global_orient"],
        player_a_transl=aligned_a["transl"],
        player_a_betas=aligned_a["betas"],
        player_b_body_pose=aligned_b["body_pose"],
        player_b_global_orient=aligned_b["global_orient"],
        player_b_transl=aligned_b["transl"],
        player_b_betas=aligned_b["betas"],
        label_type="gvhmr_pseudo_label_not_mocap",
        alignment="per-player GVHMR global, feet on y=0, x separated by ladder_gap_m",
    )

    preview = args.out / "incam_pair_preview.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(incam_a),
            "-i",
            str(incam_b),
            "-filter_complex",
            "[0:v]scale=420:544,setsar=1[a];[1:v]scale=420:544,setsar=1[b];[a][b]hstack=inputs=2",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-an",
            str(preview),
        ],
        check=True,
    )
    if args.prep.exists():
        shutil.copy(args.prep, args.out / "prep.json")

    meta = {
        "sample_id": "touch_ladder_demo",
        "stage": "gvhmr_pair_packed",
        "redistribute": False,
        "label_type": "gvhmr_pseudo_label_not_mocap",
        "n_frames": int(aligned_a["body_pose"].shape[0]),
        "fps": 24,
        "contacts": 7,
        "preview": str(preview),
        "motion": str(args.out / "cooperative_motion.npz"),
        "gvhmr_player_a": str(path_a),
        "gvhmr_player_b": str(path_b),
        "note": (
            "This file describes the packed GVHMR stage. The accepted G1 result "
            "and prescribed ball/cone scene are documented in result_manifest.json."
        ),
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

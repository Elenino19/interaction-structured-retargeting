#!/usr/bin/env python3
"""Isolate robot position correction while keeping ball/event/lane correction."""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_ablation_experiments import evaluate_variant, load_robot, yup_to_robot
from evaluate_soccer_paper import (
    mesh_metrics,
    phase_contact_and_timing,
    smoothness_metrics,
    support_and_floor_metrics,
)
from retarget_dual_g1_soccer import snap_ball_to_g1_feet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(project_root: Path, variant_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/render_g1_soccer.py"),
            "--g1-dir",
            str(variant_dir),
            "--ball-cones",
            str(variant_dir / "ball_cones.npz"),
            "--out",
            str(variant_dir / "preview.mp4"),
        ],
        cwd=project_root,
        check=True,
    )


def evaluate(
    variant_dir: Path,
    contacts: list[dict],
) -> dict:
    robots = {
        "a": load_robot(variant_dir / "robot_a.npz"),
        "b": load_robot(variant_dir / "robot_b.npz"),
    }
    environment = np.load(variant_dir / "ball_cones.npz")
    ball = yup_to_robot(environment["ball_xyz_yup"])
    legacy = evaluate_variant(variant_dir, contacts)
    return {
        "interaction": phase_contact_and_timing(robots, ball, contacts),
        "support": support_and_floor_metrics(robots, contacts),
        "smoothness": smoothness_metrics(robots),
        "passes": legacy["passes"],
        "mesh_geometry": mesh_metrics(variant_dir, robots, ball, contacts),
    }


def summary_row(name: str, metrics: dict) -> dict:
    return {
        "variant": name,
        "target_foot_distance_mean_m": metrics["interaction"]["overall"][
            "target_foot_distance_mean_m"
        ],
        "contact_surface_error_mean_m": metrics["interaction"]["overall"][
            "contact_surface_error_mean_m"
        ],
        "correct_foot_association_rate": metrics["interaction"]["overall"][
            "correct_foot_association_rate"
        ],
        "timing_error_mean_frames": metrics["interaction"]["overall"][
            "timing_error_mean_frames"
        ],
        "stance_foot_displacement_mean_m": metrics["support"][
            "stance_foot_horizontal_displacement_mean_m"
        ],
        "root_jerk_rms_m_s3": metrics["smoothness"][
            "root_jerk_rms_m_per_s3"
        ],
        "joint_jerk_rms_rad_s3": metrics["smoothness"][
            "joint_jerk_rms_rad_per_s3"
        ],
        "lane_crossing_error_mean_m": metrics["passes"][
            "lane_crossing_error_mean_m"
        ],
        "contact_foot_lane_error_mean_m": metrics["passes"][
            "contact_foot_lane_error_mean_m"
        ],
        "cone_surface_clearance_min_m": metrics["passes"][
            "cone_surface_clearance_min_m"
        ],
        "mesh_floor_penetration_max_m": metrics["mesh_geometry"][
            "foot_ground_penetration_mesh_max_m"
        ],
    }


def make_comparison(no_position: Path, full: Path, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(no_position),
            "-i",
            str(full),
            "-filter_complex",
            (
                "[0:v]scale=480:360,"
                "drawtext=text='Full - Position Correction':x=16:y=16:"
                "fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[a];"
                "[1:v]scale=480:360,"
                "drawtext=text='Full':x=16:y=16:fontsize=24:"
                "fontcolor=white:box=1:boxcolor=black@0.65[b];"
                "[a][b]hstack=inputs=2[out]"
            ),
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--event-dir",
        type=Path,
        default=Path(
            "data/exports/touch_ladder_demo/experiments/ablation/event_semantics"
        ),
    )
    parser.add_argument(
        "--full-dir",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/experiments/ablation/full"),
    )
    parser.add_argument(
        "--contacts",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/ball_cones.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data/exports/touch_ladder_demo/paper_eval/position_ablation"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    guard_release_writes(args)
    project_root = args.project_root.resolve()
    resolve = lambda path: path if path.is_absolute() else project_root / path
    event_dir = resolve(args.event_dir)
    full_dir = resolve(args.full_dir)
    contacts_path = resolve(args.contacts)
    output = resolve(args.out)
    variant_dir = output / "full_without_position"
    if args.force and variant_dir.exists():
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    contacts = json.loads(contacts_path.read_text())["contacts"]

    for name in ("robot_a.npz", "robot_b.npz", "scales.json"):
        shutil.copy2(event_dir / name, variant_dir / name)
    shutil.copy2(event_dir / "ball_cones.npz", variant_dir / "ball_cones.npz")
    snap_ball_to_g1_feet(
        variant_dir,
        contacts,
        variant_dir / "ball_cones.npz",
    )
    render(project_root, variant_dir)
    shutil.copy2(full_dir / "preview.mp4", output / "full.mp4")
    make_comparison(
        variant_dir / "preview.mp4",
        output / "full.mp4",
        output / "comparison.mp4",
    )

    metrics = {
        "schema_version": "1.0.0",
        "experiment": "position_correction_isolation",
        "definition": {
            "full_without_position": (
                "Uses the 340-iteration +Support+Event robot trajectories, then "
                "applies the same ball contact snapping, straight pass lanes, and "
                "cone geometry as Full. It omits pre-IK ladder stationing and the "
                "post-IK lateral station alignment."
            ),
            "full": "Accepted Full result copied without modification.",
        },
        "variants": {
            "full_without_position": evaluate(variant_dir, contacts),
            "full": evaluate(full_dir, contacts),
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    rows = [
        summary_row(name, metrics["variants"][name])
        for name in ("full_without_position", "full")
    ]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "1.0.0",
        "variants": {
            "full_without_position": {
                "robots_from": str(event_dir.relative_to(project_root)),
                "ball_correction": True,
                "lane_correction": True,
                "position_correction": False,
                "preview": str(
                    (variant_dir / "preview.mp4").relative_to(project_root)
                ),
            },
            "full": {
                "source": str(full_dir.relative_to(project_root)),
                "position_correction": True,
                "preview": str((output / "full.mp4").relative_to(project_root)),
            },
        },
        "comparison": str((output / "comparison.mp4").relative_to(project_root)),
        "full_sha256": sha256(output / "full.mp4"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": rows, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()

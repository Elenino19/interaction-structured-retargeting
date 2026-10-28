#!/usr/bin/env python3
"""Run isolated four-way ablations without touching the accepted result."""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retarget_dual_g1_soccer import (
    SMPLX_IDX,
    SoccerRetargetWeights,
    contact_windows,
    generic_support_windows,
    retarget_soccer,
    save_result,
)

from isr_assets import ensure_join_on_path, g1_urdf, g1_xml
ensure_join_on_path()

from core4d_dual_retarget.retarget.independent import IndependentRetargetConfig


VARIANTS = {
    "basic_ik": {
        "label": "Basic IK",
        "support": False,
        "event_semantics": False,
        "ball_event_correction": False,
        "position_correction": False,
    },
    "support": {
        "label": "+ support",
        "support": True,
        "event_semantics": False,
        "ball_event_correction": False,
        "position_correction": False,
    },
    "event_semantics": {
        "label": "+ event semantics",
        "support": True,
        "event_semantics": True,
        "ball_event_correction": False,
        "position_correction": False,
    },
    "full": {
        "label": "Full",
        "support": True,
        "event_semantics": True,
        "ball_event_correction": True,
        "position_correction": True,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_ball_from_contacts(
    n: int,
    contacts: list[dict],
    joints_a: np.ndarray,
    joints_b: np.ndarray,
) -> np.ndarray:
    """Rebuild the shared source-derived ball baseline without event corrections."""
    players = {"a": joints_a, "b": joints_b}
    other = {"a": joints_b, "b": joints_a}
    ankle_idx = {"left": SMPLX_IDX["left_ankle"], "right": SMPLX_IDX["right_ankle"]}
    waypoints: list[tuple[int, np.ndarray]] = []
    for event in contacts:
        joints = players[event["player"]]
        opposite = other[event["player"]]
        frame = int(event["frame"])
        ankle = joints[frame, ankle_idx[event["side"]]].copy()
        toward = np.mean(opposite[:, 0], axis=0) - np.mean(joints[:, 0], axis=0)
        toward[1] = 0.0
        ankle += 0.14 * toward / (np.linalg.norm(toward) + 1e-6)
        ankle[1] = 0.11
        waypoints.append((frame, ankle))
    if waypoints[0][0] != 0:
        waypoints.insert(0, (0, waypoints[0][1].copy()))
    if waypoints[-1][0] != n - 1:
        waypoints.append((n - 1, waypoints[-1][1].copy()))
    frames = np.asarray([frame for frame, _ in waypoints], dtype=np.float64)
    points = np.stack([point for _, point in waypoints])
    ball = np.zeros((n, 3), dtype=np.float64)
    timeline = np.arange(n)
    for axis in range(3):
        ball[:, axis] = np.interp(timeline, frames, points[:, axis])
    ball[:, 1] = 0.11
    smoothed = ball.copy()
    kernel = np.ones(5) / 5.0
    for axis in (0, 2):
        smoothed[:, axis] = np.convolve(
            np.pad(ball[:, axis], (2, 2), mode="edge"),
            kernel,
            mode="valid",
        )
    for frame, _ in waypoints:
        smoothed[frame] = ball[frame]
    return smoothed


def write_baseline_environment(
    source_path: Path,
    output_path: Path,
    ball_yup: np.ndarray,
) -> None:
    payload = {key: np.array(value) for key, value in np.load(source_path).items()}
    payload["ball_xyz_yup"] = ball_yup
    payload["method"] = np.asarray("source_contact_interpolation_no_event_correction")
    np.savez_compressed(output_path, **payload)


def render_variant(
    project_root: Path,
    variant_dir: Path,
    python: Path,
) -> None:
    subprocess.run(
        [
            str(python),
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


def make_comparison(output_root: Path) -> None:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    inputs: list[str] = []
    filters: list[str] = []
    for index, (name, config) in enumerate(VARIANTS.items()):
        inputs.extend(["-i", str(output_root / name / "preview.mp4")])
        label = config["label"].replace("'", r"\'")
        draw = (
            f"[{index}:v]scale=480:360,"
            f"drawtext=text='{label}':x=18:y=18:fontsize=26:"
            "fontcolor=white:box=1:boxcolor=black@0.65"
        )
        if font.exists():
            draw += f":fontfile={font}"
        filters.append(draw + f"[v{index}]")
    filters.append("[v0][v1]hstack=inputs=2[top]")
    filters.append("[v2][v3]hstack=inputs=2[bottom]")
    filters.append("[top][bottom]vstack=inputs=2[out]")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_root / "comparison_2x2.mp4"),
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
        "--joints",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/smplx_joints.npz"),
    )
    parser.add_argument(
        "--contacts",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/ball_cones.json"),
    )
    parser.add_argument(
        "--canonical-ball",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/ball_cones.npz"),
    )
    parser.add_argument(
        "--canonical-g1",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/g1"),
    )
    parser.add_argument(
        "--canonical-preview",
        type=Path,
        default=Path(
            "data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4"
        ),
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=g1_urdf(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/experiments/ablation"),
    )
    parser.add_argument("--iterations", type=int, default=340)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    guard_release_writes(args)

    project_root = args.project_root.resolve()
    resolve = lambda path: path if path.is_absolute() else project_root / path
    joints_path = resolve(args.joints)
    contacts_path = resolve(args.contacts)
    canonical_ball = resolve(args.canonical_ball)
    canonical_g1 = resolve(args.canonical_g1)
    canonical_preview = resolve(args.canonical_preview)
    output_root = resolve(args.output)
    urdf = resolve(args.urdf)
    output_root.mkdir(parents=True, exist_ok=True)

    packed = np.load(joints_path)
    source = {
        "a": np.asarray(packed["player_a_joints"]),
        "b": np.asarray(packed["player_b_joints"]),
    }
    contacts = json.loads(contacts_path.read_text())["contacts"]
    n = min(len(source["a"]), len(source["b"]))
    baseline_ball = baseline_ball_from_contacts(
        n,
        contacts,
        source["a"][:n],
        source["b"][:n],
    )
    config = IndependentRetargetConfig(
        iterations=args.iterations,
        learning_rate=0.04,
        keypoint_weight=1.0,
        root_regularization_weight=0.35,
        velocity_weight=0.035,
        acceleration_weight=0.08,
    )

    for variant, flags in VARIANTS.items():
        variant_dir = output_root / variant
        if variant_dir.exists() and args.force:
            shutil.rmtree(variant_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        if variant == "full":
            for name in ("robot_a.npz", "robot_b.npz", "scales.json"):
                shutil.copy2(canonical_g1 / name, variant_dir / name)
            shutil.copy2(canonical_ball, variant_dir / "ball_cones.npz")
            shutil.copy2(canonical_preview, variant_dir / "preview.mp4")
            provenance = "accepted_canonical_copy"
        elif not (
            (variant_dir / "robot_a.npz").exists()
            and (variant_dir / "robot_b.npz").exists()
            and not args.force
        ):
            scales = {}
            soccer_weights = SoccerRetargetWeights(
                support_plant=6.0 if flags["support"] else 0.0,
                support_floor=6.0 if flags["support"] else 0.0,
                kick_velocity=24.0 if flags["event_semantics"] else 0.0,
                event_keypoint_boost=bool(flags["event_semantics"]),
                event_smooth_relaxation=bool(flags["event_semantics"]),
            )
            for player in ("a", "b"):
                joints = source[player][:n]
                generic_plant = (
                    generic_support_windows(joints)
                    if flags["support"]
                    else np.zeros((n, 2), dtype=np.float64)
                )
                if flags["event_semantics"]:
                    kick, plant = contact_windows(
                        contacts,
                        player,
                        n,
                        step_plant=generic_plant,
                    )
                else:
                    kick = np.zeros((n, 2), dtype=np.float64)
                    plant = generic_plant
                result = retarget_soccer(
                    joints,
                    urdf,
                    config,
                    kick,
                    plant,
                    loss_weights=soccer_weights,
                )
                save_result(result, variant_dir / f"robot_{player}.npz")
                scales[f"robot_{player}"] = float(result.scale)
            (variant_dir / "scales.json").write_text(
                json.dumps(scales, indent=2) + "\n"
            )
            write_baseline_environment(
                canonical_ball,
                variant_dir / "ball_cones.npz",
                baseline_ball,
            )
            provenance = "fresh_340_iteration_ablation"
        else:
            provenance = "existing_ablation_reused"

        if variant != "full" and not (variant_dir / "ball_cones.npz").exists():
            write_baseline_environment(
                canonical_ball,
                variant_dir / "ball_cones.npz",
                baseline_ball,
            )
        if variant != "full" and not args.skip_render and (
            args.force or not (variant_dir / "preview.mp4").exists()
        ):
            render_variant(project_root, variant_dir, Path(sys.executable))
        metadata = {
            "variant": variant,
            **flags,
            "iterations": args.iterations,
            "provenance": provenance,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s_this_invocation": time.monotonic() - started,
            "inputs": {
                "joints": str(joints_path),
                "joints_sha256": sha256(joints_path),
                "contacts": str(contacts_path),
                "contacts_sha256": sha256(contacts_path),
            },
        }
        (variant_dir / "experiment.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print("completed", variant, metadata["elapsed_s_this_invocation"])

    if not args.skip_render:
        make_comparison(output_root)
    manifest = {
        "schema_version": "1.0.0",
        "experiment": "touch_ladder_ablation",
        "variants": VARIANTS,
        "iterations": args.iterations,
        "comparison_video": str(
            (output_root / "comparison_2x2.mp4").relative_to(project_root)
        ),
        "claim_boundary": (
            "Single-sequence internal ablation with geometric proxies; "
            "not a population benchmark or physics validation."
        ),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Controlled fixed-ball comparison: event-conditioned robot IK OFF vs ON.

The frozen Full ball trajectory (``data/exports/touch_ladder_demo/ball_cones.npz``)
is copied verbatim into every variant directory. No variant re-snaps the ball.
Every variant starts from the same deterministic IK initialisation (joint-limit
midpoints, zero root delta, shoulder yaw) and the same 340-iteration Adam
budget. The accepted Full joint solution is never loaded as a start point and
nothing under ``data/exports/touch_ladder_demo/g1`` is written.

Two stationing modes (``--stationing``):

frozen_reference  (PRIMARY)
    The shared human reference is the accepted Full run's stored IK target
    keypoints (``g1/robot_*.npz::target_keypoints``, 16 keypoints in the robot
    frame, including its stationing, event-frame foot grounding and the two
    post-IK world-Y alignments). The frozen ball was snapped to robots that
    tracked exactly these targets, so the ball/reference pair is consistent.
    Both variants see identical targets; only IK event conditioning differs.
    Shared confound: event-timed stationing and event-frame foot-height
    grounding of the reference are inherited by BOTH variants.

current_code  (SECONDARY)
    The reference is rebuilt from ``smplx_joints.npz`` with the current
    ``station_pair_on_ladder`` code. This code state post-dates the accepted
    run and stations the striking toe on the previous lane at several event
    frames, so the frozen ball is far from both variants. Kept only to document
    the stationing dependency.

Variants
--------
event_off
    kick = 0 (no foot-role windows), generic height/speed support windows,
    kick-velocity 0, no event keypoint boost, no event smoothness relaxation.
    Support/floor terms stay at the Full value (6/6).
event_on
    Generic support windows + ``contact_windows`` (foot-role windows and
    event plant), default ``SoccerRetargetWeights`` (kick-velocity 24, boost,
    relaxation). Same support baseline as ``event_off``.
event_on_full_recipe
    The accepted recipe: ``contact_windows`` with the event-only plant mask
    (``_apply_player_plan`` step plant, no generic support) and default weights.
    Used to check how closely the current IK code regenerates the frozen Full
    joints from the frozen targets.
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retarget_dual_g1_soccer import (  # noqa: E402
    KEYPOINT_NAMES,
    SoccerRetargetWeights,
    _apply_player_plan,
    _contact_stations,
    align_second_pass_stations,
    contact_windows,
    generic_support_windows,
    ladder_geometry,
    retarget_soccer,
    save_result,
)

from isr_assets import ensure_join_on_path, g1_urdf, g1_xml
ensure_join_on_path()

from core4d_dual_retarget.coordinates import points_core4d_to_robot  # noqa: E402
from core4d_dual_retarget.retarget.independent import (  # noqa: E402
    IndependentRetargetConfig,
    _yaw_from_shoulders,
)

VARIANTS = {
    "event_off": {
        "label": "Event IK OFF",
        "event_foot_role_windows": False,
        "event_loss_terms": False,
        "event_plant_mask_only": False,
    },
    "event_on": {
        "label": "Event IK ON",
        "event_foot_role_windows": True,
        "event_loss_terms": True,
        "event_plant_mask_only": False,
    },
    "event_on_full_recipe": {
        "label": "Event IK ON (accepted recipe)",
        "event_foot_role_windows": True,
        "event_loss_terms": True,
        "event_plant_mask_only": True,
    },
}
PRIMARY_ORDER = ("event_off", "event_on", "event_on_full_recipe")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_scene_only(project_root: Path, variant_dir: Path, xml: Path) -> None:
    sys.path.insert(0, str(project_root / "scripts"))
    from render_g1_soccer import build_scene

    env = np.load(variant_dir / "ball_cones.npz")
    build_scene(xml, env["cone_xyz_yup"], variant_dir / "soccer_scene.xml")


def render_variant(project_root: Path, variant_dir: Path) -> None:
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


def compare_to_accepted(variant_dir: Path, accepted_dir: Path) -> dict:
    report = {}
    for name in ("robot_a.npz", "robot_b.npz"):
        ours = {k: np.array(v) for k, v in np.load(variant_dir / name).items()}
        ref = {k: np.array(v) for k, v in np.load(accepted_dir / name).items()}
        entry = {}
        for key in ("target_keypoints", "joint_positions", "root_translation", "root_yaw", "robot_keypoints"):
            if ours[key].shape != ref[key].shape:
                entry[key] = {"shape_match": False}
                continue
            diff = np.abs(ours[key] - ref[key])
            entry[key] = {
                "shape_match": True,
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
            }
        report[name] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--joints", type=Path, default=Path("data/exports/touch_ladder_demo/smplx_joints.npz"))
    parser.add_argument("--contacts", type=Path, default=Path("data/exports/touch_ladder_demo/ball_cones.json"))
    parser.add_argument("--frozen-ball", type=Path, default=Path("data/exports/touch_ladder_demo/ball_cones.npz"))
    parser.add_argument(
        "--accepted-g1",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/g1"),
        help="Read-only. Source of the frozen reference targets and of the reproduction check.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=g1_urdf(),
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=g1_xml(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/exports/touch_ladder_demo/paper_eval/event_coordination_extension/fixed_ball_ablation"
        ),
    )
    parser.add_argument("--stationing", choices=("frozen_reference", "current_code"), default="frozen_reference")
    parser.add_argument("--iterations", type=int, default=340)
    parser.add_argument("--variants", nargs="*", default=list(PRIMARY_ORDER))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    guard_release_writes(args)

    project_root = args.project_root.resolve()
    resolve = lambda path: path if path.is_absolute() else project_root / path  # noqa: E731
    joints_path = resolve(args.joints)
    contacts_path = resolve(args.contacts)
    frozen_ball = resolve(args.frozen_ball)
    accepted_g1 = resolve(args.accepted_g1)
    urdf = resolve(args.urdf)
    xml = resolve(args.xml)
    output_root = resolve(args.output) / args.stationing
    output_root.mkdir(parents=True, exist_ok=True)

    frozen_ball_sha = sha256(frozen_ball)
    accepted_hashes = {name: sha256(accepted_g1 / name) for name in ("robot_a.npz", "robot_b.npz")}
    accepted = {
        player: {k: np.array(v) for k, v in np.load(accepted_g1 / f"robot_{player}.npz").items()}
        for player in ("a", "b")
    }

    packed = np.load(joints_path)
    source = {"a": np.asarray(packed["player_a_joints"]), "b": np.asarray(packed["player_b_joints"])}
    contacts = json.loads(contacts_path.read_text())["contacts"]
    n = min(len(source["a"]), len(source["b"]))
    _, gap_lanes = ladder_geometry()
    station_events = _contact_stations(contacts, gap_lanes)
    pelvis_i = KEYPOINT_NAMES.index("pelvis")

    config = IndependentRetargetConfig(
        iterations=args.iterations,
        learning_rate=0.04,
        keypoint_weight=1.0,
        root_regularization_weight=0.35,
        velocity_weight=0.035,
        acceleration_weight=0.08,
    )

    manifest_variants = {}
    for variant in args.variants:
        flags = VARIANTS[variant]
        raw_dir = output_root / variant
        aligned_dir = output_root / f"{variant}_aligned"
        if args.force:
            for folder in (raw_dir, aligned_dir):
                if folder.exists():
                    shutil.rmtree(folder)
        raw_dir.mkdir(parents=True, exist_ok=True)
        want_aligned = args.stationing == "current_code"
        if want_aligned:
            aligned_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        have_raw = (raw_dir / "robot_a.npz").exists() and (raw_dir / "robot_b.npz").exists()
        weights = SoccerRetargetWeights(
            support_plant=6.0,
            support_floor=6.0,
            kick_velocity=24.0 if flags["event_loss_terms"] else 0.0,
            event_keypoint_boost=bool(flags["event_loss_terms"]),
            event_smooth_relaxation=bool(flags["event_loss_terms"]),
        )
        per_player = {}
        if not have_raw:
            scales = {}
            for player in ("a", "b"):
                joints = source[player][:n].copy()
                # Event plant mask (support foot +/-2 frames at annotated events).
                stationed, step_plant = _apply_player_plan(joints, player, station_events, gap_lanes)
                if args.stationing == "frozen_reference":
                    targets = accepted[player]["target_keypoints"][:n]
                    base_root = targets[:, pelvis_i].copy()
                    initial_yaw = _yaw_from_shoulders(points_core4d_to_robot(joints))
                    precomputed = (targets, base_root, initial_yaw, float(accepted[player]["scale"]))
                    ik_source = joints
                    generic_plant = generic_support_windows(joints)
                else:
                    precomputed = None
                    ik_source = stationed
                    generic_plant = generic_support_windows(stationed)
                if flags["event_plant_mask_only"]:
                    kick, plant = contact_windows(contacts, player, n, step_plant=step_plant)
                elif flags["event_foot_role_windows"]:
                    kick, plant = contact_windows(contacts, player, n, step_plant=generic_plant)
                else:
                    kick = np.zeros((n, 2), dtype=np.float64)
                    plant = generic_plant
                result = retarget_soccer(
                    ik_source, urdf, config, kick, plant,
                    loss_weights=weights, precomputed_targets=precomputed,
                )
                save_result(result, raw_dir / f"robot_{player}.npz")
                scales[f"robot_{player}"] = float(result.scale)
                per_player[player] = {
                    "kick_window_frames": int((kick > 0).any(1).sum()),
                    "plant_nonzero_frames": int((plant > 0).any(1).sum()),
                    "final_loss": float(result.loss_history[-1]),
                    "best_loss": float(result.loss_history.min()),
                }
                np.savez_compressed(raw_dir / f"windows_{player}.npz", kick=kick, plant=plant)
            (raw_dir / "scales.json").write_text(json.dumps(scales, indent=2) + "\n")
            shutil.copy2(frozen_ball, raw_dir / "ball_cones.npz")
            if want_aligned:
                for name in ("robot_a.npz", "robot_b.npz", "scales.json", "ball_cones.npz"):
                    shutil.copy2(raw_dir / name, aligned_dir / name)
                align_second_pass_stations(aligned_dir, gap_lanes)
            provenance = f"fresh_{args.iterations}_iteration_fixed_ball_run"
        else:
            provenance = "existing_run_reused"

        folders = [raw_dir] + ([aligned_dir] if want_aligned else [])
        for folder in folders:
            if not (folder / "ball_cones.npz").exists():
                shutil.copy2(frozen_ball, folder / "ball_cones.npz")
            if sha256(folder / "ball_cones.npz") != frozen_ball_sha:
                raise RuntimeError(f"ball in {folder} differs from frozen Full ball")
            if not args.skip_render and (args.force or not (folder / "preview.mp4").exists()):
                render_variant(project_root, folder)
            if not (folder / "soccer_scene.xml").exists():
                build_scene_only(project_root, folder, xml)

        reproduction = {"raw_vs_accepted_g1": compare_to_accepted(raw_dir, accepted_g1)}
        if want_aligned:
            reproduction["aligned_vs_accepted_g1"] = compare_to_accepted(aligned_dir, accepted_g1)
        metadata = {
            "variant": variant,
            "stationing": args.stationing,
            **flags,
            "soccer_weights": asdict(weights),
            "ik_config": asdict(config),
            "reference": (
                "accepted Full IK target keypoints (g1/robot_*.npz::target_keypoints), shared verbatim"
                if args.stationing == "frozen_reference"
                else "current-code station_pair_on_ladder reference rebuilt from smplx_joints.npz"
            ),
            "ball": {"path": str(frozen_ball), "sha256": frozen_ball_sha, "note": "Frozen Full ball copied verbatim; never re-snapped."},
            "initialisation": (
                "Deterministic: joint-limit midpoints, zero root delta, shoulder yaw; "
                "the accepted Full joint solution is never loaded as a start point."
            ),
            "iterations": args.iterations,
            "provenance": provenance,
            "per_player": per_player,
            "elapsed_s_this_invocation": time.monotonic() - started,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "joints": str(joints_path),
                "joints_sha256": sha256(joints_path),
                "contacts": str(contacts_path),
                "contacts_sha256": sha256(contacts_path),
                "accepted_g1_sha256_read_only": accepted_hashes,
            },
            "reproduction_check": reproduction,
        }
        (raw_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest_variants[variant] = metadata
        print("completed", args.stationing, variant, f"{metadata['elapsed_s_this_invocation']:.1f}s")
        for name, entry in reproduction["raw_vs_accepted_g1"].items():
            print(
                f"  {name}: target max|d|={entry['target_keypoints']['max_abs_diff']:.4f} "
                f"joint mean|d|={entry['joint_positions']['mean_abs_diff']:.4f} "
                f"max|d|={entry['joint_positions']['max_abs_diff']:.4f} "
                f"root mean|d|={entry['root_translation']['mean_abs_diff']:.4f}"
            )

    manifest_path = output_root / "manifest.json"
    manifest = {
        "schema_version": "1.1.0",
        "experiment": "touch_ladder_fixed_ball_event_ik_ablation",
        "stationing": args.stationing,
        "primary_order": list(PRIMARY_ORDER),
        "iterations": args.iterations,
        "frozen_ball_sha256": frozen_ball_sha,
        "variants": manifest_variants,
        "claim_boundary": (
            "Single-sequence controlled kinematic comparison on a prescribed, event/scene-conditioned "
            "ball path. The shared reference carries event-timed stationing and event-frame foot "
            "grounding, so the comparison isolates event-conditioned IK refinement (foot-role windows, "
            "event loss terms, event plant mask), not all event information. Not physics, not task success."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Empirical separability probe for the Soccer Full pipeline.

For every stage, perturb robot B's input (or output) and test whether robot A's
result changes bit-for-bit. Stages: stationing (`station_pair_on_ladder`),
event windows (`contact_windows`), IK (`retarget_soccer`), post-IK alignment
(`align_second_pass_stations`), ball generation (`snap_ball_to_g1_feet`).

Writes `separability_probe.json` into the output directory. Reads frozen
artifacts only; writes only into a temporary directory and the output root.
"""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retarget_dual_g1_soccer import (  # noqa: E402
    SoccerRetargetWeights,
    align_second_pass_stations,
    contact_windows,
    ladder_geometry,
    retarget_soccer,
    save_result,
    snap_ball_to_g1_feet,
    station_pair_on_ladder,
)

from isr_assets import ensure_join_on_path, g1_urdf, g1_xml
ensure_join_on_path()
from core4d_dual_retarget.retarget.independent import IndependentRetargetConfig  # noqa: E402

URDF = g1_urdf()


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/exports/touch_ladder_demo"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/paper_eval/object_aware_comparison"),
    )
    parser.add_argument("--ik-iterations", type=int, default=60)
    args = parser.parse_args()
    guard_release_writes(args)
    args.out.mkdir(parents=True, exist_ok=True)

    packed = np.load(args.root / "smplx_joints.npz")
    joints_a = np.asarray(packed["player_a_joints"])
    joints_b = np.asarray(packed["player_b_joints"])
    contacts = json.loads((args.root / "ball_cones.json").read_text())["contacts"]
    _, gap_lanes = ladder_geometry()
    rng = np.random.default_rng(0)
    perturbed_b = joints_b + rng.normal(0.0, 0.05, joints_b.shape)
    perturbed_b[..., 2] += 0.4  # also shift B sideways by 0.4 m
    report: dict = {"perturbation": "B source joints: N(0,0.05 m) noise + 0.4 m lateral shift"}

    # 1. Stationing / event-frame grounding.
    a_ref, b_ref, plant_a_ref, plant_b_ref = station_pair_on_ladder(joints_a, joints_b, contacts, gap_lanes)
    a_alt, _, plant_a_alt, _ = station_pair_on_ladder(joints_a, perturbed_b, contacts, gap_lanes)
    report["stationing_station_pair_on_ladder"] = {
        "robot_a_targets_max_abs_change": max_abs(a_ref, a_alt),
        "robot_a_plant_max_abs_change": max_abs(plant_a_ref, plant_a_alt),
        "uses_other_robot": False,
        "note": "Only n = min(len_a, len_b) is shared; the per-player plan reads events + lane geometry.",
    }

    # 2. Event windows.
    n = len(a_ref)
    kick_ref, plant_ref = contact_windows(contacts, "a", n, step_plant=plant_a_ref)
    kick_alt, plant_alt = contact_windows(contacts, "a", n, step_plant=plant_a_alt)
    report["event_windows_contact_windows"] = {
        "kick_max_abs_change": max_abs(kick_ref, kick_alt),
        "plant_max_abs_change": max_abs(plant_ref, plant_alt),
        "uses_other_robot": False,
    }

    # 3. IK: A solved alone vs. A solved after B (shared process state), B perturbed.
    config = IndependentRetargetConfig(
        iterations=args.ik_iterations, learning_rate=0.04, keypoint_weight=1.0,
        root_regularization_weight=0.35, velocity_weight=0.035, acceleration_weight=0.08,
    )
    res_a_alone = retarget_soccer(a_ref, URDF, config, kick_ref, plant_ref, SoccerRetargetWeights())
    kick_b, plant_b = contact_windows(contacts, "b", n, step_plant=plant_b_ref)
    _ = retarget_soccer(b_ref, URDF, config, kick_b, plant_b, SoccerRetargetWeights())
    res_a_after_b = retarget_soccer(a_ref, URDF, config, kick_ref, plant_ref, SoccerRetargetWeights())
    report["ik_retarget_soccer"] = {
        "iterations": args.ik_iterations,
        "robot_a_joint_max_abs_change": max_abs(res_a_alone.joint_positions, res_a_after_b.joint_positions),
        "robot_a_root_max_abs_change": max_abs(res_a_alone.root_translation, res_a_after_b.root_translation),
        "uses_other_robot": False,
        "note": "Signature takes one robot's targets; the loss has no term referencing the other robot or the ball.",
    }

    # 4. Post-IK alignment and 5. ball generation on the frozen Full robots.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref_dir, alt_dir = tmp / "ref", tmp / "alt"
        for folder in (ref_dir, alt_dir):
            folder.mkdir()
            for name in ("robot_a.npz", "robot_b.npz"):
                shutil.copy2(args.root / "g1" / name, folder / name)
            shutil.copy2(args.root / "ball_cones.npz", folder / "ball_cones.npz")
        # Perturb B's output in alt: shift pelvis/root by +0.10 m in Y from f60 on.
        payload = {k: np.array(v) for k, v in np.load(alt_dir / "robot_b.npz").items()}
        payload["root_translation"][60:, 1] += 0.10
        payload["robot_keypoints"][60:, :, 1] += 0.10
        np.savez_compressed(alt_dir / "robot_b.npz", **payload)

        # The frozen robots are already aligned; applying alignment again is a
        # pure function of each robot's own file, which is what we test.
        align_second_pass_stations(ref_dir, gap_lanes)
        align_second_pass_stations(alt_dir, gap_lanes)
        a_ref_al = {k: np.array(v) for k, v in np.load(ref_dir / "robot_a.npz").items()}
        a_alt_al = {k: np.array(v) for k, v in np.load(alt_dir / "robot_a.npz").items()}
        report["post_ik_align_second_pass_stations"] = {
            "robot_a_root_max_abs_change": max_abs(a_ref_al["root_translation"], a_alt_al["root_translation"]),
            "robot_a_keypoints_max_abs_change": max_abs(a_ref_al["robot_keypoints"], a_alt_al["robot_keypoints"]),
            "uses_other_robot": False,
            "note": "A's offset uses A's f0 right toe; B's offset uses B's f15/f21 toes.",
        }

        # Reset robots to frozen copies for the ball probe.
        for folder in (ref_dir, alt_dir):
            shutil.copy2(args.root / "g1" / "robot_a.npz", folder / "robot_a.npz")
        shutil.copy2(args.root / "g1" / "robot_b.npz", ref_dir / "robot_b.npz")
        np.savez_compressed(alt_dir / "robot_b.npz", **payload)
        snap_ball_to_g1_feet(ref_dir, contacts, ref_dir / "ball_cones.npz")
        snap_ball_to_g1_feet(alt_dir, contacts, alt_dir / "ball_cones.npz")
        ball_ref = np.load(ref_dir / "ball_cones.npz")["ball_xyz_yup"]
        ball_alt = np.load(alt_dir / "ball_cones.npz")["ball_xyz_yup"]
        frozen = np.load(args.root / "ball_cones.npz")["ball_xyz_yup"]
        diff = np.abs(ball_ref - ball_alt).max(axis=1)
        changed = np.where(diff > 1e-9)[0]
        report["ball_snap_ball_to_g1_feet"] = {
            "resnap_of_frozen_robots_reproduces_frozen_ball_max_abs": max_abs(ball_ref, frozen),
            "ball_max_abs_change_when_B_perturbed": float(diff.max()),
            "frames_changed": [int(changed.min()), int(changed.max())] if len(changed) else [],
            "n_frames_changed": int(len(changed)),
            "uses_other_robot": True,
            "note": (
                "Each waypoint reads the toucher's toe/ankle AND the other robot's pelvis (direction); "
                "cone X is the mean of both roots. Robots never receive feedback from the ball."
            ),
        }

    (args.out / "separability_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

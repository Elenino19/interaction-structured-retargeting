#!/usr/bin/env python3
"""Strong independent object-aware baseline on the frozen ball + frozen reference.

Two per-robot solvers, no communication, each given: the frozen Full ball
(verbatim), its own event/active-foot annotations, the shared accepted IK
reference targets, the same G1 model, support terms, initialisation and 340
Adam iterations — plus an explicit object term pulling the annotated toe to the
ball surface (||toe − ball|| = 0.11 m) at its own annotated frames. This is
strictly richer than the accepted Full IK, whose loss never references the ball.

Nothing frozen is written. Outputs go to
paper_eval/object_aware_comparison/independent_object_aware/<variant>/.
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
import sys
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
    contact_windows,
    generic_support_windows,
    ladder_geometry,
    retarget_soccer,
    save_result,
)
from run_fixed_ball_event_ablation import build_scene_only, render_variant, sha256  # noqa: E402

from isr_assets import ensure_join_on_path, g1_urdf, g1_xml
ensure_join_on_path()
from core4d_dual_retarget.coordinates import points_core4d_to_robot  # noqa: E402
from core4d_dual_retarget.retarget.independent import IndependentRetargetConfig, _yaw_from_shoulders  # noqa: E402

VARIANTS = {
    # name: (event IK on?, ball weight)
    "obj_w30_event_on": (True, 30.0),
    "obj_w10_event_on": (True, 10.0),
    "obj_w100_event_on": (True, 100.0),
    "obj_w30_event_off": (False, 30.0),
}
BALL_DISTANCE_M = 0.11


def yup_to_robot(points: np.ndarray) -> np.ndarray:
    return np.stack([points[..., 0], -points[..., 2], points[..., 1]], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--root", type=Path, default=Path("data/exports/touch_ladder_demo"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/paper_eval/object_aware_comparison/independent_object_aware"),
    )
    parser.add_argument("--urdf", type=Path, default=g1_urdf())
    parser.add_argument("--xml", type=Path, default=g1_xml())
    parser.add_argument("--iterations", type=int, default=340)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    guard_release_writes(args)

    project_root = args.project_root.resolve()
    root = args.root if args.root.is_absolute() else project_root / args.root
    output = args.output if args.output.is_absolute() else project_root / args.output
    output.mkdir(parents=True, exist_ok=True)

    frozen_ball_path = root / "ball_cones.npz"
    frozen_ball_sha = sha256(frozen_ball_path)
    ball_robot = yup_to_robot(np.load(frozen_ball_path)["ball_xyz_yup"])
    contacts = json.loads((root / "ball_cones.json").read_text())["contacts"]
    packed = np.load(root / "smplx_joints.npz")
    source = {"a": np.asarray(packed["player_a_joints"]), "b": np.asarray(packed["player_b_joints"])}
    accepted = {p: {k: np.array(v) for k, v in np.load(root / "g1" / f"robot_{p}.npz").items()} for p in "ab"}
    n = min(len(source["a"]), len(source["b"]), len(ball_robot))
    _, gap_lanes = ladder_geometry()
    station_events = _contact_stations(contacts, gap_lanes)
    pelvis_i = KEYPOINT_NAMES.index("pelvis")
    toe_index = {"left": KEYPOINT_NAMES.index("left_toe"), "right": KEYPOINT_NAMES.index("right_toe")}

    config = IndependentRetargetConfig(
        iterations=args.iterations, learning_rate=0.04, keypoint_weight=1.0,
        root_regularization_weight=0.35, velocity_weight=0.035, acceleration_weight=0.08,
    )
    manifest = {}
    for variant in args.variants:
        event_on, weight = VARIANTS[variant]
        vdir = output / variant
        if args.force and vdir.exists():
            shutil.rmtree(vdir)
        vdir.mkdir(parents=True, exist_ok=True)
        if not (vdir / "robot_a.npz").exists():
            scales = {}
            for player in "ab":
                joints = source[player][:n].copy()
                _, step_plant = _apply_player_plan(joints, player, station_events, gap_lanes)
                targets = accepted[player]["target_keypoints"][:n]
                precomputed = (targets, targets[:, pelvis_i].copy(),
                               _yaw_from_shoulders(points_core4d_to_robot(joints)), float(accepted[player]["scale"]))
                if event_on:
                    kick, plant = contact_windows(contacts, player, n, step_plant=step_plant)
                    weights = SoccerRetargetWeights()
                else:
                    kick = np.zeros((n, 2))
                    plant = generic_support_windows(joints)
                    weights = SoccerRetargetWeights(kick_velocity=0.0, event_keypoint_boost=False,
                                                    event_smooth_relaxation=False)
                # This robot's own annotated events only (receive + kick).
                events = []
                for index, c in enumerate(contacts):
                    if c["player"] != player:
                        continue
                    if index > 0:
                        events.append((int(c["frame"]), toe_index[c["side"]]))
                    events.append((int(c.get("kick_frame", c["frame"])), toe_index[c.get("kick_side", c["side"])]))
                result = retarget_soccer(
                    joints, args.urdf, config, kick, plant, loss_weights=weights,
                    precomputed_targets=precomputed,
                    ball_contact={"ball": ball_robot[:n], "events": events, "weight": weight, "distance": BALL_DISTANCE_M},
                )
                save_result(result, vdir / f"robot_{player}.npz")
                scales[f"robot_{player}"] = float(result.scale)
            (vdir / "scales.json").write_text(json.dumps(scales, indent=2) + "\n")
            shutil.copy2(frozen_ball_path, vdir / "ball_cones.npz")
        if sha256(vdir / "ball_cones.npz") != frozen_ball_sha:
            raise RuntimeError("ball copy differs from frozen ball")
        if not args.skip_render and not (vdir / "preview.mp4").exists():
            render_variant(project_root, vdir)
        if not (vdir / "soccer_scene.xml").exists():
            build_scene_only(project_root, vdir, args.xml)
        meta = {
            "variant": variant, "event_ik": event_on, "ball_term_weight": weight,
            "ball_term_distance_m": BALL_DISTANCE_M,
            "ball_term_frames": "this robot's annotated receive and kick frames only",
            "reference": "accepted Full IK target keypoints (shared verbatim)",
            "ball": {"sha256": frozen_ball_sha, "note": "frozen Full ball, verbatim"},
            "ik_config": asdict(config), "communication_between_robots": "none",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (vdir / "experiment.json").write_text(json.dumps(meta, indent=2) + "\n")
        manifest[variant] = meta
        print("completed", variant)
    (output / "manifest.json").write_text(json.dumps({
        "experiment": "independent_object_aware_baseline", "frozen_ball_sha256": frozen_ball_sha,
        "predefined_primary": "obj_w30_event_on", "variants": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()

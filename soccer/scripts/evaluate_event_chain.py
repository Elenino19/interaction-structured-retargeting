#!/usr/bin/env python3
"""Task-level event-chain consistency evaluator (kinematic, single sequence).

Reads frozen event annotations and any number of (robot_a, robot_b, ball)
variants and reports:

* per-event foot association (receive and kick kept separate),
* event timing under the existing +/-4 frame earliest-minimum rule,
* cross-agent kick -> next-receive transfer-chain consistency
  (foot-role-only and geometry-gated, with a predefined threshold sweep),
* foot-ball surface relation: ball-centre-to-toe distance, signed
  point-to-surface deviation, nearest foot-mesh vertex clearance proxy,
* support sliding, floor penetration, root/joint jerk (reused definitions).

All numbers are kinematic event-consistency metrics on a prescribed,
event/scene-conditioned ball path. They are not physical task success rates.
"""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_ablation_experiments import (  # noqa: E402
    BALL_RADIUS_M,
    evaluate_variant,
    load_robot,
    yup_to_robot,
)
from evaluate_soccer_paper import (  # noqa: E402
    FPS,
    TIMING_WINDOW_RADIUS_FRAMES,
    sha256,
    smoothness_metrics,
    support_and_floor_metrics,
)

FEET = (("a", "left"), ("a", "right"), ("b", "left"), ("b", "right"))
THRESHOLDS_M = (0.03, 0.05, 0.08)
PRIMARY_THRESHOLD_M = 0.05
GATES = ("center", "mesh")

_FIXED = "data/exports/touch_ladder_demo/paper_eval/event_coordination_extension/fixed_ball_ablation/"
_INDEP = "data/exports/touch_ladder_demo/paper_eval/object_aware_comparison/independent_object_aware/"
DEFAULT_ENTRIES = {
    # label: (variant_dir, ball_source, stationing_matched_to_full_ball)
    "basic_ik": ("data/exports/touch_ladder_demo/experiments/ablation/basic_ik", "own_source_baseline_ball", False),
    "support": ("data/exports/touch_ladder_demo/experiments/ablation/support", "own_source_baseline_ball", False),
    "event_semantics": ("data/exports/touch_ladder_demo/experiments/ablation/event_semantics", "own_source_baseline_ball", False),
    "full_without_position": ("data/exports/touch_ladder_demo/paper_eval/position_ablation/full_without_position", "own_resnapped_ball", True),
    "full": ("data/exports/touch_ladder_demo/experiments/ablation/full", "frozen_full_ball", True),
    "fixed_ball/frozen_reference/event_off": (_FIXED + "frozen_reference/event_off", "frozen_full_ball", True),
    "fixed_ball/frozen_reference/event_on": (_FIXED + "frozen_reference/event_on", "frozen_full_ball", True),
    "fixed_ball/frozen_reference/event_on_full_recipe": (_FIXED + "frozen_reference/event_on_full_recipe", "frozen_full_ball", True),
    "fixed_ball/current_code/event_off": (_FIXED + "current_code/event_off", "frozen_full_ball", False),
    "fixed_ball/current_code/event_on": (_FIXED + "current_code/event_on", "frozen_full_ball", False),
    "fixed_ball/current_code/event_off_aligned": (_FIXED + "current_code/event_off_aligned", "frozen_full_ball", False),
    "fixed_ball/current_code/event_on_aligned": (_FIXED + "current_code/event_on_aligned", "frozen_full_ball", False),
    "independent_object_aware/obj_w30_event_on": (_INDEP + "obj_w30_event_on", "frozen_full_ball", True),
    "independent_object_aware/obj_w10_event_on": (_INDEP + "obj_w10_event_on", "frozen_full_ball", True),
    "independent_object_aware/obj_w100_event_on": (_INDEP + "obj_w100_event_on", "frozen_full_ball", True),
    "independent_object_aware/obj_w30_event_off": (_INDEP + "obj_w30_event_off", "frozen_full_ball", True),
}
COMPARISON_ENTRIES = (
    "full",
    "fixed_ball/frozen_reference/event_off",
    "fixed_ball/frozen_reference/event_on",
    "fixed_ball/frozen_reference/event_on_full_recipe",
    "independent_object_aware/obj_w30_event_on",
    "independent_object_aware/obj_w10_event_on",
    "independent_object_aware/obj_w100_event_on",
    "independent_object_aware/obj_w30_event_off",
)


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #
def annotated_events(contacts: list[dict]) -> list[dict]:
    """13 events: 6 receives (index > 0) and 7 kicks, in annotation order."""
    events = []
    for index, event in enumerate(contacts):
        if index > 0:
            events.append(
                {
                    "event_id": f"e{index:02d}_recv",
                    "contact_index": index,
                    "phase": "receive",
                    "player": event["player"],
                    "frame": int(event["frame"]),
                    "foot": event["side"],
                }
            )
        events.append(
            {
                "event_id": f"e{index:02d}_kick",
                "contact_index": index,
                "phase": "kick",
                "player": event["player"],
                "frame": int(event.get("kick_frame", event["frame"])),
                "foot": event.get("kick_side", event["side"]),
            }
        )
    return events


def transfer_pairs(contacts: list[dict]) -> list[dict]:
    """Source-annotated kick -> next receive pairs (always cross-agent here)."""
    pairs = []
    for index in range(len(contacts) - 1):
        kick = contacts[index]
        recv = contacts[index + 1]
        pairs.append(
            {
                "pair_id": f"p{index + 1}",
                "kick_event_id": f"e{index:02d}_kick",
                "receive_event_id": f"e{index + 1:02d}_recv",
                "sender": kick["player"],
                "sender_foot": kick.get("kick_side", kick["side"]),
                "kick_frame": int(kick.get("kick_frame", kick["frame"])),
                "receiver": recv["player"],
                "receiver_foot": recv["side"],
                "receive_frame": int(recv["frame"]),
                "cross_agent": kick["player"] != recv["player"],
            }
        )
    return pairs


# --------------------------------------------------------------------------- #
# Mesh probe (official G1 foot meshes via MuJoCo)
# --------------------------------------------------------------------------- #
class MeshProbe:
    def __init__(self, scene: Path, robots: dict[str, dict[str, np.ndarray]]):
        import mujoco

        from isr_assets import ensure_join_on_path
        ensure_join_on_path()
        from core4d_dual_retarget.mujoco_scene import _set_robot_qpos

        self.mujoco = mujoco
        self.set_qpos = _set_robot_qpos
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.robots = robots
        self.geoms: dict[tuple[str, str], int] = {}
        for player, prefix in (("a", "r1_"), ("b", "r2_")):
            for side in ("left", "right"):
                body = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{side}_ankle_roll_link"
                )
                matches = [
                    g
                    for g in range(self.model.ngeom)
                    if self.model.geom_bodyid[g] == body
                    and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                ]
                if not matches:
                    raise RuntimeError(f"no foot mesh for {prefix}{side}")
                self.geoms[(player, side)] = matches[0]
        self._frame = -1

    def _pose(self, frame: int) -> None:
        if frame == self._frame:
            return
        for player, prefix in (("a", "r1_"), ("b", "r2_")):
            robot = self.robots[player]
            self.set_qpos(
                self.model,
                self.data,
                prefix,
                robot["joint_names"],
                robot["joint_positions"][frame],
                robot["root_translation"][frame],
                float(robot["root_yaw"][frame]),
            )
        self.mujoco.mj_forward(self.model, self.data)
        self._frame = frame

    def vertices(self, frame: int, foot: tuple[str, str]) -> np.ndarray:
        self._pose(frame)
        geom = self.geoms[foot]
        mesh = int(self.model.geom_dataid[geom])
        start = int(self.model.mesh_vertadr[mesh])
        count = int(self.model.mesh_vertnum[mesh])
        local = self.model.mesh_vert[start : start + count]
        rotation = self.data.geom_xmat[geom].reshape(3, 3)
        return local @ rotation.T + self.data.geom_xpos[geom]

    def ball_gap(self, frame: int, foot: tuple[str, str], ball: np.ndarray) -> float:
        verts = self.vertices(frame, foot)
        return float(np.min(np.linalg.norm(verts - ball, axis=1)) - BALL_RADIUS_M)

    def ground_depth(self, frame: int, foot: tuple[str, str]) -> float:
        verts = self.vertices(frame, foot)
        return max(0.0, -float(np.min(verts[:, 2])))


# --------------------------------------------------------------------------- #
# Per-variant evaluation
# --------------------------------------------------------------------------- #
def _toe(robot: dict[str, np.ndarray], side: str) -> np.ndarray:
    names = list(robot["keypoint_names"])
    return robot["robot_keypoints"][:, names.index(f"{side}_toe")]


def kick_velocity_fidelity(
    robots: dict[str, dict[str, np.ndarray]],
    contacts: list[dict],
    before: int = 8,
    after: int = 2,
) -> dict:
    """Motion-fidelity supplement (NOT an event-consistency metric).

    How well the robot's annotated kick-foot toe velocity tracks the reference
    (stored ``target_keypoints``) in the kick window kick-8..kick+2.
    """
    errors = []
    peak_robot = []
    peak_ref = []
    for event in contacts:
        robot = robots[event["player"]]
        names = list(robot["keypoint_names"])
        idx = names.index(f"{event.get('kick_side', event['side'])}_toe")
        frame = int(event.get("kick_frame", event["frame"]))
        n = len(robot["robot_keypoints"])
        lo, hi = max(0, frame - before), min(n, frame + after + 1)
        pred = np.diff(robot["robot_keypoints"][lo:hi, idx], axis=0) * FPS
        ref = np.diff(robot["target_keypoints"][lo:hi, idx], axis=0) * FPS
        errors.append(np.linalg.norm(pred - ref, axis=1))
        peak_robot.append(float(np.linalg.norm(pred, axis=1).max()))
        peak_ref.append(float(np.linalg.norm(ref, axis=1).max()))
    err = np.concatenate(errors)
    return {
        "velocity_rmse_m_s": float(np.sqrt(np.mean(err**2))),
        "peak_speed_robot_mean_m_s": float(np.mean(peak_robot)),
        "peak_speed_reference_mean_m_s": float(np.mean(peak_ref)),
    }


def evaluate_entry(
    label: str,
    variant_dir: Path,
    contacts: list[dict],
    ball_source: str,
    stationing_matched: bool,
    use_mesh: bool = True,
) -> dict:
    robots = {
        "a": load_robot(variant_dir / "robot_a.npz"),
        "b": load_robot(variant_dir / "robot_b.npz"),
    }
    env = np.load(variant_dir / "ball_cones.npz")
    ball = yup_to_robot(env["ball_xyz_yup"])
    n = min(len(ball), *(len(r["robot_keypoints"]) for r in robots.values()))
    toes = {foot: _toe(robots[foot[0]], foot[1])[:n] for foot in FEET}
    dist = {foot: np.linalg.norm(ball[:n] - toes[foot], axis=1) for foot in FEET}

    probe = None
    mesh_note = "disabled"
    if use_mesh:
        scene = variant_dir / "soccer_scene.xml"
        if scene.exists():
            try:
                probe = MeshProbe(scene, robots)
                mesh_note = "ok"
            except Exception as exc:  # noqa: BLE001
                mesh_note = f"mesh probe failed: {exc}"
        else:
            mesh_note = f"missing {scene.name}"

    # ---- per-event rows ---------------------------------------------------
    event_rows = []
    predicted_frames: dict[str, int] = {}
    for event in annotated_events(contacts):
        frame = event["frame"]
        target_foot = (event["player"], event["foot"])
        opposite_foot = (event["player"], "right" if event["foot"] == "left" else "left")
        d_target = float(dist[target_foot][frame])
        d_opposite = float(dist[opposite_foot][frame])
        all_d = {foot: float(dist[foot][frame]) for foot in FEET}
        nearest = min(all_d, key=all_d.get)
        other_player = "b" if event["player"] == "a" else "a"
        d_other_agent_min = min(all_d[(other_player, s)] for s in ("left", "right"))
        lo = max(0, frame - TIMING_WINDOW_RADIUS_FRAMES)
        hi = min(n, frame + TIMING_WINDOW_RADIUS_FRAMES + 1)
        predicted = lo + int(np.argmin(dist[target_foot][lo:hi]))
        predicted_frames[event["event_id"]] = predicted
        surface_dev = d_target - BALL_RADIUS_M
        row = {
            "variant": label,
            **event,
            "predicted_frame": predicted,
            "timing_error_frames": abs(predicted - frame),
            "timing_error_s": abs(predicted - frame) / FPS,
            "ball_center_to_target_toe_m": d_target,
            "ball_center_to_opposite_toe_m": d_opposite,
            "target_toe_surface_deviation_signed_m": surface_dev,
            "target_toe_surface_deviation_abs_m": abs(surface_dev),
            "toe_center_inside_ball": bool(surface_dev < 0.0),
            "correct_foot_association": bool(d_target < d_opposite),
            "nearest_of_four_feet": f"{nearest[0]}_{nearest[1]}",
            "actor_assignment_correct": bool(nearest[0] == event["player"]),
            "foot_assignment_correct_4way": bool(nearest == target_foot),
            "other_agent_min_toe_distance_m": d_other_agent_min,
            "actor_margin_m": d_other_agent_min - min(d_target, d_opposite),
        }
        if probe is not None:
            gaps = {foot: probe.ball_gap(frame, foot, ball[frame]) for foot in FEET}
            nearest_mesh = min(gaps, key=gaps.get)
            row.update(
                {
                    "mesh_target_foot_gap_signed_m": gaps[target_foot],
                    "mesh_target_foot_gap_abs_m": abs(gaps[target_foot]),
                    "mesh_opposite_foot_gap_signed_m": gaps[opposite_foot],
                    "mesh_target_penetrates_ball": bool(gaps[target_foot] < 0.0),
                    "mesh_nearest_of_four_feet": f"{nearest_mesh[0]}_{nearest_mesh[1]}",
                    "mesh_foot_assignment_correct_4way": bool(nearest_mesh == target_foot),
                    "mesh_correct_foot_association": bool(
                        gaps[target_foot] < gaps[opposite_foot]
                    ),
                    "mesh_target_foot_ground_depth_m": probe.ground_depth(frame, target_foot),
                }
            )
        else:
            row.update(
                {
                    "mesh_target_foot_gap_signed_m": float("nan"),
                    "mesh_target_foot_gap_abs_m": float("nan"),
                    "mesh_opposite_foot_gap_signed_m": float("nan"),
                    "mesh_target_penetrates_ball": None,
                    "mesh_nearest_of_four_feet": "",
                    "mesh_foot_assignment_correct_4way": None,
                    "mesh_correct_foot_association": None,
                    "mesh_target_foot_ground_depth_m": float("nan"),
                }
            )
        for tau in THRESHOLDS_M:
            key = f"{tau:.2f}".replace(".", "p")
            row[f"center_gate_pass_tau{key}"] = bool(abs(surface_dev) <= tau)
            row[f"mesh_gate_pass_tau{key}"] = (
                bool(abs(row["mesh_target_foot_gap_signed_m"]) <= tau)
                if probe is not None
                else None
            )
        event_rows.append(row)
    by_id = {row["event_id"]: row for row in event_rows}

    # ---- transfer pairs ---------------------------------------------------
    pair_rows = []
    for pair in transfer_pairs(contacts):
        kick = by_id[pair["kick_event_id"]]
        recv = by_id[pair["receive_event_id"]]
        role_ok = bool(
            kick["actor_assignment_correct"]
            and recv["actor_assignment_correct"]
            and kick["foot_assignment_correct_4way"]
            and recv["foot_assignment_correct_4way"]
            and kick["predicted_frame"] < recv["predicted_frame"]
        )
        row = {
            "variant": label,
            **pair,
            "sender_assigned": kick["actor_assignment_correct"],
            "receiver_assigned": recv["actor_assignment_correct"],
            "sender_foot_matched": kick["foot_assignment_correct_4way"],
            "receiver_foot_matched": recv["foot_assignment_correct_4way"],
            "predicted_kick_frame": kick["predicted_frame"],
            "predicted_receive_frame": recv["predicted_frame"],
            "order_ok": bool(kick["predicted_frame"] < recv["predicted_frame"]),
            "kick_surface_deviation_signed_m": kick["target_toe_surface_deviation_signed_m"],
            "receive_surface_deviation_signed_m": recv["target_toe_surface_deviation_signed_m"],
            "kick_mesh_gap_signed_m": kick["mesh_target_foot_gap_signed_m"],
            "receive_mesh_gap_signed_m": recv["mesh_target_foot_gap_signed_m"],
            "kick_actor_margin_m": kick["actor_margin_m"],
            "receive_actor_margin_m": recv["actor_margin_m"],
            "role_chain_consistent": role_ok,
        }
        for gate in GATES:
            for tau in THRESHOLDS_M:
                key = f"{tau:.2f}".replace(".", "p")
                k_pass = kick[f"{gate}_gate_pass_tau{key}"]
                r_pass = recv[f"{gate}_gate_pass_tau{key}"]
                both = None if (k_pass is None or r_pass is None) else bool(k_pass and r_pass)
                row[f"{gate}_geometry_ok_tau{key}"] = both
                row[f"{gate}_gated_chain_consistent_tau{key}"] = (
                    None if both is None else bool(role_ok and both)
                )
        pair_rows.append(row)

    # ---- aggregate --------------------------------------------------------
    recv_rows = [r for r in event_rows if r["phase"] == "receive"]
    kick_rows = [r for r in event_rows if r["phase"] == "kick"]

    def rate(rows: list[dict], key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    def mean(rows: list[dict], key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    def maxv(rows: list[dict], key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.max(vals)) if vals else float("nan")

    support = support_and_floor_metrics(robots, contacts)
    smooth = smoothness_metrics(robots)
    kick_fidelity = kick_velocity_fidelity(robots, contacts)
    legacy = evaluate_variant(variant_dir, contacts)
    mesh_ground_max = float("nan")
    mesh_ground_mean = float("nan")
    if probe is not None:
        depths = [probe.ground_depth(f, foot) for f in range(n) for foot in FEET]
        mesh_ground_max = float(np.max(depths))
        mesh_ground_mean = float(np.mean(depths))

    summary = {
        "variant": label,
        "ball_source": ball_source,
        "stationing_matched_to_frozen_ball": stationing_matched,
        "n_frames": n,
        "n_receive_events": len(recv_rows),
        "n_kick_events": len(kick_rows),
        "n_transfer_pairs": len(pair_rows),
        "n_sequences": 1,
        "receive_correct_foot_association_rate": rate(recv_rows, "correct_foot_association"),
        "kick_correct_foot_association_rate": rate(kick_rows, "correct_foot_association"),
        "overall_correct_foot_association_rate": rate(event_rows, "correct_foot_association"),
        "receive_actor_assignment_rate": rate(recv_rows, "actor_assignment_correct"),
        "kick_actor_assignment_rate": rate(kick_rows, "actor_assignment_correct"),
        "receive_foot_assignment_4way_rate": rate(recv_rows, "foot_assignment_correct_4way"),
        "kick_foot_assignment_4way_rate": rate(kick_rows, "foot_assignment_correct_4way"),
        "receive_timing_error_mean_frames": mean(recv_rows, "timing_error_frames"),
        "kick_timing_error_mean_frames": mean(kick_rows, "timing_error_frames"),
        "receive_timing_error_max_frames": maxv(recv_rows, "timing_error_frames"),
        "kick_timing_error_max_frames": maxv(kick_rows, "timing_error_frames"),
        "receive_target_toe_distance_mean_m": mean(recv_rows, "ball_center_to_target_toe_m"),
        "kick_target_toe_distance_mean_m": mean(kick_rows, "ball_center_to_target_toe_m"),
        "overall_target_toe_distance_mean_m": mean(event_rows, "ball_center_to_target_toe_m"),
        "overall_opposite_toe_distance_mean_m": mean(event_rows, "ball_center_to_opposite_toe_m"),
        "surface_deviation_signed_mean_m": mean(event_rows, "target_toe_surface_deviation_signed_m"),
        "surface_deviation_abs_mean_m": mean(event_rows, "target_toe_surface_deviation_abs_m"),
        "surface_deviation_abs_max_m": maxv(event_rows, "target_toe_surface_deviation_abs_m"),
        "toe_center_inside_ball_count": int(sum(bool(r["toe_center_inside_ball"]) for r in event_rows)),
        "mesh_gap_signed_mean_m": mean(event_rows, "mesh_target_foot_gap_signed_m"),
        "mesh_gap_abs_mean_m": mean(event_rows, "mesh_target_foot_gap_abs_m"),
        "mesh_gap_abs_max_m": maxv(event_rows, "mesh_target_foot_gap_abs_m"),
        "mesh_ball_penetration_count": (
            int(sum(bool(r["mesh_target_penetrates_ball"]) for r in event_rows))
            if probe is not None
            else None
        ),
        "mesh_foot_assignment_4way_rate": rate(event_rows, "mesh_foot_assignment_correct_4way"),
        "role_chain_consistency_rate": rate(pair_rows, "role_chain_consistent"),
        "role_chain_all_pairs_consistent": bool(all(r["role_chain_consistent"] for r in pair_rows)),
        "order_ok_rate": rate(pair_rows, "order_ok"),
        "support_foot_displacement_mean_m": support["stance_foot_horizontal_displacement_mean_m"],
        "support_foot_speed_p95_m_s": support["stance_foot_speed_p95_m_per_s"],
        "keypoint_floor_penetration_max_m": support["floor_penetration_keypoint_max_m"],
        "mesh_floor_penetration_max_m": mesh_ground_max,
        "mesh_floor_penetration_mean_m": mesh_ground_mean,
        "root_jerk_rms_m_s3": smooth["root_jerk_rms_m_per_s3"],
        "joint_jerk_rms_rad_s3": smooth["joint_jerk_rms_rad_per_s3"],
        "root_acceleration_rms_m_s2": smooth["root_acceleration_rms_m_per_s2"],
        "joint_acceleration_rms_rad_s2": smooth["joint_acceleration_rms_rad_per_s2"],
        "kick_toe_velocity_tracking_rmse_m_s_MOTION_FIDELITY": kick_fidelity["velocity_rmse_m_s"],
        "kick_toe_peak_speed_robot_mean_m_s_MOTION_FIDELITY": kick_fidelity["peak_speed_robot_mean_m_s"],
        "kick_toe_peak_speed_reference_mean_m_s_MOTION_FIDELITY": kick_fidelity["peak_speed_reference_mean_m_s"],
        "lane_crossing_error_mean_m_BALL_PROPERTY": legacy["passes"]["lane_crossing_error_mean_m"],
        "cone_surface_clearance_min_m_BALL_PROPERTY": legacy["passes"]["cone_surface_clearance_min_m"],
        "mesh_status": mesh_note,
    }
    for gate in GATES:
        for tau in THRESHOLDS_M:
            key = f"{tau:.2f}".replace(".", "p")
            summary[f"{gate}_gate_event_pass_rate_tau{key}"] = rate(event_rows, f"{gate}_gate_pass_tau{key}")
            summary[f"{gate}_gated_chain_consistency_rate_tau{key}"] = rate(
                pair_rows, f"{gate}_gated_chain_consistent_tau{key}"
            )
    ball_hash = sha256(variant_dir / "ball_cones.npz")
    return {
        "summary": summary,
        "events": event_rows,
        "pairs": pair_rows,
        "support": support,
        "smoothness": smooth,
        "inputs": {
            "variant_dir": str(variant_dir),
            "robot_a_sha256": sha256(variant_dir / "robot_a.npz"),
            "robot_b_sha256": sha256(variant_dir / "robot_b.npz"),
            "ball_sha256": ball_hash,
        },
        "distance_curves": {f"{p}_{s}": dist[(p, s)].tolist() for p, s in FEET},
        "predicted_frames": predicted_frames,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contacts",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/ball_cones.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/paper_eval/event_coordination_extension"),
    )
    parser.add_argument(
        "--entries",
        nargs="*",
        default=[k for k in DEFAULT_ENTRIES if not k.startswith("independent_object_aware/")],
        help="Variant labels; 'COMPARISON' expands to the object-aware comparison set.",
    )
    parser.add_argument("--skip-mesh", action="store_true")
    args = parser.parse_args()
    guard_release_writes(args)
    args.out.mkdir(parents=True, exist_ok=True)
    contacts = json.loads(args.contacts.read_text())["contacts"]
    if args.entries == ["COMPARISON"]:
        args.entries = list(COMPARISON_ENTRIES)

    results = {}
    event_rows: list[dict] = []
    pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    for label in args.entries:
        variant_dir, ball_source, matched = DEFAULT_ENTRIES[label]
        variant_dir = Path(variant_dir)
        if not (variant_dir / "robot_a.npz").exists():
            print("skip missing", label, variant_dir)
            continue
        result = evaluate_entry(
            label, variant_dir, contacts, ball_source, matched, use_mesh=not args.skip_mesh
        )
        results[label] = result
        event_rows.extend(result["events"])
        pair_rows.extend(result["pairs"])
        summary_rows.append(result["summary"])
        print(
            label,
            "recv_assoc=%.3f kick_assoc=%.3f role_chain=%.3f mesh=%s"
            % (
                result["summary"]["receive_correct_foot_association_rate"],
                result["summary"]["kick_correct_foot_association_rate"],
                result["summary"]["role_chain_consistency_rate"],
                result["summary"]["mesh_status"],
            ),
        )

    write_csv(args.out / "event_metrics.csv", event_rows)
    write_csv(args.out / "transfer_chain_metrics.csv", pair_rows)
    write_csv(args.out / "event_chain_summary.csv", summary_rows)

    fixed = [row for row in summary_rows if row["variant"].startswith("fixed_ball/") or row["variant"] == "full"]
    write_csv(args.out / "fixed_ball_ablation.csv", fixed)

    sensitivity = []
    for row in summary_rows:
        for gate in GATES:
            for tau in THRESHOLDS_M:
                key = f"{tau:.2f}".replace(".", "p")
                sensitivity.append(
                    {
                        "variant": row["variant"],
                        "gate": gate,
                        "threshold_m": tau,
                        "event_pass_rate": row[f"{gate}_gate_event_pass_rate_tau{key}"],
                        "geometry_gated_chain_consistency_rate": row[
                            f"{gate}_gated_chain_consistency_rate_tau{key}"
                        ],
                        "role_only_chain_consistency_rate": row["role_chain_consistency_rate"],
                        "n_events": 13,
                        "n_pairs": 6,
                    }
                )
    write_csv(args.out / "threshold_sensitivity.csv", sensitivity)

    payload = {
        "schema_version": "1.0.0",
        "sequence_id": "touch_ladder_demo",
        "fps": FPS,
        "ball_radius_m": BALL_RADIUS_M,
        "timing_window_radius_frames": TIMING_WINDOW_RADIUS_FRAMES,
        "thresholds_m": list(THRESHOLDS_M),
        "primary_threshold_m": PRIMARY_THRESHOLD_M,
        "contacts_sha256": sha256(args.contacts),
        "metric_class": (
            "kinematic event-consistency metrics on a prescribed event/scene-"
            "conditioned ball path; not physical soccer task success"
        ),
        "sample_size_note": (
            "One 144-frame sequence; 6 receive + 7 kick events and 6 transfer pairs "
            "are dependent samples from that single sequence."
        ),
        "variants": {
            label: {k: v for k, v in result.items() if k != "distance_curves"}
            for label, result in results.items()
        },
        "distance_curves": {label: result["distance_curves"] for label, result in results.items()},
    }
    (args.out / "event_chain_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

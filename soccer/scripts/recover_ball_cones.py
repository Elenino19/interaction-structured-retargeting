#!/usr/bin/env python3
"""Recover a soccer ball trajectory and five disc cones from the source clip.

Cones are static environment. The ball is a free object reconstructed from
YOLO 2D detections, a ground homography from ankles+cones, and contact
heights. This is a geometric estimate, not mocap.
"""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import json
from pathlib import Path
import sys as _isr_sys
_isr_sys.path.insert(0, str(Path(__file__).resolve().parent))
from isr_assets import ensure_join_on_path, g1_urdf, g1_xml, gvhmr_root, yolo_weights

import cv2
import numpy as np
from ultralytics import YOLO


BALL_RADIUS_M = 0.11
CONE_COUNT = 5
CONE_SPACING_M = 0.50


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def detect_people_and_ball(model: YOLO, frame: np.ndarray):
    result = model.predict(frame, verbose=False, conf=0.25)[0]
    people = []
    balls = []
    if result.boxes is None:
        return people, balls
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    for box, c, cf in zip(xyxy, cls, conf):
        if int(c) == 0:
            people.append((box, float(cf)))
        elif int(c) == 32:
            balls.append((box, float(cf)))
    people.sort(key=lambda item: 0.5 * (item[0][0] + item[0][2]))
    return people, balls


def cone_centers(frame: np.ndarray, people: list) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 90, 90), (18, 255, 255)) | cv2.inRange(
        hsv, (165, 90, 90), (180, 255, 255)
    )
    mask = cv2.medianBlur(mask, 5)
    person_mask = np.zeros(mask.shape, dtype=np.uint8)
    for box, _ in people:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(person_mask, (x1, y1), (x2, y2), 255, -1)
    mask[person_mask > 0] = 0
    mask[: int(0.50 * mask.shape[0])] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 35 or area > 900:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if y < 0.55 * mask.shape[0]:
            continue
        if h == 0 or w / h > 2.4 or h / max(w, 1) > 2.5:
            continue
        blobs.append((x + 0.5 * w, y + 0.5 * h, area))
    blobs.sort(key=lambda item: item[1], reverse=True)
    if len(blobs) < CONE_COUNT:
        return np.zeros((0, 2))
    chosen = np.array([[b[0], b[1]] for b in blobs[:CONE_COUNT]], dtype=np.float64)
    order = np.argsort(chosen[:, 1])[::-1]
    return chosen[order]


def point_segment_distance(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    segment = ends - starts
    alpha = np.sum((points - starts) * segment, axis=-1) / np.maximum(
        np.sum(segment * segment, axis=-1), 1e-8
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    closest = starts + alpha[:, None] * segment
    return np.linalg.norm(points - closest, axis=-1), alpha


def detect_contacts(
    ball_uv: np.ndarray,
    feet_a_uv: np.ndarray,
    feet_b_uv: np.ndarray,
    want: int = 7,
) -> list[dict]:
    """Find alternating contacts from source-image ball-to-foot distance.

    The earlier detector selected 3D ankle-speed peaks, which can occur many
    frames after impact. Here each foot is the projected ankle-to-toe segment
    in the original video. A small dynamic program chooses the alternating
    seven-contact sequence while preserving realistic 12--28 frame gaps.
    """
    n = len(ball_uv)
    feet = {"a": feet_a_uv[:n], "b": feet_b_uv[:n]}
    distances: dict[tuple[str, str], np.ndarray] = {}
    alphas: dict[tuple[str, str], np.ndarray] = {}
    scales: dict[str, float] = {}
    for player in ("a", "b"):
        tracks = feet[player]
        for side, ankle, toe in (("left", 0, 1), ("right", 2, 3)):
            distances[(player, side)], alphas[(player, side)] = point_segment_distance(
                ball_uv, tracks[:, ankle], tracks[:, toe]
            )
        scales[player] = float(
            np.median(
                np.concatenate(
                    [
                        np.linalg.norm(tracks[:, 1] - tracks[:, 0], axis=-1),
                        np.linalg.norm(tracks[:, 3] - tracks[:, 2], axis=-1),
                    ]
                )
            )
        )

    expected_players = ["a" if index % 2 == 0 else "b" for index in range(want)]
    candidates: list[list[tuple[int, str, float, float]]] = []
    for state, player in enumerate(expected_players):
        left = distances[(player, "left")]
        right = distances[(player, "right")]
        nearest = np.minimum(left, right)
        frames = [
            frame
            for frame in range(1, n - 1)
            if nearest[frame] <= nearest[frame - 1]
            and nearest[frame] <= nearest[frame + 1]
        ]
        if state == 0:
            if float(nearest[0]) <= 40.0:
                frames = [0] + frames
            frames = [frame for frame in frames if frame <= 24]
        elif state == want - 1:
            frames = [frame for frame in frames if frame >= n - 24]
        entries = []
        for frame in frames:
            side = "left" if left[frame] <= right[frame] else "right"
            distance = float(distances[(player, side)][frame])
            entries.append(
                (
                    frame,
                    side,
                    distance,
                    float(alphas[(player, side)][frame]),
                )
            )
        candidates.append(entries)

    # frame -> (cost, path), where each path item is frame/side/gap/alpha.
    states: dict[int, tuple[float, list[tuple[int, str, float, float]]]] = {}
    for item in candidates[0]:
        frame, _, gap_px, _ = item
        cost = gap_px / max(scales["a"], 1.0) + 0.02 * frame
        states[frame] = (cost, [item])
    for state in range(1, want):
        player = expected_players[state]
        next_states: dict[int, tuple[float, list[tuple[int, str, float, float]]]] = {}
        for item in candidates[state]:
            frame, _, gap_px, _ = item
            options = []
            for previous_frame, (cost, path) in states.items():
                interval = frame - previous_frame
                if 10 <= interval <= 36:
                    total = (
                        cost
                        + gap_px / max(scales[player], 1.0)
                        + 0.012 * (interval - 20) ** 2
                    )
                    options.append((total, path + [item]))
            if options:
                next_states[frame] = min(options, key=lambda value: value[0])
        states = next_states
    if not states:
        raise RuntimeError("Could not find an alternating seven-contact foot sequence")
    _, sequence = min(
        states.values(),
        key=lambda value: value[0] + 0.02 * abs((n - 5) - value[1][-1][0]),
    )

    contacts = []
    for player, (frame, side, gap_px, foot_alpha) in zip(
        expected_players, sequence, strict=True
    ):
        contacts.append(
            {
                "frame": int(frame),
                "player": player,
                "side": side,
                "foot_alpha": foot_alpha,
                "ball_foot_gap_px": gap_px,
                "score": float(1.0 / (1.0 + gap_px)),
            }
        )
    return contacts


def ball_from_contacts(
    n: int,
    contacts: list[dict],
    joints_a: np.ndarray,
    joints_b: np.ndarray,
    corridor_x: float,
) -> np.ndarray:
    ball = np.zeros((n, 3))
    players = {"a": joints_a, "b": joints_b}
    other = {"a": joints_b, "b": joints_a}
    ankle_idx = {"left": 7, "right": 8}
    waypoints = []
    for event in contacts:
        j = players[event["player"]]
        k = other[event["player"]]
        ankle = j[event["frame"], ankle_idx[event["side"]]]
        toward = np.mean(k[:, 0], axis=0) - np.mean(j[:, 0], axis=0)
        toward[1] = 0.0
        norm = np.linalg.norm(toward) + 1e-6
        pos = ankle.copy()
        pos += 0.14 * toward / norm
        pos[1] = BALL_RADIUS_M
        waypoints.append((event["frame"], pos))
    if not waypoints:
        raise RuntimeError("No contacts to build a pass trajectory")
    if waypoints[0][0] != 0:
        waypoints.insert(0, (0, waypoints[0][1].copy()))
    if waypoints[-1][0] != n - 1:
        waypoints.append((n - 1, waypoints[-1][1].copy()))
    frames = np.array([w[0] for w in waypoints], dtype=np.float64)
    pts = np.stack([w[1] for w in waypoints], axis=0)
    t = np.arange(n)
    for dim in range(3):
        ball[:, dim] = np.interp(t, frames, pts[:, dim])
    ball[:, 1] = BALL_RADIUS_M
    hold = {event["frame"] for event in contacts}
    smoothed = ball.copy()
    kernel = np.ones(5) / 5.0
    for dim in (0, 2):
        smoothed[:, dim] = np.convolve(
            np.pad(ball[:, dim], (2, 2), mode="edge"), kernel, mode="valid"
        )
    for frame in hold:
        smoothed[frame] = ball[frame]
    return smoothed


def homography(img_pts: np.ndarray, world_xz: np.ndarray) -> np.ndarray:
    return cv2.findHomography(img_pts.astype(np.float32), world_xz.astype(np.float32))[0]


def apply_h(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    pts = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
    mapped = pts @ H.T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return mapped


def interpolate_gaps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    idx = np.arange(len(values))
    if valid.sum() == 0:
        raise RuntimeError("No valid ball detections")
    for dim in range(values.shape[1]):
        values[:, dim] = np.interp(idx, idx[valid], values[valid, dim])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/clips/touch_ladder_demo_source.mp4"),
    )
    parser.add_argument(
        "--joints",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/smplx_joints.npz"),
    )
    parser.add_argument(
        "--yolo",
        type=Path,
        default=yolo_weights(),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/ball_cones.npz"),
    )
    parser.add_argument(
        "--foot-tracks",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/source_foot_tracks.npz"),
    )
    args = parser.parse_args()
    guard_release_writes(args)

    frames = read_frames(args.video)
    packed = np.load(args.joints)
    foot_tracks = np.load(args.foot_tracks)
    joints_a = packed["player_a_joints"]
    joints_b = packed["player_b_joints"]
    n = min(len(frames), len(joints_a), len(joints_b))
    frames = frames[:n]
    joints_a = joints_a[:n]
    joints_b = joints_b[:n]

    model = YOLO(str(args.yolo))
    ball_uv = np.full((n, 2), np.nan)
    person_u = np.full((n, 2), np.nan)
    cone_stack = []
    for i, frame in enumerate(frames):
        people, balls = detect_people_and_ball(model, frame)
        if len(people) >= 2:
            person_u[i] = [
                0.5 * (people[0][0][0] + people[0][0][2]),
                0.5 * (people[1][0][0] + people[1][0][2]),
            ]
        if balls:
            ball_uv[i] = [
                0.5 * (balls[0][0][0] + balls[0][0][2]),
                0.5 * (balls[0][0][1] + balls[0][0][3]),
            ]
        cones = cone_centers(frame, people)
        if len(cones) == CONE_COUNT:
            cone_stack.append(cones)

    valid = np.isfinite(ball_uv).all(axis=1)
    ball_uv = interpolate_gaps(np.nan_to_num(ball_uv, nan=0.0), valid)
    person_ok = np.isfinite(person_u).all(axis=1)
    person_u = interpolate_gaps(np.nan_to_num(person_u, nan=0.0), person_ok)

    median_cones = np.median(np.stack(cone_stack, axis=0), axis=0) if cone_stack else None
    if median_cones is None:
        raise SystemExit("Could not find five cones after masking players")

    ankle_a_xz = np.array(
        [np.mean(joints_a[:, [7, 8], 0]), np.mean(joints_a[:, [7, 8], 2])]
    )
    ankle_b_xz = np.array(
        [np.mean(joints_b[:, [7, 8], 0]), np.mean(joints_b[:, [7, 8], 2])]
    )
    cone_world = np.stack(
        [
            np.zeros(CONE_COUNT),
            np.linspace(0.0, CONE_SPACING_M * (CONE_COUNT - 1), CONE_COUNT),
        ],
        axis=1,
    )
    cone_world[:, 0] = 0.5 * (ankle_a_xz[0] + ankle_b_xz[0])
    cone_world[:, 1] += min(ankle_a_xz[1], ankle_b_xz[1]) - 0.4
    corridor_x = float(cone_world[0, 0])
    contacts = None
    for want in (8, 7, 6):
        try:
            contacts = detect_contacts(
                ball_uv,
                foot_tracks["player_a_foot_uv"],
                foot_tracks["player_b_foot_uv"],
                want=want,
            )
            break
        except RuntimeError:
            continue
    if contacts is None:
        raise SystemExit("Could not find an alternating foot-contact sequence")
    ball_xyz = ball_from_contacts(n, contacts, joints_a, joints_b, corridor_x)
    cones_xyz = np.stack(
        [cone_world[:, 0], np.full(CONE_COUNT, 0.05), cone_world[:, 1]],
        axis=1,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    contact_frames = np.array([c["frame"] for c in contacts], dtype=np.int32)
    np.savez_compressed(
        args.out,
        ball_xyz_yup=ball_xyz,
        cone_xyz_yup=cones_xyz,
        ball_uv=ball_uv,
        cone_uv=median_cones,
        ball_detected=valid,
        player_a_foot_uv=foot_tracks["player_a_foot_uv"][:n],
        player_b_foot_uv=foot_tracks["player_b_foot_uv"][:n],
        ball_radius_m=BALL_RADIUS_M,
        contact_frames=contact_frames,
        method="contact_snapped_ground_pass",
    )
    meta = {
        "ball_detected_frames": int(valid.sum()),
        "n_frames": n,
        "cone_count": CONE_COUNT,
        "contacts": contacts,
        "note": "Ball stays on the ground and is snapped to the kicking foot at detected pass contacts.",
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

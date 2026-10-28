#!/usr/bin/env python3
"""Cut the first Touch Ladder camera shot and make two player crops.

The 7MLC / CONTROLLO clip is local-demo only. It is not a redistributable
source. This script keeps 3:23–3:29 (203.00–209.00 s), the near-end ladder
passes, where the cone gaps are large in the frame.
"""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import json
import subprocess
from pathlib import Path
import sys as _isr_sys
_isr_sys.path.insert(0, str(Path(__file__).resolve().parent))
from isr_assets import ensure_join_on_path, g1_urdf, g1_xml, gvhmr_root, yolo_weights

import cv2
import numpy as np
from ultralytics import YOLO


CLIP_START_S = 203.0
CLIP_END_S = 209.0


def default_source_video() -> Path:
    """Human source clip. Not stored in this repository."""
    import os

    value = os.environ.get("SOCCER_SOURCE_VIDEO")
    if not value:
        raise SystemExit(
            "SOCCER_SOURCE_VIDEO is unset. The human source video is not redistributed."
        )
    return Path(value)


def cut_source(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = CLIP_END_S - CLIP_START_S
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(CLIP_START_S),
            "-t",
            f"{duration:.3f}",
            "-i",
            str(src),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            str(dst),
        ],
        check=True,
    )


def enlarge_xyxy(
    box: np.ndarray,
    width: int,
    height: int,
    scale: float,
    min_w: int = 420,
    min_h: int = 500,
) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(float(min_w), (x2 - x1) * scale)
    bh = max(float(min_h), (y2 - y1) * scale)
    nx1 = int(np.clip(cx - 0.5 * bw, 0, width - 1))
    ny1 = int(np.clip(cy - 0.5 * bh, 0, height - 1))
    nx2 = int(np.clip(cx + 0.5 * bw, nx1 + 1, width))
    ny2 = int(np.clip(cy + 0.5 * bh, ny1 + 1, height))
    return np.array([nx1, ny1, nx2, ny2], dtype=np.int32)


def two_person_boxes(result) -> tuple[np.ndarray, np.ndarray] | None:
    if result.boxes is None or len(result.boxes) < 2:
        return None
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    people = [xyxy[i] for i in range(len(xyxy)) if int(cls[i]) == 0 and conf[i] >= 0.35]
    if len(people) < 2:
        return None
    people.sort(key=lambda b: 0.5 * (b[0] + b[2]))
    return people[0], people[-1]


def crop_with_other_masked(
    frame: np.ndarray, keep: np.ndarray, other: np.ndarray
) -> np.ndarray:
    masked = frame.copy()
    ox1, oy1, ox2, oy2 = [int(v) for v in other]
    cv2.rectangle(masked, (ox1, oy1), (ox2, oy2), (0, 0, 0), thickness=-1)
    x1, y1, x2, y2 = [int(v) for v in keep]
    return masked[y1:y2, x1:x2]


def write_crops(src: Path, out_a: Path, out_b: Path, model: YOLO) -> dict:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_a: list[np.ndarray] = []
    frames_b: list[np.ndarray] = []
    last_pair = None
    heights_a = []
    heights_b = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, classes=[0], verbose=False, conf=0.35)[0]
        pair = two_person_boxes(result) or last_pair
        if pair is None:
            raise RuntimeError("Need two person detections to start the crops")
        last_pair = pair
        left, right = pair
        keep_a = enlarge_xyxy(left, width, height, 1.8)
        keep_b = enlarge_xyxy(right, width, height, 1.8)
        heights_a.append(float(left[3] - left[1]))
        heights_b.append(float(right[3] - right[1]))
        frames_a.append(crop_with_other_masked(frame, keep_a, right))
        frames_b.append(crop_with_other_masked(frame, keep_b, left))
    cap.release()

    def dump(frames: list[np.ndarray], path: Path) -> tuple[int, int]:
        max_h = max(f.shape[0] for f in frames)
        max_w = max(f.shape[1] for f in frames)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (max_w, max_h),
        )
        for crop in frames:
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            h, w = crop.shape[:2]
            y0 = (max_h - h) // 2
            x0 = (max_w - w) // 2
            canvas[y0 : y0 + h, x0 : x0 + w] = crop
            writer.write(canvas)
        writer.release()
        return max_w, max_h

    size_a = dump(frames_a, out_a)
    size_b = dump(frames_b, out_b)
    return {
        "n_frames": len(frames_a),
        "fps": fps,
        "source_wh": [width, height],
        "player_a_crop_wh": list(size_a),
        "player_b_crop_wh": list(size_b),
        "player_a_mean_height_px": float(np.mean(heights_a)),
        "player_b_mean_height_px": float(np.mean(heights_b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/clips"),
    )
    parser.add_argument("--yolo", type=Path, default=None)
    args = parser.parse_args()
    guard_release_writes(args)
    if args.source is None:
        args.source = default_source_video()
    if args.yolo is None:
        args.yolo = yolo_weights()

    sample_id = "touch_ladder_demo"
    src_cut = args.out_dir / f"{sample_id}_source.mp4"
    crop_a = args.out_dir / f"{sample_id}_player_a.mp4"
    crop_b = args.out_dir / f"{sample_id}_player_b.mp4"
    meta_path = args.out_dir / f"{sample_id}_prep.json"

    if not args.source.exists():
        raise SystemExit(f"Source video not found: {args.source}")
    cut_source(args.source, src_cut)
    model = YOLO(str(args.yolo))
    stats = write_crops(src_cut, crop_a, crop_b, model)
    meta = {
        "sample_id": sample_id,
        "task": "touch_ladder_first_touch_return",
        "role": "local_demo_only",
        "redistribute": False,
        "source": {
            "path": str(args.source),
            "attribution": "7MLC / MZC CONTROLLO training video (WeChat forward)",
            "license": "not granted for dataset release",
            "start_s": CLIP_START_S,
            "end_s": CLIP_END_S,
            "note": "Near-end touch ladder, 3:23-3:29. Local demo only.",
        },
        "contacts": {
            "description": "Near-end ladder: gap N2-N3, gap N1-N2, outside N1, back through N1-N2, then adjacent gaps.",
        },
        "crops": {
            "player_a": str(crop_a),
            "player_b": str(crop_b),
            "player_a_id": "left",
            "player_b_id": "right",
        },
        "prep": stats,
        "label_type": "gvhmr_pseudo_label_not_mocap",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

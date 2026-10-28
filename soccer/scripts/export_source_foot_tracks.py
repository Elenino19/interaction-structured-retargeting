#!/usr/bin/env python3
"""Project GVHMR incamera feet back into the uncropped source video."""

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

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def enlarge_xyxy(
    box: np.ndarray, width: int, height: int, scale: float = 1.8
) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    bw = max(420.0, (x2 - x1) * scale)
    bh = max(500.0, (y2 - y1) * scale)
    return np.array(
        [
            int(np.clip(cx - 0.5 * bw, 0, width - 1)),
            int(np.clip(cy - 0.5 * bh, 0, height - 1)),
            int(np.clip(cx + 0.5 * bw, 1, width)),
            int(np.clip(cy + 0.5 * bh, 1, height)),
        ],
        dtype=np.int32,
    )


def video_shape(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, frames


def incamera_feet(
    result_path: Path, gvhmr_root: Path
) -> np.ndarray:
    result = torch.load(result_path, map_location="cpu")
    sys.path.insert(0, str(gvhmr_root))
    previous = Path.cwd()
    os.chdir(gvhmr_root)
    try:
        from hmr4d.utils.smplx_utils import make_smplx

        model = make_smplx("supermotion")
        params = {
            key: value.float() for key, value in result["smpl_params_incam"].items()
        }
        with torch.no_grad():
            joints = model(**params).joints.cpu().numpy()
    finally:
        os.chdir(previous)
    # ankle, toe for anatomical left then right.
    xyz = joints[:, [7, 10, 8, 11]]
    intrinsics = result["K_fullimg"].cpu().numpy()
    uv = np.empty((*xyz.shape[:2], 2), dtype=np.float64)
    uv[..., 0] = (
        intrinsics[:, None, 0, 0] * xyz[..., 0] / xyz[..., 2]
        + intrinsics[:, None, 0, 2]
    )
    uv[..., 1] = (
        intrinsics[:, None, 1, 1] * xyz[..., 1] / xyz[..., 2]
        + intrinsics[:, None, 1, 2]
    )
    return uv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("data/clips/touch_ladder_demo_source.mp4")
    )
    parser.add_argument(
        "--crop-a", type=Path, default=Path("data/clips/touch_ladder_demo_player_a.mp4")
    )
    parser.add_argument(
        "--crop-b", type=Path, default=Path("data/clips/touch_ladder_demo_player_b.mp4")
    )
    parser.add_argument(
        "--result-a",
        type=Path,
        default=Path(
            "data/gvhmr/touch_ladder_demo/player_a/"
            "touch_ladder_demo_player_a/hmr4d_results.pt"
        ),
    )
    parser.add_argument(
        "--result-b",
        type=Path,
        default=Path(
            "data/gvhmr/touch_ladder_demo/player_b/"
            "touch_ladder_demo_player_b/hmr4d_results.pt"
        ),
    )
    parser.add_argument(
        "--yolo",
        type=Path,
        default=yolo_weights(),
    )
    parser.add_argument("--gvhmr-root", type=Path, default=gvhmr_root())
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/source_foot_tracks.npz"),
    )
    args = parser.parse_args()
    guard_release_writes(args)

    paths = [args.source, args.crop_a, args.crop_b, args.result_a, args.result_b]
    paths = [path.resolve() for path in paths]
    source, crop_a, crop_b, result_a, result_b = paths
    source_w, source_h, source_n = video_shape(source)
    crop_shapes = {
        "a": video_shape(crop_a)[:2],
        "b": video_shape(crop_b)[:2],
    }
    projected = {
        "a": incamera_feet(result_a, args.gvhmr_root.resolve()),
        "b": incamera_feet(result_b, args.gvhmr_root.resolve()),
    }
    frame_count = min(source_n, len(projected["a"]), len(projected["b"]))
    source_feet = {
        player: np.zeros((frame_count, 4, 2), dtype=np.float64)
        for player in ("a", "b")
    }

    detector = YOLO(str(args.yolo.resolve()))
    cap = cv2.VideoCapture(str(source))
    last_pair = None
    for frame in range(frame_count):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"Source ended at frame {frame}")
        result = detector.predict(image, classes=[0], verbose=False, conf=0.35)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        people = sorted(boxes, key=lambda box: 0.5 * (box[0] + box[2]))
        if len(people) >= 2:
            last_pair = (people[0], people[-1])
        if last_pair is None:
            raise RuntimeError("No two-person detection available")
        for player, box in zip(("a", "b"), last_pair, strict=True):
            x1, y1, x2, y2 = enlarge_xyxy(box, source_w, source_h)
            crop_w, crop_h = crop_shapes[player]
            pad_x = (crop_w - (x2 - x1)) // 2
            pad_y = (crop_h - (y2 - y1)) // 2
            source_feet[player][frame, :, 0] = (
                projected[player][frame, :, 0] - pad_x + x1
            )
            source_feet[player][frame, :, 1] = (
                projected[player][frame, :, 1] - pad_y + y1
            )
    cap.release()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        player_a_foot_uv=source_feet["a"],
        player_b_foot_uv=source_feet["b"],
        keypoint_names=np.asarray(
            ["left_ankle", "left_toe", "right_ankle", "right_toe"]
        ),
        coordinate_space=np.asarray("source_video_pixels"),
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()

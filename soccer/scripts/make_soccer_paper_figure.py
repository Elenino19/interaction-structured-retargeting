#!/usr/bin/env python3
"""Build publication-safe and local-only qualitative soccer figures."""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("basic_ik", "event_semantics", "full")
METHOD_LABELS = {
    "basic_ik": "Basic IK",
    "event_semantics": "+ Support & Event Semantics",
    "full": "Full",
}


def read_frame(path: Path, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame} from {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pad_source(image: np.ndarray, height: int = 720) -> tuple[np.ndarray, int]:
    if image.shape[0] >= height:
        return image, 0
    top = (height - image.shape[0]) // 2
    bottom = height - image.shape[0] - top
    return np.pad(image, ((top, bottom), (0, 0), (0, 0))), top


def interaction_sample(metrics: dict, method: str, phase: str, frame: int) -> dict:
    samples = metrics["variants"][method]["interaction"]["samples"]
    return next(
        sample
        for sample in samples
        if sample["phase"] == phase and sample["frame"] == frame
    )


def draw_source_annotation(
    image: np.ndarray,
    frame: int,
    player: str | None,
    foot: str | None,
    environment: np.lib.npyio.NpzFile,
    top_pad: int,
) -> np.ndarray:
    output = image.copy()
    ball = environment["ball_uv"][frame].astype(int)
    cv2.circle(
        output,
        (int(ball[0]), int(ball[1]) + top_pad),
        18,
        (255, 220, 0),
        4,
    )
    if player and foot:
        tracks = environment[f"player_{player}_foot_uv"]
        index = 1 if foot == "left" else 3
        toe = tracks[frame, index].astype(int)
        cv2.circle(
            output,
            (int(toe[0]), int(toe[1]) + top_pad),
            18,
            (0, 230, 255),
            4,
        )
    return output


def add_cell_label(axis, text: str, good: bool | None = None) -> None:
    color = "white" if good is None else ("#6dff8a" if good else "#ff786d")
    axis.text(
        0.02,
        0.04,
        text,
        transform=axis.transAxes,
        fontsize=8.5,
        color=color,
        verticalalignment="bottom",
        bbox={"facecolor": "black", "alpha": 0.68, "pad": 3, "edgecolor": "none"},
    )


def make_figure(
    *,
    source_path: Path,
    ablation_root: Path,
    metrics: dict,
    environment: np.lib.npyio.NpzFile,
    rows: list[dict],
    methods: tuple[str, ...],
    output: Path,
    include_source: bool,
) -> None:
    columns = (("source",) if include_source else ()) + methods
    figure, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(3.45 * len(columns), 2.72 * len(rows)),
        constrained_layout=False,
    )
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            axis = axes[row_index, column_index]
            if column == "source":
                image, pad = pad_source(read_frame(source_path, row["frame"]))
                image = draw_source_annotation(
                    image,
                    row["frame"],
                    row.get("player"),
                    row.get("foot"),
                    environment,
                    pad,
                )
                axis.imshow(image)
                add_cell_label(
                    axis,
                    "yellow: ball; cyan: annotated foot"
                    if row["phase"] != "lane"
                    else "yellow: source ball",
                )
            else:
                image = read_frame(
                    ablation_root / column / "preview.mp4",
                    row["frame"],
                )
                axis.imshow(image)
                if row["phase"] == "lane":
                    sample = metrics["variants"][column]["passes"]["samples"][1]
                    add_cell_label(
                        axis,
                        (
                            f"lane error {100 * sample['crossing_error_m']:.1f} cm\n"
                            f"cone clearance {100 * sample['cone_surface_clearance_m']:.1f} cm"
                        ),
                        sample["cone_surface_clearance_m"] >= 0.0,
                    )
                else:
                    sample = interaction_sample(
                        metrics, column, row["phase"], row["frame"]
                    )
                    add_cell_label(
                        axis,
                        (
                            f"target-foot d {100 * sample['target_foot_distance_m']:.1f} cm\n"
                            f"correct foot {'yes' if sample['correct_foot_association'] else 'no'}"
                        ),
                        sample["correct_foot_association"],
                    )
            if row_index == 0:
                axis.set_title(
                    "Source (local only)"
                    if column == "source"
                    else METHOD_LABELS[column],
                    fontsize=12,
                    fontweight="bold",
                )
            if column_index == 0:
                axis.set_ylabel(row["label"], fontsize=10, fontweight="bold")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#666666")
    if include_source:
        figure.suptitle(
            "LOCAL REVIEW ONLY — source-frame redistribution permission not granted",
            fontsize=13,
            fontweight="bold",
            color="#a00000",
        )
    else:
        figure.suptitle(
            "Touch Ladder qualitative ablation (kinematic/geometric result)",
            fontsize=13,
            fontweight="bold",
        )
    figure.text(
        0.5,
        0.012,
        (
            "Selected interaction frames are shown for qualitative comparison; "
            "per-frame annotations are not aggregate scores. "
            "The event-aware variant includes support constraints."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.02, 0.052, 0.995, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/clips/touch_ladder_demo_source.mp4"),
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/experiments/ablation"),
    )
    parser.add_argument(
        "--paper-metrics",
        type=Path,
        default=Path(
            "data/exports/touch_ladder_demo/paper_eval/"
            "sequences/touch_ladder_demo/metrics.json"
        ),
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/ball_cones.npz"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/paper_eval/figures"),
    )
    args = parser.parse_args()
    guard_release_writes(args)

    metrics = json.loads(args.paper_metrics.read_text())
    environment = np.load(args.environment)
    ball = np.stack(
        [
            environment["ball_xyz_yup"][:, 0],
            -environment["ball_xyz_yup"][:, 2],
            environment["ball_xyz_yup"][:, 1],
        ],
        axis=-1,
    )
    cones = np.stack(
        [
            environment["cone_xyz_yup"][:, 0],
            -environment["cone_xyz_yup"][:, 2],
            environment["cone_xyz_yup"][:, 1],
        ],
        axis=-1,
    )
    pass_two = ball[21:43]
    lane_frame = 21 + int(np.argmin(np.abs(pass_two[:, 0] - np.mean(cones[:, 0]))))
    rows = [
        {
            "frame": 15,
            "phase": "receive",
            "player": "b",
            "foot": "right",
            "label": "B receives\nright foot (f15)",
        },
        {
            "frame": 21,
            "phase": "kick",
            "player": "b",
            "foot": "left",
            "label": "B kicks\nleft foot (f21)",
        },
        {
            "frame": lane_frame,
            "phase": "lane",
            "label": f"Pass crosses\ntarget lane (f{lane_frame})",
        },
        {
            "frame": 42,
            "phase": "receive",
            "player": "a",
            "foot": "right",
            "label": "A receives\nright foot (f42)",
        },
    ]
    publication = args.out_dir / "soccer_ablation_qualitative.png"
    make_figure(
        source_path=args.source,
        ablation_root=args.ablation_root,
        metrics=metrics,
        environment=environment,
        rows=rows,
        methods=METHODS,
        output=publication,
        include_source=False,
    )
    local = args.out_dir / "soccer_ablation_qualitative_local_only.png"
    make_figure(
        source_path=args.source,
        ablation_root=args.ablation_root,
        metrics=metrics,
        environment=environment,
        rows=rows,
        methods=METHODS,
        output=local,
        include_source=True,
    )
    metadata = {
        "publication_safe_figure": str(publication),
        "local_only_source_comparison": str(local),
        "frames": rows,
        "caption": (
            "Selected interaction frames are shown for qualitative comparison; "
            "per-frame annotations are not aggregate scores. The event-aware "
            "variant includes support constraints."
        ),
        "license_note": (
            "The publication-safe figure excludes source frames because current "
            "source redistribution permission is not granted."
        ),
    }
    (args.out_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

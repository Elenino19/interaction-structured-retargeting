#!/usr/bin/env python3
"""Figures for the event-coordination extension.

* interaction_timeline.{png,pdf}: ball-centre distance to all four toes over the
  144 frames, annotated receive/kick frames, sender/receiver role bands and
  predicted closest-approach frames, for the accepted Full and the fixed-ball
  event IK OFF/ON variants.
* qualitative_comparison.{png,pdf}: rendered frames at selected events for the
  fixed-ball OFF/ON pair (same ball, same reference) with per-frame numbers.
  Selected frames are illustrative; they are not aggregate scores.

Runs in the core4d venv (matplotlib + ffmpeg; no cv2 required).
"""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import image as mpimg  # noqa: E402

FOOT_STYLE = {
    "a_left": ("#1f77b4", "-", "A left toe"),
    "a_right": ("#1f77b4", "--", "A right toe"),
    "b_left": ("#d62728", "-", "B left toe"),
    "b_right": ("#d62728", "--", "B right toe"),
}
TIMELINE_VARIANTS = [
    ("full", "Accepted Full (frozen robots, frozen ball)"),
    ("fixed_ball/frozen_reference/event_off", "Fixed ball, shared reference: event IK OFF"),
    ("fixed_ball/frozen_reference/event_on", "Fixed ball, shared reference: event IK ON"),
]
QUAL_EVENTS = [
    ("e01_kick", 21, "f21: B left-foot kick"),
    ("e02_recv", 42, "f42: A right-foot receive"),
    ("e03_kick", 71, "f71: B right-foot touch"),
    ("e05_kick", 110, "f110: B right-foot touch"),
]
QUAL_VARIANTS = [
    ("fixed_ball/frozen_reference/event_off", "Event IK OFF"),
    ("fixed_ball/frozen_reference/event_on", "Event IK ON"),
]


def extract_frame(video: Path, frame: int, out_png: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
            "-vf", f"select=eq(n\\,{frame})", "-vframes", "1", str(out_png),
        ],
        check=True,
    )


def timeline(metrics: dict, out_base: Path) -> None:
    contacts = metrics["contacts"]
    n_frames = 144
    fig, axes = plt.subplots(len(TIMELINE_VARIANTS), 1, figsize=(11.5, 2.9 * len(TIMELINE_VARIANTS)), sharex=True)
    for ax, (label, title) in zip(axes, TIMELINE_VARIANTS):
        curves = metrics["distance_curves"][label]
        events = metrics["variants"][label]["events"]
        frames = np.arange(n_frames)
        # Sender/receiver role bands: between a kick and the next receive the
        # kicker is the sender and the next annotated player the receiver.
        for index in range(len(contacts) - 1):
            k0 = int(contacts[index].get("kick_frame", contacts[index]["frame"]))
            r1 = int(contacts[index + 1]["frame"])
            sender = contacts[index]["player"]
            color = "#1f77b4" if sender == "a" else "#d62728"
            ax.axvspan(k0, r1, color=color, alpha=0.06, lw=0)
            ax.text((k0 + r1) / 2, 0.93, f"{sender.upper()}→{contacts[index + 1]['player'].upper()}",
                    ha="center", va="top", fontsize=7.5, color=color, alpha=0.9)
        for key, (color, style, name) in FOOT_STYLE.items():
            ax.plot(frames, curves[key][:n_frames], color=color, ls=style, lw=1.3, label=name)
        ax.axhline(0.11, color="k", lw=0.8, ls=":", label="ball radius 0.11 m")
        by_frame: dict[int, list[dict]] = {}
        for event in events:
            is_kick = event["phase"] == "kick"
            ax.axvline(event["frame"], color="k", lw=0.9 if is_kick else 0.6, ls="-" if is_kick else "--", alpha=0.55)
            ax.plot(event["predicted_frame"], event["ball_center_to_target_toe_m"],
                    marker="o" if is_kick else "s", ms=5, mfc="none", mec="k", mew=1.0)
            by_frame.setdefault(event["frame"], []).append(event)
        for frame, group in by_frame.items():
            phases = "+".join(e["phase"][:4] for e in group)
            tag = f"{group[0]['player'].upper()}{group[0]['foot'][0].upper()} {phases}"
            ax.text(frame + 0.6, 0.80, tag, rotation=90, fontsize=6.5, va="top", ha="left")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("ball centre to toe [m]")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("frame (24 fps)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.text(
        0.01, -0.035,
        "Solid vertical = annotated kick, dashed vertical = annotated receive; markers = predicted closest-approach frame "
        "(earliest minimum in ±4 frames). Shaded band = annotated sender→receiver transfer. Kinematic proxies on a "
        "prescribed ball path; single 144-frame sequence.",
        fontsize=7.5, ha="left", va="top",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def qualitative(metrics: dict, roots: dict[str, Path], out_base: Path) -> None:
    rows = len(QUAL_VARIANTS)
    cols = len(QUAL_EVENTS)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.75 * rows + 0.5))
    with tempfile.TemporaryDirectory() as tmp:
        for r, (label, name) in enumerate(QUAL_VARIANTS):
            by_id = {e["event_id"]: e for e in metrics["variants"][label]["events"]}
            video = roots[label] / "preview.mp4"
            for c, (event_id, frame, title) in enumerate(QUAL_EVENTS):
                png = Path(tmp) / f"{r}_{c}.png"
                extract_frame(video, frame, png)
                ax = axes[r, c]
                img = mpimg.imread(png)
                h, w = img.shape[:2]
                # crop to the central 70% width to enlarge robots
                x0, x1 = int(0.20 * w), int(0.86 * w)
                ax.imshow(img[int(0.22 * h):int(0.92 * h), x0:x1])
                ax.set_xticks([])
                ax.set_yticks([])
                e = by_id[event_id]
                ax.text(
                    0.02, 0.02,
                    f"toe–centre {e['ball_center_to_target_toe_m']:.3f} m\n"
                    f"surface dev {e['target_toe_surface_deviation_signed_m']:+.3f} m\n"
                    f"mesh gap {e['mesh_target_foot_gap_signed_m']:+.3f} m\n"
                    f"nearest foot: {e['nearest_of_four_feet'].replace('_', ' ')}",
                    transform=ax.transAxes, fontsize=6.6, va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
                )
                if r == 0:
                    ax.set_title(title, fontsize=9)
                if c == 0:
                    ax.set_ylabel(name, fontsize=10)
    fig.suptitle("Fixed frozen ball + shared reference: event-conditioned IK OFF vs ON", fontsize=11, y=0.995)
    fig.text(
        0.01, 0.005,
        "Selected interaction frames for qualitative comparison; per-frame annotations are not aggregate scores. "
        "Same 144-frame ball path, same reference targets, same initialisation and iteration budget. "
        "Negative surface deviation / mesh gap = toe or foot mesh inside the 0.11 m ball (kinematic proxy, no physics).",
        fontsize=7, ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/exports/touch_ladder_demo/paper_eval/event_coordination_extension"))
    parser.add_argument("--contacts", type=Path, default=Path("data/exports/touch_ladder_demo/ball_cones.json"))
    args = parser.parse_args()
    guard_release_writes(args, args.root / 'figure_metadata.json', args.root / 'interaction_timeline.png')
    metrics = json.loads((args.root / "event_chain_metrics.json").read_text())
    metrics["contacts"] = json.loads(args.contacts.read_text())["contacts"]
    roots = {
        label: Path(metrics["variants"][label]["inputs"]["variant_dir"])
        for label in metrics["variants"]
    }
    timeline(metrics, args.root / "interaction_timeline")
    qualitative(metrics, roots, args.root / "qualitative_comparison")
    (args.root / "figure_metadata.json").write_text(
        json.dumps(
            {
                "interaction_timeline": {"variants": TIMELINE_VARIANTS, "frames": 144},
                "qualitative_comparison": {
                    "variants": QUAL_VARIANTS,
                    "events": QUAL_EVENTS,
                    "note": "Selected frames; not aggregate scores. Rendered G1 previews only; no source video frames.",
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote figures under", args.root)


if __name__ == "__main__":
    main()

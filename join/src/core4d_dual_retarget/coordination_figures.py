"""Figures for the cross-agent coordination evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .contact import PHASE_NAMES
from .coordination_eval import SequenceSignals, _runs

PHASE_COLORS = ("#f2f2f2", "#dcdcdc", "#c6dbef", "#fdd0a2", "#c7e9c0", "#f2f2f2")
LANES = (
    ("R1 expected", 5.0),
    ("R1 predicted", 4.0),
    ("R2 expected", 3.0),
    ("R2 predicted", 2.0),
    ("Joint expected", 1.0),
    ("Joint predicted", 0.0),
)


def _save(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".png"), dpi=300)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def _draw_timeline_axis(
    axis: plt.Axes,
    signals: SequenceSignals,
    metrics: dict[str, object],
    *,
    show_phase_legend: bool,
) -> None:
    fps = signals.fps
    frames = len(signals.phases)
    time = np.arange(frames) / fps
    for index, name in enumerate(PHASE_NAMES):
        for start, end in _runs(signals.phases == index):
            axis.axvspan(
                start / fps,
                end / fps,
                color=PHASE_COLORS[index],
                alpha=0.55,
                lw=0,
                label=name if show_phase_legend else None,
            )
    expected_joint = signals.expected[1] & signals.expected[2]
    predicted_joint = signals.predicted[1] & signals.predicted[2]
    series = (
        signals.expected[1],
        signals.predicted[1],
        signals.expected[2],
        signals.predicted[2],
        expected_joint,
        predicted_joint,
    )
    colors = ("tab:blue", "tab:blue", "tab:orange", "tab:orange", "black", "tab:green")
    styles = ("--", "-", "--", "-", "--", "-")
    for (label, y), signal, color, style in zip(LANES, series, colors, styles, strict=True):
        axis.step(
            time,
            y + signal.astype(float) * 0.7,
            where="post",
            color=color,
            linestyle=style,
            linewidth=1.1,
        )
    for start, end in _runs(predicted_joint):
        axis.axvspan(start / fps, end / fps, ymin=0.0, ymax=0.14, color="tab:green", alpha=0.35, lw=0)
    timing = metrics["timing"]
    for key, color, style, name, y_text, align in (
        ("expected_joint_onset_frame", "black", "--", "expected joint onset", 5.95, "right"),
        ("predicted_joint_onset_frame", "tab:green", "-", "predicted joint onset", 1.9, "left"),
    ):
        frame = timing[key]
        if frame is not None:
            axis.axvline(frame / fps, color=color, linestyle=style, linewidth=0.9)
            axis.text(
                frame / fps + (0.05 if align == "left" else -0.05),
                y_text,
                f"{name} f{frame}",
                fontsize=5.5,
                color=color,
                ha=align,
                va="top",
            )
    axis.set_yticks([y + 0.35 for _, y in LANES])
    axis.set_yticklabels([label for label, _ in LANES], fontsize=7)
    axis.set_ylim(-0.2, 6.0)
    axis.set_xlim(0, frames / fps)
    axis.grid(alpha=0.15, axis="x")
    jc = metrics["joint_contact"]
    axis.set_title(
        f"{signals.label}: joint F1 {jc['f1']:.3f}, IoU {jc['iou']:.3f}, "
        f"coverage {jc['recall']:.2f}, onset err {timing['joint_onset_error_frames']} f, "
        f"dropouts {timing['dropout_events']}",
        fontsize=7.5,
        loc="left",
    )


def plot_joint_contact_timeline(
    signals: SequenceSignals,
    metrics: dict[str, object],
    output_path: Path,
) -> None:
    """Single-panel expected/predicted/joint contact timeline with phase bands."""
    figure, axis = plt.subplots(figsize=(7.16, 3.1), layout="constrained")
    _draw_timeline_axis(axis, signals, metrics, show_phase_legend=True)
    axis.set_xlabel("Time [s]")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axis.legend(
        unique.values(),
        unique.keys(),
        fontsize=5.8,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        frameon=False,
    )
    figure.suptitle(
        f"Join contact timeline — {signals.sequence_id} — {signals.label}",
        fontsize=9,
    )
    _save(figure, output_path)


def plot_joint_contact_timeline_variants(
    items: list[tuple[SequenceSignals, dict[str, object]]],
    output_path: Path,
) -> None:
    """Stacked timelines for several variants of the same sequence."""
    figure, axes = plt.subplots(
        len(items), 1, figsize=(7.16, 2.1 * len(items)), sharex=True, layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for index, (axis, (signals, metrics)) in enumerate(zip(axes, items, strict=True)):
        _draw_timeline_axis(axis, signals, metrics, show_phase_legend=index == 0)
    axes[-1].set_xlabel("Time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axes[-1].legend(
        unique.values(),
        unique.keys(),
        fontsize=5.5,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        frameon=False,
    )
    _save(figure, output_path)


def plot_coordination_comparison(
    rows: list[dict[str, object]],
    output_path: Path,
    *,
    valid_key: str = "valid_joint_coverage_tau_0.5cm",
) -> None:
    """Paper-ready overlap / bilateral-error comparison across variants."""
    labels = [str(row["display"]).replace(" ", "\n", 1) for row in rows]
    x = np.arange(len(rows))
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(7.16, 2.5), layout="constrained", gridspec_kw={"width_ratios": (1.5, 1)}
    )
    metrics = (
        ("joint_f1", "Joint-contact F1", "#4c72b0"),
        ("joint_iou", "Joint IoU", "#8da0cb"),
        (valid_key, "Geometry-valid coverage (≤0.5 cm)", "#55a868"),
    )
    width = 0.26
    for offset, (key, name, color) in zip((-width, 0.0, width), metrics, strict=True):
        values = [float(row[key]) if row[key] is not None else 0.0 for row in rows]
        bars = left.bar(x + offset, values, width, label=name, color=color)
        for bar, value in zip(bars, values, strict=True):
            left.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.2,
            )
    left.set_xticks(x)
    left.set_xticklabels(labels, fontsize=6.5)
    left.set_ylim(0, 1.12)
    left.set_ylabel("Fraction of frames")
    left.set_title("Simultaneous-contact overlap (018)", fontsize=8)
    left.legend(fontsize=5.8, loc="upper right", ncol=1)
    left.grid(alpha=0.2, axis="y")

    worse = [float(row["bilateral_max_error_cm"]) for row in rows]
    r1 = [float(row["robot1_anchor_error_cm"]) for row in rows]
    r2 = [float(row["robot2_anchor_error_cm"]) for row in rows]
    right.bar(x, worse, 0.5, color="#c44e52", label="per-frame max over robots")
    right.scatter(x - 0.12, r1, marker="o", s=14, color="tab:blue", label="robot 1", zorder=3)
    right.scatter(x + 0.12, r2, marker="s", s=14, color="tab:orange", label="robot 2", zorder=3)
    right.set_xticks(x)
    right.set_xticklabels(labels, fontsize=6.5)
    right.set_ylabel("Anchor error [cm]")
    right.set_title("Bilateral object-relative error", fontsize=8)
    right.legend(fontsize=5.8, loc="upper right")
    right.grid(alpha=0.2, axis="y")
    _save(figure, output_path)

#!/usr/bin/env python3
"""Publication figures for the frozen Soccer Full demo.

Does not rerun IK. Reads the accepted source clip, the frozen robot preview,
and the frozen event annotations. Writes only under paper_eval/submission_package/.
"""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import image as mpimg  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

FPS = 24.0
N_FRAMES = 144
CLIP_START_S = 203.0
A_COLOR = "#1d4e89"
B_COLOR = "#b03a2e"
KICK_COLOR = "#1a1a1a"
RECV_COLOR = "#2e7d32"
TRANSFER_COLOR = "#6c3483"

EVENTS = [
    {
        "id": "e00_kick",
        "frame": 0,
        "kind": "kick",
        "player": "a",
        "foot": "right",
        "title": "A right-foot kick",
        "subtitle": "Opening send  A→B",
        "role": "sender",
    },
    {
        "id": "e01_recv",
        "frame": 15,
        "kind": "receive",
        "player": "b",
        "foot": "right",
        "title": "B right-foot receive",
        "subtitle": "Transfer arrival",
        "role": "receiver",
    },
    {
        "id": "e01_kick",
        "frame": 21,
        "kind": "kick",
        "player": "b",
        "foot": "left",
        "title": "B left-foot kick",
        "subtitle": "Limb-role switch",
        "role": "sender",
    },
    {
        "id": "pass2_lane",
        "frame": 26,
        "kind": "transfer",
        "player": None,
        "foot": None,
        "title": "Pass in the lane",
        "subtitle": "B→A transfer",
        "role": "ball",
    },
    {
        "id": "e02_recv",
        "frame": 42,
        "kind": "receive",
        "player": "a",
        "foot": "right",
        "title": "A right-foot receive",
        "subtitle": "Actor-role switch",
        "role": "receiver",
    },
    {
        "id": "e02_kick",
        "frame": 45,
        "kind": "kick",
        "player": "a",
        "foot": "right",
        "title": "A right-foot kick",
        "subtitle": "Same-foot redirect",
        "role": "sender",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_frame(video: Path, frame: int, out_png: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame})",
            "-vframes",
            "1",
            str(out_png),
        ],
        check=True,
    )


def crop_array(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = image.shape[:2]
    y0, y1, x0, x1 = box
    return image[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


def annotate_source(
    rgb: np.ndarray,
    frame: int,
    player: str | None,
    foot: str | None,
    environment: np.lib.npyio.NpzFile,
) -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    ball = environment["ball_uv"][frame]
    bx, by = int(ball[0]), int(ball[1])
    draw.ellipse((bx - 14, by - 14, bx + 14, by + 14), outline=(255, 210, 0), width=3)
    if player and foot:
        tracks = environment[f"player_{player}_foot_uv"]
        toe = tracks[frame, 1 if foot == "left" else 3]
        tx, ty = int(toe[0]), int(toe[1])
        draw.ellipse((tx - 14, ty - 14, tx + 14, ty + 14), outline=(0, 220, 255), width=3)
    return np.asarray(image)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_source_robot_figure(
    frames: dict[str, dict[int, np.ndarray]],
    output: Path,
    *,
    include_source: bool,
    watermark: bool,
) -> None:
    n = len(EVENTS)
    rows = 2 if include_source else 1
    fig_w = 2.35 * n
    fig_h = 2.72 * rows + (0.62 if watermark else 0.28)
    fig, axes = plt.subplots(rows, n, figsize=(fig_w, fig_h))
    if rows == 1:
        axes = np.asarray(axes)[None, :]
    source_crop = (0.18, 0.86, 0.02, 0.98)  # drop sky + WeChat subtitle
    robot_crop = (0.22, 0.99, 0.05, 0.95)
    for c, event in enumerate(EVENTS):
        frame = event["frame"]
        if include_source:
            axes[0, c].imshow(crop_array(frames["source"][frame], source_crop))
            axes[0, c].set_xticks([])
            axes[0, c].set_yticks([])
            if c == 0:
                axes[0, c].set_ylabel("Source", fontsize=11, fontweight="bold")
        r = 1 if include_source else 0
        axes[r, c].imshow(crop_array(frames["robot"][frame], robot_crop))
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
        if c == 0:
            axes[r, c].set_ylabel("G1 Full", fontsize=11, fontweight="bold")
        t_s = frame / FPS
        axes[0 if include_source else r, c].set_title(
            f"{event['title']}\n{event['subtitle']}  ·  f{frame} ({t_s:.2f} s)",
            fontsize=7.8,
            pad=5,
        )
        for ax in axes[:, c]:
            for spine in ax.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(0.7)
    top = 0.90 if watermark else 0.93
    if watermark:
        fig.suptitle(
            "LOCAL REVIEW ONLY — source redistribution permission is not granted",
            fontsize=11,
            fontweight="bold",
            color="#9b1c1c",
            y=0.995,
        )
    elif include_source:
        fig.suptitle(
            "Source clip and frozen Soccer Full (same 24 fps index)",
            fontsize=12,
            fontweight="bold",
            y=0.995,
        )
    else:
        fig.suptitle(
            "Frozen Soccer Full — selected interaction events",
            fontsize=12,
            fontweight="bold",
            y=0.995,
        )
    if include_source:
        note = (
            "Same 24 fps index: pipeline frame i = packed-clip frame i = frozen-robot frame i "
            "(original time 203 + i/24 s; no frame-rate conversion). Yellow = detected ball, "
            "cyan = annotated toe. Selected frames illustrate the chain; they are not aggregate "
            "scores. The ball path is synthesized after IK. Source frames require redistribution permission."
        )
    else:
        note = (
            "Frozen Soccer Full preview, 24 fps, 144 frames. Frame i is original time 203 + i/24 s. "
            "Selected interaction frames for qualitative presentation; per-frame views are not aggregate scores. "
            "The object path is prescribed after IK and is not physical foot–ball contact."
        )
    fig.text(0.01, 0.01, note, fontsize=6.6, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0.01, 0.055, 0.995, top))
    save(fig, output)


def make_timeline(contacts: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 3.55))
    ax.set_xlim(-8, 150)
    ax.set_ylim(-0.22, 2.72)
    ax.axhline(0.55, color="#dddddd", lw=10, solid_capstyle="butt", zorder=0)
    ax.axhline(1.70, color="#dddddd", lw=10, solid_capstyle="butt", zorder=0)
    ax.text(-6.5, 1.70, "Actor A", ha="right", va="center", fontsize=11, fontweight="bold", color=A_COLOR)
    ax.text(-6.5, 0.55, "Actor B", ha="right", va="center", fontsize=11, fontweight="bold", color=B_COLOR)

    y = {"a": 1.70, "b": 0.55}
    transfers = []
    markers = []
    for index, contact in enumerate(contacts):
        player = contact["player"]
        recv_f = int(contact["frame"])
        kick_f = int(contact.get("kick_frame", contact["frame"]))
        recv_foot = contact["side"]
        kick_foot = contact.get("kick_side", contact["side"])
        same = index > 0 and recv_f == kick_f
        if same:
            markers.append(("both", player, recv_f, recv_foot, kick_foot))
        else:
            if index > 0:
                markers.append(("receive", player, recv_f, recv_foot, recv_foot))
            markers.append(("kick", player, kick_f, kick_foot, kick_foot))
        if index + 1 < len(contacts):
            nxt = contacts[index + 1]
            transfers.append(
                {
                    "from_player": player,
                    "from_frame": kick_f,
                    "to_player": nxt["player"],
                    "to_frame": int(nxt["frame"]),
                    "label": f"{player.upper()}→{nxt['player'].upper()}",
                }
            )

    for kind, player, frame, recv_foot, kick_foot in markers:
        yy = y[player]
        color = A_COLOR if player == "a" else B_COLOR
        above = player == "a"
        if kind == "kick":
            ax.plot(frame, yy, "o", ms=9.5, color=color, markeredgecolor="white", markeredgewidth=0.8, zorder=4)
            label = f"K {kick_foot[0].upper()}\nf{frame}"
        elif kind == "receive":
            ax.plot(frame, yy, "s", ms=8.2, color="white", markeredgecolor=color, markeredgewidth=1.7, zorder=4)
            label = f"R {recv_foot[0].upper()}\nf{frame}"
        else:
            ax.plot(frame, yy, "s", ms=9.0, color="white", markeredgecolor=color, markeredgewidth=1.6, zorder=4)
            ax.plot(frame, yy, "o", ms=5.8, color=color, zorder=5)
            label = f"RK {kick_foot[0].upper()}\nf{frame}"
        ax.text(
            frame,
            yy + (0.36 if above else -0.40),
            label,
            ha="center",
            va="center",
            fontsize=6.4,
            color=color,
        )

    for item in transfers:
        ax.annotate(
            "",
            xy=(item["to_frame"], y[item["to_player"]]),
            xytext=(item["from_frame"], y[item["from_player"]]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=TRANSFER_COLOR,
                lw=1.15,
                connectionstyle="arc3,rad=0.12",
                mutation_scale=9,
            ),
            zorder=3,
        )
        mid = 0.5 * (item["from_frame"] + item["to_frame"])
        ax.text(mid, 1.12, item["label"], ha="center", va="center", fontsize=7.2, color=TRANSFER_COLOR)

    ax.annotate(
        "limb switch\nR recv → L kick",
        xy=(21, 0.55),
        xytext=(28, -0.12),
        fontsize=6.5,
        color=B_COLOR,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=B_COLOR, lw=0.6),
    )
    ax.annotate(
        "same-foot carry\nR recv → R kick",
        xy=(45, 1.70),
        xytext=(52, 2.28),
        fontsize=6.5,
        color=A_COLOR,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=A_COLOR, lw=0.6),
    )

    ax.set_yticks([])
    sec = np.arange(0, 7)
    ax.set_xticks(sec * FPS)
    ax.set_xticklabels([f"{s:.0f} s  (f{min(int(s * FPS), 143)})" for s in sec])
    ax.set_xlabel("Pipeline time at 24 fps   (frame i = original 203 + i/24 s; no frame-rate conversion)", fontsize=8)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=A_COLOR, ms=8, label="Kick"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="white",
            markeredgecolor=A_COLOR,
            ms=7,
            label="Receive",
        ),
        Line2D([0], [0], color=TRANSFER_COLOR, lw=1.4, label="Annotated transfer"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=7.5, ncol=3, bbox_to_anchor=(1.0, 1.18))
    ax.set_title(
        "Interaction-state timeline  ·  supplied receive/kick labels, not discovered by the solver",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    fig.text(
        0.01,
        0.01,
        "One 144-frame sequence; 6 receive + 7 kick events and 6 kick→receive transfers. "
        "RK = receive and kick on the same annotated frame. Kinematic labels on a prescribed object path.",
        fontsize=7,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.99, 0.92))
    save(fig, output)


def make_results_card(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.35))
    ax.axis("off")
    ax.set_title(
        "Reused quantitative results  ·  one 144-frame sequence, not a population study",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    columns = [
        "Setting",
        "Limb assoc.\n(13 events)",
        "Transfer chain\n(6 pairs)",
        "Centre-gated\nτ = 0.05 m",
        "Kick-foot vel.\nRMSE (m/s)",
        "Root jerk\n(m/s³)",
    ]
    rows = [
        ["Accepted Full", "13/13", "6/6 role", "6/6", "0.48", "784"],
        ["Fixed-ball Event IK OFF", "13/13", "6/6 role", "6/6", "0.66", "437"],
        ["Fixed-ball Event IK ON", "13/13", "6/6 role", "6/6", "0.20", "797"],
        ["Indep. object-aware (w=30)", "13/13", "6/6 role", "6/6", "0.21", "835"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 2.05)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cfcfcf")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#1d4e89")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7.6)
        elif r == 1:
            cell.set_facecolor("#eef3f8")
    fig.text(
        0.01,
        0.04,
        "Event IK ON vs OFF does not change actor/limb/chain assignment on a fixed ball. "
        "It improves kick-foot velocity tracking and increases jerk. "
        "The independent object-aware solver has no cross-robot communication; "
        "Soccer Full is two per-robot IK solves plus post-IK ball synthesis. "
        "Numbers from paper_eval/event_coordination_extension and object_aware_comparison.",
        fontsize=7.2,
        va="bottom",
    )
    save(fig, output)


def make_side_by_side_video(source: Path, robot: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    src_draw = "drawtext=text='Source  24 fps':x=16:y=16:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55"
    rob_draw = "drawtext=text='Frozen Soccer Full  24 fps':x=16:y=16:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55"
    if font.exists():
        src_draw += f":fontfile={font}"
        rob_draw += f":fontfile={font}"
    filter_complex = (
        f"[0:v]scale=960:544,pad=960:720:0:88:black,{src_draw}[s];"
        f"[1:v]scale=960:720,{rob_draw}[r];"
        "[s][r]hstack=inputs=2[out]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-i",
            str(robot),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, default=Path("data/clips/touch_ladder_demo_source.mp4"))
    parser.add_argument(
        "--robot",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4"),
    )
    parser.add_argument("--contacts", type=Path, default=Path("data/exports/touch_ladder_demo/ball_cones.json"))
    parser.add_argument("--environment", type=Path, default=Path("data/exports/touch_ladder_demo/ball_cones.npz"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/paper_eval/submission_package"),
    )
    args = parser.parse_args()
    guard_release_writes(args)
    root = args.project_root.resolve()
    resolve = lambda p: p if p.is_absolute() else root / p  # noqa: E731
    source = resolve(args.source)
    robot = resolve(args.robot)
    contacts_path = resolve(args.contacts)
    env_path = resolve(args.environment)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "local_only").mkdir(exist_ok=True)

    expected = {
        "accepted_preview": "c904d2ed28760466df36261f35f6fd281049faf9618a1f266c51cad3dd46039e",
        "accepted_robot_a": "ef5277f2db390ef8119a3bcc1fe21f801e8b72e4a3843fd3b2fa9a3b9f80d96b",
        "accepted_robot_b": "eb5646a331064c83772351a74cc3dc951618ee69ff00e4130ebe6c212a54568b",
        "accepted_ball": "b608f0a9ea049d33067bdb421a0c24a0f0cdc8ee55cf454444f0de9c56fef47f",
    }
    hashes = {
        "accepted_preview": sha256(robot),
        "accepted_robot_a": sha256(root / "data/exports/touch_ladder_demo/g1/robot_a.npz"),
        "accepted_robot_b": sha256(root / "data/exports/touch_ladder_demo/g1/robot_b.npz"),
        "accepted_ball": sha256(env_path),
        "source_clip": sha256(source),
        "contacts": sha256(contacts_path),
    }
    for key, value in expected.items():
        if hashes[key] != value:
            raise SystemExit(f"Frozen hash mismatch for {key}: {hashes[key]} != {value}")

    contacts = json.loads(contacts_path.read_text())["contacts"]
    environment = np.load(env_path)
    style()

    needed = sorted({event["frame"] for event in EVENTS})
    frames = {"source": {}, "robot": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for frame in needed:
            src_png = tmp_path / f"src_{frame}.png"
            rob_png = tmp_path / f"rob_{frame}.png"
            extract_frame(source, frame, src_png)
            extract_frame(robot, frame, rob_png)
            src = mpimg.imread(src_png)
            if src.dtype != np.uint8:
                src = (np.clip(src, 0, 1) * 255).astype(np.uint8)
            event = next(e for e in EVENTS if e["frame"] == frame)
            frames["source"][frame] = annotate_source(
                src, frame, event["player"], event["foot"], environment
            )
            rob = mpimg.imread(rob_png)
            if rob.dtype != np.uint8:
                rob = (np.clip(rob, 0, 1) * 255).astype(np.uint8)
            frames["robot"][frame] = rob

    make_source_robot_figure(
        frames,
        out / "local_only/fig_source_to_robot.png",
        include_source=True,
        watermark=False,
    )
    make_source_robot_figure(
        frames,
        out / "local_only/fig_source_to_robot_watermarked.png",
        include_source=True,
        watermark=True,
    )
    make_source_robot_figure(
        frames,
        out / "fig_robot_events.png",
        include_source=False,
        watermark=False,
    )
    make_timeline(contacts, out / "fig_interaction_timeline.png")
    make_results_card(out / "fig_results_reuse.png")
    make_side_by_side_video(source, robot, out / "local_only/source_robot_side_by_side.mp4")

    event_rows = []
    for index, contact in enumerate(contacts):
        recv_f = int(contact["frame"])
        kick_f = int(contact.get("kick_frame", contact["frame"]))
        if index > 0:
            event_rows.append(
                {
                    "event_id": f"e{index:02d}_recv",
                    "phase": "receive",
                    "player": contact["player"],
                    "foot": contact["side"],
                    "pipeline_frame": recv_f,
                    "time_s": recv_f / FPS,
                    "original_time_s": CLIP_START_S + recv_f / FPS,
                }
            )
        event_rows.append(
            {
                "event_id": f"e{index:02d}_kick",
                "phase": "kick",
                "player": contact["player"],
                "foot": contact.get("kick_side", contact["side"]),
                "pipeline_frame": kick_f,
                "time_s": kick_f / FPS,
                "original_time_s": CLIP_START_S + kick_f / FPS,
            }
        )
    pairs = []
    for index in range(len(contacts) - 1):
        kick = contacts[index]
        recv = contacts[index + 1]
        pairs.append(
            {
                "pair_id": f"p{index + 1}",
                "sender": kick["player"],
                "sender_foot": kick.get("kick_side", kick["side"]),
                "kick_frame": int(kick.get("kick_frame", kick["frame"])),
                "receiver": recv["player"],
                "receiver_foot": recv["side"],
                "receive_frame": int(recv["frame"]),
                "direction": f"{kick['player'].upper()}→{recv['player'].upper()}",
            }
        )

    metadata = {
        "schema_version": "1.0.0",
        "sequence_id": "touch_ladder_demo",
        "temporal_alignment": {
            "original_fps": 24.0,
            "packed_clip_fps": 24.0,
            "robot_preview_fps": 24.0,
            "n_frames": N_FRAMES,
            "frame_rate_conversion": "none",
            "index_rule": (
                "Pipeline frame i equals packed-clip frame i equals frozen-robot frame i. "
                "Original-video time is 203.0 + i/24 seconds (clip 203–209 s)."
            ),
            "clip_start_s": CLIP_START_S,
            "clip_end_s": 209.0,
        },
        "figure_events": EVENTS,
        "all_annotated_events": event_rows,
        "transfer_pairs": pairs,
        "source_redistribution": False,
        "frozen_hashes": hashes,
        "frozen_hash_check": "passed",
        "ik_rerun": False,
    }
    (out / "event_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()

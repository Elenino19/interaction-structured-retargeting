"""Polished full-sequence demo of one method, with the frozen v41 camera and quality.

960x720, 15 fps, libx264 quality 8 — the settings of ``videos/retarget/<seq>_clearance_v41.mp4`` —
but the camera is taken from the v41 trajectory and held fixed, so the demo is directly comparable
with the v41 video and with the side-by-side renders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio

from core4d_dual_retarget.visual_refinement import SceneRenderer

V41_SMOOTH = Path("outputs/balanced_clearance_v41_smooth")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--method-root", type=Path, required=True, help="smoothed result root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=V41_SMOOTH)
    args = parser.parse_args()

    renderer = SceneRenderer(args.sequence, args.reference_root, width=960, height=720)
    trajectories = renderer.load(args.method_root)
    frames = len(trajectories[1]["joint_positions"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.out, fps=15, codec="libx264", quality=8, macro_block_size=None) as writer:
        for frame in range(frames):
            writer.append_data(renderer.render_frame(trajectories, frame))
    renderer.close()
    print(f"wrote {args.out} ({frames} frames, v41 camera)")


if __name__ == "__main__":
    main()

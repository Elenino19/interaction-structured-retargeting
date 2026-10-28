#!/usr/bin/env python3
"""Run official GVHMR independently on two pre-cropped player videos."""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import subprocess
import sys
from pathlib import Path


def run_one(
    python: Path, gvhmr_root: Path, video: Path, output_root: Path, static: bool
) -> Path:
    command = [
        str(python),
        "tools/demo/demo.py",
        f"--video={video.resolve()}",
        f"--output_root={output_root.resolve()}",
    ]
    if static:
        command.append("-s")
    subprocess.run(command, cwd=gvhmr_root, check=True)
    result = output_root / video.stem / "hmr4d_results.pt"
    if not result.exists():
        matches = list((output_root / video.stem).rglob("hmr4d_results.pt"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Cannot locate GVHMR output for {video}")
        result = matches[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_id")
    parser.add_argument("--player-a", type=Path, required=True)
    parser.add_argument("--player-b", type=Path, required=True)
    parser.add_argument("--gvhmr-root", type=Path, default=Path("third_party/GVHMR"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=Path("data/gvhmr"))
    parser.add_argument(
        "--dynamic-camera",
        action="store_true",
        help="Run visual odometry. Fixed panoramic clips should omit this flag.",
    )
    args = parser.parse_args()
    guard_release_writes(args)

    demo = args.gvhmr_root / "tools/demo/demo.py"
    if not demo.exists():
        raise SystemExit(
            f"Official GVHMR checkout not found at {args.gvhmr_root}. "
            "See README.md prerequisites."
        )
    output_root = args.output / args.clip_id
    output_root.mkdir(parents=True, exist_ok=True)
    results = {
        "player_a": run_one(
            args.python,
            args.gvhmr_root,
            args.player_a,
            output_root / "player_a",
            not args.dynamic_camera,
        ),
        "player_b": run_one(
            args.python,
            args.gvhmr_root,
            args.player_b,
            output_root / "player_b",
            not args.dynamic_camera,
        ),
    }
    for player, path in results.items():
        print(f"{player}: {path}")
    print(
        "Independent GVHMR outputs are not in a shared world frame. "
        "Joint alignment is required before dataset export."
    )


if __name__ == "__main__":
    main()

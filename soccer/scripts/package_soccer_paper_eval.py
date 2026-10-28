#!/usr/bin/env python3
"""Copy reusable ablation videos into the isolated paper-evaluation bundle."""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/experiments/ablation"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/paper_eval/videos"),
    )
    args = parser.parse_args()
    guard_release_writes(args)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {
        "comparison_2x2": args.ablation_root / "comparison_2x2.mp4",
        "basic_ik": args.ablation_root / "basic_ik/preview.mp4",
        "support": args.ablation_root / "support/preview.mp4",
        "event_semantics": args.ablation_root / "event_semantics/preview.mp4",
        "full": args.ablation_root / "full/preview.mp4",
    }
    manifest = {
        "schema_version": "1.0.0",
        "source_video_included": False,
        "source_video_note": (
            "The human source is intentionally excluded because redistribution "
            "permission is not granted."
        ),
        "videos": {},
    }
    for name, source in sources.items():
        destination = args.out / f"{name}.mp4"
        shutil.copy2(source, destination)
        manifest["videos"][name] = {
            "source": str(source),
            "path": str(destination),
            "sha256": sha256(destination),
        }
    (args.out / "video_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

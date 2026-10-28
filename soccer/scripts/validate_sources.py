#!/usr/bin/env python3
"""Fail closed unless at least one source passed visual acceptance."""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/source_audit.json"),
    )
    args = parser.parse_args()
    guard_release_writes(args)

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    accepted = [source for source in data["sources"] if source["status"] == "accepted"]
    if not accepted:
        raise SystemExit(
            "No source passed the visual and licensing gates; dataset generation stopped."
        )
    print("\n".join(source["id"] for source in accepted))


if __name__ == "__main__":
    main()

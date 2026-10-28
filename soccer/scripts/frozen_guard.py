"""Refuse writes that would replace bundled Soccer results."""

from __future__ import annotations

import os
from pathlib import Path

from isr_assets import SOCCER_ROOT

PROTECTED = SOCCER_ROOT / "data" / "exports" / "touch_ladder_demo"


def refuse_write(*paths: Path) -> None:
    if os.environ.get("ISR_ALLOW_FROZEN_OVERWRITE") == "1":
        return
    root = PROTECTED.resolve()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved == root or root in resolved.parents:
            raise SystemExit(
                "Refusing to write inside the bundled Soccer result tree:\n"
                f"  {resolved}\n"
                "Copy the inputs to a new directory and pass that path. "
                "The frozen Full trajectories, ball, and annotations stay read-only."
            )


def guard_release_writes(args, *extra: Path) -> None:
    targets: list[Path] = list(extra)
    for name in ("out", "out_dir", "output"):
        value = getattr(args, name, None)
        if isinstance(value, Path):
            targets.append(value)
    if targets:
        refuse_write(*targets)

#!/usr/bin/env python3
"""Level 1 for Join: report whether a private trajectory bundle is visible.

This repository does not contain Join motions or videos. They show CORE4D
object geometry or are optimization artifacts held outside the public tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REQUIRED = [
    "outputs/balanced_clearance_v41_smooth",
    "outputs/paper_eval/ext_jc_visual_refinement/postprocess/ext_jc_w10",
    "outputs/paper_eval/ext_jc_visual_refinement/postprocess/ext_jc_w10_surrogate",
    "outputs/paper_eval/ext_jc_visual_refinement/postprocess/postprocess_metrics.csv",
]


def main() -> None:
    root = os.environ.get("ISR_JOIN_PRIVATE_ROOT")
    if not root:
        print("BLOCKED ISR_JOIN_PRIVATE_ROOT is unset.")
        print("Join trajectories are not bundled. Set the variable to a local read-only bundle.")
        sys.exit(2)
    base = Path(root)
    missing = [item for item in REQUIRED if not (base / item).exists()]
    if missing:
        print("BLOCKED private bundle is incomplete:")
        for item in missing:
            print(" ", item)
        sys.exit(2)
    print("PASS private Join bundle is readable at", base)
    print("Do not copy it into this repository until the license audit is resolved.")


if __name__ == "__main__":
    main()

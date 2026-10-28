"""Portable locations for code and externally obtained assets.

Scientific constants (gaps, weights, iteration counts, camera parameters)
do not live in this module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SOCCER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SOCCER_ROOT.parent
JOIN_SRC = Path(os.environ["ISR_JOIN_SRC"]) if os.environ.get("ISR_JOIN_SRC") else REPO_ROOT / "join" / "src"


def ensure_join_on_path() -> None:
    entry = str(JOIN_SRC)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def g1_description() -> Path:
    """Directory that contains g1_29dof_rev_1_0.urdf, the MJCF, and meshes."""
    for key in ("UNITREE_G1_ROOT", "ISR_G1_DESCRIPTION"):
        value = os.environ.get(key)
        if value:
            return Path(value)
    return REPO_ROOT / "join" / "third_party" / "unitree_ros" / "robots" / "g1_description"


def g1_urdf() -> Path:
    return g1_description() / "g1_29dof_rev_1_0.urdf"


def g1_xml() -> Path:
    return g1_description() / "g1_29dof_rev_1_0.xml"


def gvhmr_root() -> Path:
    return Path(os.environ.get("GVHMR_ROOT", Path.home() / "GVHMR"))


def yolo_weights() -> Path:
    value = os.environ.get("YOLO_WEIGHTS")
    if value:
        return Path(value)
    return gvhmr_root() / "inputs" / "checkpoints" / "yolo" / "yolov8x.pt"

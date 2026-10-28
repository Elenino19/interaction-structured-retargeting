#!/usr/bin/env python3
"""Level 1: load the bundled Soccer Full trajectories. No optimization, no render."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "soccer" / "data" / "exports" / "touch_ladder_demo"

# Frozen Full only. Prepared GVHMR joints stay outside this tree while
# SMPL-X redistribution is pending. See docs/smplx_joints_provenance.md.
EXPECTED = {
    "g1/robot_a.npz": "ef5277f2db390ef8119a3bcc1fe21f801e8b72e4a3843fd3b2fa9a3b9f80d96b",
    "g1/robot_b.npz": "eb5646a331064c83772351a74cc3dc951618ee69ff00e4130ebe6c212a54568b",
    "ball_cones.npz": "b608f0a9ea049d33067bdb421a0c24a0f0cdc8ee55cf454444f0de9c56fef47f",
    "ball_cones.json": "cad51d649d353ad0918834fb0f03c89aafdbb1fbdb8c42b6d25f60be85bb2156",
    "comparison/current_hop_timed_lateral.mp4": "c904d2ed28760466df36261f35f6fd281049faf9618a1f266c51cad3dd46039e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failed = False
    for relative, expected in EXPECTED.items():
        path = DEMO / relative
        if not path.is_file():
            print(f"MISSING {relative}")
            failed = True
            continue
        digest = sha256(path)
        status = "OK" if digest == expected else "HASH"
        if digest != expected:
            failed = True
        print(f"{status} {relative} {digest}")

    held_joints = DEMO / "smplx_joints.npz"
    if held_joints.exists():
        print("FAIL smplx_joints.npz is inside the release tree while redistribution is pending")
        sys.exit(1)
    print("PENDING smplx_joints.npz (not in this tree; upstream redistribution is unresolved)")

    robot_a = np.load(DEMO / "g1" / "robot_a.npz")
    robot_b = np.load(DEMO / "g1" / "robot_b.npz")
    ball = np.load(DEMO / "ball_cones.npz")
    contacts = json.loads((DEMO / "ball_cones.json").read_text())["contacts"]
    receives = max(len(contacts) - 1, 0)
    kicks = len(contacts)
    transfers = receives
    print(
        "shapes",
        "A", tuple(robot_a["joint_positions"].shape),
        "B", tuple(robot_b["joint_positions"].shape),
        "ball", tuple(ball["ball_xyz_yup"].shape),
        "contact_records", len(contacts),
        "receives", receives,
        "kicks", kicks,
        "transfers", transfers,
    )
    if robot_a["joint_positions"].shape != (144, 29):
        failed = True
    if robot_b["joint_positions"].shape != (144, 29):
        failed = True
    if (len(contacts), receives, kicks, transfers) != (7, 6, 7, 6):
        failed = True
    if failed:
        sys.exit(1)
    print("PASS soccer frozen load")


if __name__ == "__main__":
    main()

"""URDF metadata needed by the G1 retargeting optimizer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JointLimit:
    name: str
    lower: float
    upper: float
    velocity: float | None
    effort: float | None


def _optional_float(element: ET.Element, key: str) -> float | None:
    value = element.get(key)
    return float(value) if value is not None else None


def load_actuated_joint_limits(urdf_path: Path) -> list[JointLimit]:
    """Read ordered revolute/continuous joint limits from a URDF."""
    root = ET.parse(urdf_path).getroot()
    joints: list[JointLimit] = []
    for element in root.findall("joint"):
        joint_type = element.get("type")
        if joint_type not in {"revolute", "continuous"}:
            continue
        name = element.get("name")
        if not name:
            raise ValueError("Actuated joint without a name")
        limit = element.find("limit")
        if limit is None:
            raise ValueError(f"Joint {name!r} has no limit element")
        if joint_type == "continuous":
            lower, upper = -3.141592653589793, 3.141592653589793
        else:
            lower_value, upper_value = limit.get("lower"), limit.get("upper")
            if lower_value is None or upper_value is None:
                raise ValueError(f"Revolute joint {name!r} has incomplete limits")
            lower, upper = float(lower_value), float(upper_value)
        joints.append(
            JointLimit(
                name=name,
                lower=lower,
                upper=upper,
                velocity=_optional_float(limit, "velocity"),
                effort=_optional_float(limit, "effort"),
            )
        )
    return joints


G1_29DOF_KEYPOINT_LINKS = {
    "pelvis": "pelvis",
    "left_hip": "left_hip_pitch_link",
    "right_hip": "right_hip_pitch_link",
    "left_knee": "left_knee_link",
    "right_knee": "right_knee_link",
    "left_ankle": "left_ankle_pitch_link",
    "right_ankle": "right_ankle_pitch_link",
    "neck": "torso_link",
    "left_shoulder": "left_shoulder_pitch_link",
    "right_shoulder": "right_shoulder_pitch_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_wrist": "left_rubber_hand",
    "right_wrist": "right_rubber_hand",
}

SMPLX_KEYPOINT_INDICES = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "neck": 12,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}


def validate_semantic_mapping(available_links: set[str]) -> None:
    """Fail early when a different G1 variant changes required link names."""
    missing = sorted(set(G1_29DOF_KEYPOINT_LINKS.values()) - available_links)
    if missing:
        raise ValueError(f"G1 URDF is missing mapped links: {missing}")

"""CORE4D metadata parsing and target-sequence selection."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActionLabel:
    """Normalized information encoded by a CORE4D action label."""

    sequence_id: str
    raw_label: str
    action: str
    obstacle: str | None


def parse_action_label(sequence_id: str, raw_label: str) -> ActionLabel:
    """Parse labels such as ``join_obs0`` without hiding malformed metadata."""
    parts = raw_label.rsplit("_", maxsplit=1)
    if len(parts) == 2 and parts[1].startswith("obs"):
        action, obstacle = parts
    else:
        action, obstacle = raw_label, None
    return ActionLabel(
        sequence_id=sequence_id,
        raw_label=raw_label,
        action=action,
        obstacle=obstacle,
    )


def load_action_labels(path: Path) -> list[ActionLabel]:
    """Load and normalize the official ``action_labels.json`` mapping."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return [parse_action_label(str(key), str(value)) for key, value in payload.items()]


def select_sequences(
    labels: Iterable[ActionLabel],
    *,
    action: str,
    obstacle: str | None = None,
) -> list[ActionLabel]:
    """Select sequences deterministically by action and optional obstacle label."""
    selected = [
        label
        for label in labels
        if label.action == action and (obstacle is None or label.obstacle == obstacle)
    ]
    return sorted(selected, key=lambda label: label.sequence_id)


def infer_object_category(metadata: object) -> str | None:
    """Best-effort category extraction across known CORE4D metadata variants."""
    if isinstance(metadata, str):
        candidates = [metadata]
    elif isinstance(metadata, dict):
        preferred_keys = (
            "category",
            "object_category",
            "obj_category",
            "object_name",
            "obj_name",
            "name",
            "obj_model_path",
        )
        candidates = [str(metadata[key]) for key in preferred_keys if key in metadata]
        candidates.extend(str(value) for value in metadata.values() if isinstance(value, str))
    else:
        return None

    aliases = {
        "box": ("box",),
        "board": ("board",),
        "barrel": ("barrel", "bucket"),
        "stick": ("stick",),
        "chair": ("chair",),
        "desk": ("desk", "table"),
    }
    lowered = " ".join(candidates).lower()
    for category, tokens in aliases.items():
        if any(token in lowered for token in tokens):
            return category
    return None

#!/usr/bin/env python3
"""Find candidate throw-in/control/return chains in SoccerTrack-v2 BAS labels."""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import json
import os
import urllib.request
from pathlib import Path

MATCH_IDS = (
    "117092",
    "117093",
    "118575",
    "118576",
    "118577",
    "118578",
    "128057",
    "128058",
    "132831",
    "132877",
)
HF_TEMPLATE = (
    "https://huggingface.co/datasets/atomscott/soccertrack-v2/resolve/main/"
    "bas/{match_id}/{match_id}_12_class_events.json"
)
FOLLOW_UP_LABELS = {"Pass", "Drive", "Header", "High Pass"}


def fetch_annotations(match_id: str, token: str) -> dict:
    request = urllib.request.Request(
        HF_TEMPLATE.format(match_id=match_id),
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def position_seconds(event: dict) -> float:
    return int(event["position"]) / 1000.0


def half(event: dict) -> str:
    return str(event["gameTime"]).split("-", maxsplit=1)[0].strip()


def discover(data: dict, lookahead_seconds: float) -> list[dict]:
    events = sorted(
        data.get("annotations", []),
        key=lambda item: (half(item), position_seconds(item)),
    )
    candidates: list[dict] = []
    for index, event in enumerate(events):
        if event.get("label") != "Throw In" or event.get("visibility") == "not shown":
            continue
        start = position_seconds(event)
        later = []
        for next_event in events[index + 1 :]:
            if half(next_event) != half(event):
                break
            delta = position_seconds(next_event) - start
            if delta > lookahead_seconds:
                break
            if (
                next_event.get("team") == event.get("team")
                and next_event.get("label") in FOLLOW_UP_LABELS
                and next_event.get("visibility") != "not shown"
            ):
                later.append(
                    {
                        "label": next_event["label"],
                        "time_seconds": position_seconds(next_event),
                        "delta_seconds": round(delta, 3),
                        "player_id": next_event.get("player_id"),
                    }
                )
        candidates.append(
            {
                "half": int(half(event)),
                "throw_time_seconds": start,
                "throw_player_id": event.get("player_id"),
                "team": event.get("team"),
                "follow_up_events": later,
                "suggested_clip_start": max(0.0, start - 3.0),
                "suggested_clip_end": start + lookahead_seconds + 2.0,
                "requires_visual_review": True,
            }
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/manifests"))
    parser.add_argument("--match", action="append", choices=MATCH_IDS)
    parser.add_argument("--lookahead", type=float, default=12.0)
    args = parser.parse_args()
    guard_release_writes(args)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is required because SoccerTrack-v2 is gated at download time."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    for match_id in args.match or MATCH_IDS:
        data = fetch_annotations(match_id, token)
        candidates = discover(data, args.lookahead)
        destination = args.output / f"{match_id}_throw_in_candidates.json"
        destination.write_text(
            json.dumps(
                {
                    "match_id": match_id,
                    "lookahead_seconds": args.lookahead,
                    "candidates": candidates,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"{match_id}: {len(candidates)} throw-ins -> {destination}")


if __name__ == "__main__":
    main()

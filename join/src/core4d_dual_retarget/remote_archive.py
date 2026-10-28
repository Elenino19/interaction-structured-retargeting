"""Selective access to large CORE4D ZIP archives over HTTP range requests."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from remotezip import RemoteZip

V1_MOTIONS_URL = (
    "https://huggingface.co/datasets/leolyliu/CORE4D/resolve/main/"
    "CORE4D_Real/human_object_motions.zip"
)

V2_BATCH2_URL = (
    "https://huggingface.co/datasets/leolyliu/CORE4D/resolve/main/"
    "CORE4D_Real_human_object_motions_v2/batch2.zip"
)

V1_SEQUENCE_FILENAMES = (
    "smooth_objposes.npy",
    "object_metadata.json",
    "aligned_frame_ids.txt",
)

V2_SEQUENCE_MEMBERS = {
    "person_1/result.npz": "person1_poses.npz",
    "person_2/result.npz": "person2_poses.npz",
}


def _sequence_suffix(sequence_id: str, filename: str) -> str:
    return f"/{sequence_id.strip('/')}/{filename}"


def locate_member(names: Iterable[str], sequence_id: str, filename: str) -> str | None:
    """Find a sequence file regardless of the archive's outer directory."""
    suffix = _sequence_suffix(sequence_id, filename)
    exact = f"{sequence_id.strip('/')}/{filename}"
    return next((name for name in names if name == exact or name.endswith(suffix)), None)


def _copy_member_with_retries(
    archive: RemoteZip,
    member: str,
    target: Path,
    *,
    attempts: int = 4,
) -> None:
    expected_size = archive.getinfo(member).file_size
    if target.exists() and target.stat().st_size == expected_size:
        return
    temporary = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            temporary.unlink(missing_ok=True)
            with archive.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=256 * 1024)
            actual_size = temporary.stat().st_size
            if actual_size != expected_size:
                raise OSError(
                    f"Incomplete member {member}: {actual_size} != {expected_size} bytes"
                )
            temporary.replace(target)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(2**attempt)


def read_sequence_metadata(
    archive_url: str,
    sequence_ids: Iterable[str],
) -> dict[str, dict[str, object]]:
    """Read only object metadata for selected sequences from a remote ZIP."""
    results: dict[str, dict[str, object]] = {}
    with RemoteZip(archive_url) as archive:
        names = archive.namelist()
        for sequence_id in sequence_ids:
            member = locate_member(names, sequence_id, "object_metadata.json")
            if member is None:
                continue
            with archive.open(member) as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                results[sequence_id] = payload
            else:
                results[sequence_id] = {"raw": payload}
    return results


def extract_sequence(
    archive_url: str,
    sequence_id: str,
    destination: Path,
    filenames: Iterable[str] = V1_SEQUENCE_FILENAMES,
    rename: dict[str, str] | None = None,
) -> list[Path]:
    """Stream selected sequence files without downloading the complete archive."""
    written: list[Path] = []
    sequence_parts = PurePosixPath(sequence_id).parts
    if len(sequence_parts) != 2 or any(part in {"", ".", ".."} for part in sequence_parts):
        raise ValueError(f"Invalid CORE4D sequence ID: {sequence_id!r}")

    with RemoteZip(archive_url) as archive:
        names = archive.namelist()
        for filename in filenames:
            member = locate_member(names, sequence_id, filename)
            if member is None:
                continue
            output_name = (rename or {}).get(filename, filename)
            target = destination.joinpath(*sequence_parts, output_name)
            _copy_member_with_retries(archive, member, target)
            written.append(target)
    return written


def extract_sequences(
    archive_url: str,
    sequence_ids: Iterable[str],
    destination: Path,
    filenames: Iterable[str],
    rename: dict[str, str] | None = None,
    *,
    skip_existing: bool = True,
) -> dict[str, list[Path]]:
    """Extract several sequences while reusing one remote archive connection."""
    results: dict[str, list[Path]] = {}
    requested_filenames = tuple(filenames)
    with RemoteZip(archive_url) as archive:
        names = archive.namelist()
        for sequence_id in sequence_ids:
            sequence_parts = PurePosixPath(sequence_id).parts
            if len(sequence_parts) != 2 or any(
                part in {"", ".", ".."} for part in sequence_parts
            ):
                raise ValueError(f"Invalid CORE4D sequence ID: {sequence_id!r}")
            written: list[Path] = []
            for filename in requested_filenames:
                member = locate_member(names, sequence_id, filename)
                if member is None:
                    continue
                output_name = (rename or {}).get(filename, filename)
                target = destination.joinpath(*sequence_parts, output_name)
                expected_size = archive.getinfo(member).file_size
                if skip_existing and target.exists() and target.stat().st_size == expected_size:
                    written.append(target)
                    continue
                _copy_member_with_retries(archive, member, target)
                written.append(target)
            results[sequence_id] = written
    return results

"""Dataset manifests.

The IMDA National Speech Corpus is licensed and is not redistributed by this
repository. Evaluation is driven by a manifest of local paths, so the code is
public while the audio and transcripts stay wherever you are licensed to hold
them.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("audio_id", "audio_path", "transcript")


@dataclass(frozen=True)
class ManifestEntry:
    """One utterance: an identifier, an audio file, and its reference text."""

    audio_id: str
    audio_path: Path
    transcript: str


def load_manifest(path: Path, *, require_audio: bool = False) -> list[ManifestEntry]:
    """Read a manifest CSV, rejecting malformed or ambiguous rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        entries = list(_rows(reader, path.parent, require_audio=require_audio))

    if not entries:
        raise ValueError("manifest contains no rows")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for entry in entries:
        if entry.audio_id in seen:
            duplicates.add(entry.audio_id)
        seen.add(entry.audio_id)
    if duplicates:
        raise ValueError(
            f"manifest has duplicate audio_id values: {sorted(duplicates)[:5]}"
        )
    return entries


def _rows(
    reader: csv.DictReader[str], root: Path, *, require_audio: bool
) -> Iterator[ManifestEntry]:
    for number, row in enumerate(reader, start=2):
        audio_id = (row.get("audio_id") or "").strip()
        transcript = (row.get("transcript") or "").strip()
        raw_path = (row.get("audio_path") or "").strip()
        if not audio_id:
            raise ValueError(f"row {number}: audio_id must not be empty")
        if not transcript:
            raise ValueError(f"row {number}: transcript must not be empty")
        if not raw_path:
            raise ValueError(f"row {number}: audio_path must not be empty")

        audio_path = Path(raw_path)
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        if require_audio and not audio_path.is_file():
            raise ValueError(f"row {number}: audio file not found: {audio_path}")
        yield ManifestEntry(audio_id, audio_path, transcript)

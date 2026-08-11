from __future__ import annotations

from pathlib import Path

import pytest

from whisper_singapore_english.manifest import load_manifest


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text(body, encoding="utf-8")
    return path


def test_valid_manifest_resolves_paths_relative_to_the_manifest(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path,
        "audio_id,audio_path,transcript\n1,audio/a.wav,hello there\n",
    )
    entries = load_manifest(manifest)
    assert len(entries) == 1
    assert entries[0].audio_path == tmp_path / "audio" / "a.wav"
    assert entries[0].transcript == "hello there"


def test_missing_columns_are_rejected(tmp_path: Path) -> None:
    manifest = _write(tmp_path, "audio_id,transcript\n1,hello\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(manifest)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path,
        "audio_id,audio_path,transcript\n1,a.wav,hello\n1,b.wav,there\n",
    )
    with pytest.raises(ValueError, match="duplicate audio_id"):
        load_manifest(manifest)


def test_empty_fields_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="transcript must not be empty"):
        load_manifest(_write(tmp_path, "audio_id,audio_path,transcript\n1,a.wav,\n"))
    with pytest.raises(ValueError, match="audio_id must not be empty"):
        load_manifest(_write(tmp_path, "audio_id,audio_path,transcript\n,a.wav,hi\n"))


def test_empty_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        load_manifest(_write(tmp_path, "audio_id,audio_path,transcript\n"))


def test_missing_audio_is_only_checked_when_requested(tmp_path: Path) -> None:
    manifest = _write(tmp_path, "audio_id,audio_path,transcript\n1,nope.wav,hello\n")
    load_manifest(manifest)
    with pytest.raises(ValueError, match="audio file not found"):
        load_manifest(manifest, require_audio=True)

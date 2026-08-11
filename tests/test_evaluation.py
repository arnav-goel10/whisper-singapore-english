from __future__ import annotations

import json
from pathlib import Path

import pytest

from whisper_singapore_english.evaluation import evaluate, write_report
from whisper_singapore_english.manifest import ManifestEntry


class FakeTranscriber:
    """Returns canned hypotheses so scoring is tested without a model."""

    def __init__(self, name: str, outputs: dict[str, str]) -> None:
        self.name = name
        self._outputs = outputs
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path) -> str:
        self.calls.append(audio_path)
        return self._outputs[audio_path.name]


def _entries() -> list[ManifestEntry]:
    return [
        ManifestEntry("1", Path("a.wav"), "the quick brown fox"),
        ManifestEntry("2", Path("b.wav"), "jumps over the lazy dog"),
    ]


def test_evaluate_scores_every_entry_once() -> None:
    transcriber = FakeTranscriber(
        "fake", {"a.wav": "the quick brown fox", "b.wav": "jumps over the lazy dog"}
    )
    result = evaluate(_entries(), transcriber, split="test")
    assert [p.name for p in transcriber.calls] == ["a.wav", "b.wav"]
    assert result.breakdown.corpus_wer == 0.0
    assert result.breakdown.utterances == 2
    assert result.system == "fake"
    assert result.split == "test"


def test_evaluate_aggregates_errors_across_utterances() -> None:
    transcriber = FakeTranscriber(
        "fake", {"a.wav": "the quick green fox", "b.wav": "jumps over the lazy dog"}
    )
    result = evaluate(_entries(), transcriber, split="test")
    # 1 substitution over 9 reference words.
    assert result.breakdown.reference_words == 9
    assert result.breakdown.substitutions == 1
    assert result.breakdown.corpus_wer == pytest.approx(1 / 9)


def test_empty_manifest_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty manifest"):
        evaluate([], FakeTranscriber("fake", {}), split="test")


def test_report_is_deterministic_json(tmp_path: Path) -> None:
    transcriber = FakeTranscriber(
        "fake", {"a.wav": "the quick brown fox", "b.wav": "jumps over the lazy dog"}
    )
    result = evaluate(_entries(), transcriber, split="test")
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    write_report([result], first)
    write_report([result], second)
    assert first.read_text() == second.read_text()
    payload = json.loads(first.read_text())
    assert payload[0]["corpus_wer"] == 0.0
    assert payload[0]["system"] == "fake"

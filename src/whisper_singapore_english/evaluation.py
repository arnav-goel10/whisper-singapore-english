"""Evaluation over a manifest.

The model is reached through the :class:`Transcriber` protocol, so the scoring
path is exercised by tests without torch installed, and the zero-shot baseline
and the LoRA adapter run through exactly the same code.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from whisper_singapore_english.manifest import ManifestEntry
from whisper_singapore_english.wer import WerBreakdown, corpus_wer


class Transcriber(Protocol):
    """Anything that turns an audio file into text."""

    @property
    def name(self) -> str:
        """A short identifier for the system being scored."""
        ...

    def transcribe(self, audio_path: Path) -> str:
        """Return the hypothesis transcript for ``audio_path``."""
        ...


@dataclass(frozen=True)
class EvaluationResult:
    """Scores for one system over one split."""

    system: str
    split: str
    breakdown: WerBreakdown

    def to_dict(self) -> dict[str, object]:
        return {
            "system": self.system,
            "split": self.split,
            "utterances": self.breakdown.utterances,
            "reference_words": self.breakdown.reference_words,
            "substitutions": self.breakdown.substitutions,
            "deletions": self.breakdown.deletions,
            "insertions": self.breakdown.insertions,
            "corpus_wer": round(self.breakdown.corpus_wer, 6),
            "mean_utterance_wer": round(self.breakdown.mean_utterance_wer, 6),
        }


def evaluate(
    entries: Sequence[ManifestEntry], transcriber: Transcriber, *, split: str
) -> EvaluationResult:
    """Score ``transcriber`` over ``entries``."""
    if not entries:
        raise ValueError("cannot evaluate an empty manifest")
    hypotheses = [transcriber.transcribe(entry.audio_path) for entry in entries]
    references = [entry.transcript for entry in entries]
    return EvaluationResult(
        system=transcriber.name,
        split=split,
        breakdown=corpus_wer(references, hypotheses),
    )


def write_report(results: Sequence[EvaluationResult], path: Path) -> None:
    """Write results as sorted JSON so reruns produce reviewable diffs."""
    if not results:
        raise ValueError("cannot write an empty report")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.to_dict() for result in results]
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

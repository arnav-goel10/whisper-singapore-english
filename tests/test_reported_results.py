"""Guard the figures quoted in the README against the checked-in evidence.

Every number in the README is reproduced here. If a metric file changes, or
someone edits a headline claim, this fails.
"""

from __future__ import annotations

import csv
import json
from itertools import pairwise
from pathlib import Path

import pytest

from whisper_singapore_english.manifest import load_manifest

RESULTS = Path(__file__).parents[1] / "results"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_validation_wer_improves_monotonically() -> None:
    rows = _rows("validation_metrics.csv")
    wers = [float(r["eval_wer"]) for r in rows]
    steps = [int(r["step"]) for r in rows]
    assert steps == sorted(steps)
    assert all(later < earlier for earlier, later in pairwise(wers))


def test_reported_validation_endpoints() -> None:
    rows = _rows("validation_metrics.csv")
    first, last = rows[0], rows[-1]
    assert int(first["step"]) == 500
    assert float(first["eval_wer"]) == pytest.approx(0.331, abs=5e-4)
    assert int(last["step"]) == 3000
    assert float(last["eval_wer"]) == pytest.approx(0.238, abs=5e-4)


def test_training_loss_falls_over_the_run() -> None:
    rows = _rows("training_metrics.csv")
    losses = [float(r["train_loss"]) for r in rows]
    assert losses[0] == pytest.approx(2.7069)
    assert losses[-1] == pytest.approx(0.4144)
    assert losses[-1] < losses[0] / 5


def test_zero_shot_baselines_match_the_readme() -> None:
    payload = json.loads((RESULTS / "zero_shot_baselines.json").read_text())
    by_system = {entry["system"]: entry for entry in payload}
    tiny = by_system["openai/whisper-tiny"]
    small = by_system["openai/whisper-small"]

    assert tiny["corpus_wer"] == pytest.approx(0.662, abs=5e-4)
    assert small["corpus_wer"] == pytest.approx(0.4584, abs=5e-4)
    # whisper-small is the larger model and must be the stronger baseline.
    assert small["corpus_wer"] < tiny["corpus_wer"]
    # Corpus WER weights by reference length and is lower than the utterance mean.
    for entry in payload:
        assert entry["corpus_wer"] < entry["mean_utterance_wer"]
        assert entry["condition"] == "zero-shot"
        assert entry["split"] == "test"


def test_fine_tuned_tiny_beats_zero_shot_small() -> None:
    payload = json.loads((RESULTS / "zero_shot_baselines.json").read_text())
    small = next(e for e in payload if e["system"] == "openai/whisper-small")
    final = float(_rows("validation_metrics.csv")[-1]["eval_wer"])
    # The headline claim. Splits differ, which the README states explicitly.
    assert final < small["corpus_wer"]


def test_sample_manifest_is_loadable_and_synthetic() -> None:
    root = Path(__file__).parents[1] / "data" / "sample" / "manifest.csv"
    entries = load_manifest(root)
    assert len(entries) == 3
    assert all(entry.audio_id.startswith("demo-") for entry in entries)

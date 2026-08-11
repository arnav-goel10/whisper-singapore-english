from __future__ import annotations

import pytest

from whisper_singapore_english.wer import corpus_wer, utterance_wer


def test_identical_transcripts_score_zero() -> None:
    assert utterance_wer("the quick brown fox", "the quick brown fox") == 0.0


def test_single_substitution() -> None:
    assert utterance_wer("the quick brown fox", "the quick green fox") == pytest.approx(
        0.25
    )


def test_single_deletion_and_insertion() -> None:
    assert utterance_wer("a b c d", "a b d") == pytest.approx(0.25)
    assert utterance_wer("a b c d", "a b c x d") == pytest.approx(0.25)


def test_wer_can_exceed_one() -> None:
    assert utterance_wer("hello", "hello there my friend") == pytest.approx(3.0)


def test_empty_hypothesis_scores_one() -> None:
    assert utterance_wer("one two three", "") == pytest.approx(1.0)


def test_error_counts_are_categorised() -> None:
    breakdown = corpus_wer(["a b c d"], ["a x c d e"])
    assert (breakdown.substitutions, breakdown.insertions, breakdown.deletions) == (
        1,
        1,
        0,
    )
    assert breakdown.reference_words == 4
    assert breakdown.corpus_wer == pytest.approx(0.5)


def test_corpus_wer_weights_by_reference_length_not_by_utterance() -> None:
    # One long correct utterance and one short wrong one. Corpus WER must be
    # dominated by the long utterance; the mean of per-utterance rates is not.
    references = ["one two three four five six seven eight nine ten", "yes"]
    hypotheses = ["one two three four five six seven eight nine ten", "no"]
    breakdown = corpus_wer(references, hypotheses)
    assert breakdown.corpus_wer == pytest.approx(1 / 11)
    assert breakdown.mean_utterance_wer == pytest.approx(0.5)
    assert breakdown.corpus_wer < breakdown.mean_utterance_wer


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="count mismatch"):
        corpus_wer(["a"], ["a", "b"])


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        corpus_wer([], [])


def test_empty_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one word"):
        utterance_wer("", "something")

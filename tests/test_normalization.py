from __future__ import annotations

from whisper_singapore_english.normalization import normalize, tokenize
from whisper_singapore_english.wer import utterance_wer


def test_case_and_punctuation_are_folded() -> None:
    assert normalize("The Quick, Brown Fox!") == "the quick brown fox"


def test_imda_annotation_markers_are_stripped() -> None:
    # Real IMDA transcripts mark uncertain segments with '**'.
    assert normalize("** Jingisukan Breadsticks") == "jingisukan breadsticks"


def test_curly_apostrophes_match_straight_ones() -> None:
    assert normalize("don’t") == normalize("don't") == "don't"  # noqa: RUF001


def test_whitespace_is_collapsed() -> None:
    assert normalize("  a   b \n c ") == "a b c"


def test_normalization_makes_punctuation_differences_free() -> None:
    # A hypothesis differing only in punctuation and case must score zero.
    assert utterance_wer("Davian Donita and Marion", "davian donita and marion.") == 0.0


def test_tokenize_of_empty_text_is_empty() -> None:
    assert tokenize("   ***  ") == []

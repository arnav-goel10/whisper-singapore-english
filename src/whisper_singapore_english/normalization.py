"""Text normalization applied before word error rate is computed.

WER is only comparable when both sides are normalized identically, so this
module is the single definition used by evaluation, and it is applied to
reference and hypothesis alike.
"""

from __future__ import annotations

import re
import unicodedata

# The IMDA transcripts mark uncertain or non-lexical segments with a leading
# ``**``. Those markers are annotation metadata, not spoken words, so they are
# removed rather than scored as tokens.
_ANNOTATION_MARKER = re.compile(r"\*+")
# Ambiguous glyphs are intentional: these are the apostrophes that appear in
# real transcripts and must all fold to a straight quote.
_APOSTROPHES = dict.fromkeys(map(ord, "‘’ʼ´`"), "'")  # noqa: RUF001
_KEEP = re.compile(r"[^a-z0-9' ]+")
_COLLAPSE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Return ``text`` folded to the canonical form used for scoring."""
    folded = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES)
    folded = _ANNOTATION_MARKER.sub(" ", folded.lower())
    folded = _KEEP.sub(" ", folded)
    return _COLLAPSE.sub(" ", folded).strip()


def tokenize(text: str) -> list[str]:
    """Return the normalized word tokens of ``text``."""
    normalized = normalize(text)
    return normalized.split() if normalized else []

"""Word error rate.

Two figures are reported because they answer different questions and are
routinely confused:

``corpus_wer``
    Total edit distance divided by total reference length. This is the standard
    ASR reporting metric and weights every spoken word equally.

``mean_utterance_wer``
    The average of per-utterance rates. Short utterances dominate it, so it is
    typically higher. It is reported only for comparison and should never be
    presented as "the" WER.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from whisper_singapore_english.normalization import tokenize


@dataclass(frozen=True)
class WerBreakdown:
    """Aggregate error counts and the rates derived from them."""

    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    utterances: int
    mean_utterance_wer: float

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def corpus_wer(self) -> float:
        if self.reference_words == 0:
            raise ValueError("cannot compute WER against zero reference words")
        return self.errors / self.reference_words


def _alignment_counts(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) via Levenshtein backtrace."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(cost[i - 1][j - 1], cost[i - 1][j], cost[i][j - 1])

    subs = dels = ins = 0
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and cost[i][j] == cost[i - 1][j - 1]
        ):
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return subs, dels, ins


def utterance_wer(reference: str, hypothesis: str) -> float:
    """Return the WER of a single utterance."""
    ref = tokenize(reference)
    if not ref:
        raise ValueError("reference must contain at least one word")
    subs, dels, ins = _alignment_counts(ref, tokenize(hypothesis))
    return (subs + dels + ins) / len(ref)


def corpus_wer(references: Sequence[str], hypotheses: Sequence[str]) -> WerBreakdown:
    """Return the aggregate error breakdown over paired references/hypotheses."""
    if len(references) != len(hypotheses):
        raise ValueError(
            "reference/hypothesis count mismatch: "
            f"{len(references)} != {len(hypotheses)}"
        )
    if not references:
        raise ValueError("cannot compute WER over an empty corpus")

    subs = dels = ins = words = 0
    rates: list[float] = []
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref = tokenize(reference)
        if not ref:
            raise ValueError("every reference must contain at least one word")
        s, d, i = _alignment_counts(ref, tokenize(hypothesis))
        subs, dels, ins, words = subs + s, dels + d, ins + i, words + len(ref)
        rates.append((s + d + i) / len(ref))

    return WerBreakdown(
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        reference_words=words,
        utterances=len(references),
        mean_utterance_wer=sum(rates) / len(rates),
    )

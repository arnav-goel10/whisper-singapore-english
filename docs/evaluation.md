# Evaluation

## Metrics

Given a reference and a hypothesis, word-level Levenshtein alignment yields
substitutions `S`, deletions `D`, and insertions `I`, against `N` reference
words.

```text
corpus WER          = sum(S + D + I) / sum(N)      over the whole split
mean utterance WER  = mean( (S + D + I) / N )      averaged over utterances
```

Corpus WER is the standard ASR figure. Mean per-utterance WER is reported beside
it because the two diverge substantially and quoting whichever is lower is an
easy way to mislead. On the zero-shot whisper-tiny baseline the split is 66.2%
against 77.4%.

WER is unbounded above: a hypothesis longer than the reference can exceed 1.0
through insertions. Rates are not clipped.

## Normalization

Both sides pass through the same function before scoring. It folds Unicode to
NFKC, maps curly apostrophes and modifier letters to a straight quote, lowercases,
strips remaining punctuation, and collapses whitespace. IMDA's `**` uncertainty
markers are removed.

Normalization choices move WER, so they belong in version control rather than in
a notebook cell. Stripping `**` alone changes the reference token count.

## What the published numbers support

Measured, and stated in the README:

- whisper-tiny zero-shot scores 66.2% corpus WER on the test split.
- whisper-small zero-shot scores 45.8% on the same split.
- whisper-tiny with a 540,672-parameter LoRA adapter reaches 23.8% validation
  WER at step 3,000, down from 33.1% at step 500.

Not supported:

- **A controlled before/after on one split.** The baselines are test-split; the
  fine-tuned figures are validation-split. `whisper-sg evaluate` closes this by
  scoring the adapter on the test manifest.
- **A converged result.** Validation WER was still falling at step 3,000.
- **Generalisation beyond this corpus.** Every figure describes IMDA National
  Speech Corpus data. Singaporean-accented English elsewhere is untested.
- **Any claim about whisper-small fine-tuned.** Only its zero-shot baseline was
  measured.

## Reproducing

Zero-shot baseline, then the adapted model, over the same manifest:

```bash
whisper-sg evaluate --manifest /path/to/test_manifest.csv \
  --model-id openai/whisper-tiny --split test --output results/baseline.json

whisper-sg evaluate --manifest /path/to/test_manifest.csv \
  --model-id openai/whisper-tiny --adapter /path/to/adapter \
  --split test --output results/finetuned.json
```

Both write sorted JSON with error counts and reference word totals, so results
diff cleanly and a WER can be recomputed from its components.

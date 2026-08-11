# Architecture

```text
manifest.csv                 licensed audio stays local
      |
      v
load_manifest                strict rows, resolved paths, unique ids
      |
      v
Transcriber (Protocol)       whisper-tiny | whisper-tiny + LoRA | test stub
      |                      torch imported lazily, only in asr.py
      v
normalize / tokenize         one definition, applied to both sides
      |
      v
corpus_wer                   single alignment -> S, D, I, both rates
      |
      v
EvaluationResult -> JSON     sorted keys, reviewable diffs
```

## Why the model sits behind a protocol

`Transcriber` is a structural `Protocol` with one method. Three things follow.

The scoring path is testable without torch, so the suite runs in under a second
and CI needs no ML stack. The zero-shot baseline and the adapted model run
through *identical* evaluation code, which removes a common source of
accidentally-flattering comparisons. And a stub transcriber can return canned
hypotheses, so alignment behaviour is asserted against hand-computed answers
rather than model output.

`torch`, `transformers`, `peft`, and `librosa` are imported inside
`WhisperTranscriber.load()` and are declared as the optional `asr` extra. A
missing extra raises a message naming the install command.

## Modules

### `normalization.py`

The single definition of what counts as a word. Unicode is NFKC-folded, curly
apostrophes and modifier letters fold to a straight quote, case is lowered,
remaining punctuation is dropped, and whitespace is collapsed.

IMDA transcripts mark uncertain or non-lexical segments with a leading `**`.
Those markers are annotation metadata rather than speech, so they are stripped
instead of being scored as tokens. Leaving them in would inflate the reference
length and quietly depress every WER figure.

### `wer.py`

One Levenshtein pass produces the alignment; the backtrace classifies each edit
as a substitution, deletion, or insertion. `WerBreakdown` exposes those counts
alongside both rates.

Corpus WER is total edits over total reference words. Mean per-utterance WER is
the average of individual rates. They are different numbers and the gap is not
small: on the zero-shot whisper-tiny baseline they are 66.2% and 77.4%. Short
utterances carry the same weight as long ones in the mean, and a three-word
utterance with one error scores 0.33 while a thirty-word utterance with one
error scores 0.03. Corpus WER is the standard reporting metric, so it is what
the README leads with, and the mean is published beside it rather than quietly
chosen when it flatters.

### `manifest.py`

Evaluation is driven by a CSV of identifiers, paths, and reference text. This is
what keeps the repository publishable: the corpus is referenced, never vendored.

Rows are validated before any model loads, because discovering a malformed
manifest after a model has been pulled and half a split transcribed is an
expensive way to find a typo.

### `lora.py`

`LoraSpec.trainable_parameters` computes adapter size from architecture rather
than reporting a number from a training log. For each adapted module the count
is `r * (in_features + out_features)`. Attention projections are square at
`d_model`; the feed-forward pair maps `d_model` to `ffn_dim` and back.

Encoder layers carry self-attention only. Decoder layers additionally carry
cross-attention, so their attention projections are counted twice. For
whisper-tiny with the reported configuration this yields 540,672, matching the
released checkpoint exactly, which is asserted as a known-answer test.

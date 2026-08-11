# Data Card

## Source corpus

Training and evaluation used the **IMDA National Speech Corpus (NSC)**, a
Singaporean English speech corpus released by the Infocomm Media Development
Authority of Singapore.

The corpus is **not redistributed by this repository**. No audio, no reference
transcripts, and no corpus-derived text are checked in. Obtain the corpus from
IMDA under its own terms.

## Split sizes used

| Split | Utterances |
| --- | --- |
| Train | 22,376 |
| Validation | 4,796 |
| Test | 4,796 |
| Total | 31,966 |

## What is checked in

- `data/sample/manifest.csv`: three synthetic rows written for this repository.
  They exercise the manifest loader and contain no corpus content.
- `results/training_metrics.csv`: step and training loss from the reported run.
- `results/validation_metrics.csv`: step, evaluation loss, and evaluation WER.
- `results/zero_shot_baselines.json`: aggregate corpus and mean per-utterance
  WER for the two zero-shot baselines.

These are scalar training logs and aggregate scores. They contain no
transcripts, no audio, and no per-utterance corpus text.

## What is deliberately absent

- Corpus audio and transcripts.
- Per-utterance prediction files. These pair model output with reference text,
  so publishing them would republish the references.
- Trained adapter weights. They are derived from licensed data, and their
  redistribution terms are not ours to decide.

## Using your own data

Evaluation is manifest-driven. Point `whisper-sg evaluate` at a CSV of
`audio_id`, `audio_path`, and `transcript` describing files you are licensed to
hold. Relative paths resolve against the manifest's directory.

## Limitations

Every published figure describes IMDA NSC data. The corpus is read and prompted
speech recorded under specific conditions, so it is not a general sample of
Singaporean English, and results here do not transfer automatically to
spontaneous conversation, telephony audio, or other accents.

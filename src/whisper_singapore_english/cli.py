"""Command line entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from whisper_singapore_english.evaluation import evaluate, write_report
from whisper_singapore_english.lora import WHISPER_TINY, LoraSpec
from whisper_singapore_english.manifest import load_manifest


def _evaluate(args: argparse.Namespace) -> int:
    from whisper_singapore_english.asr import WhisperTranscriber

    entries = load_manifest(args.manifest, require_audio=True)
    print(f"{len(entries)} utterances from {args.manifest}")
    transcriber = WhisperTranscriber(
        model_id=args.model_id, adapter_path=args.adapter, device=args.device
    ).load()
    result = evaluate(entries, transcriber, split=args.split)
    breakdown = result.breakdown
    print(f"system              {result.system}")
    print(f"split               {result.split}")
    print(f"corpus WER          {breakdown.corpus_wer:.4f}")
    print(f"mean utterance WER  {breakdown.mean_utterance_wer:.4f}")
    print(
        f"errors              {breakdown.errors} "
        f"(S={breakdown.substitutions} D={breakdown.deletions} "
        f"I={breakdown.insertions}) "
        f"over {breakdown.reference_words} reference words"
    )
    if args.output is not None:
        write_report([result], args.output)
        print(f"wrote {args.output}")
    return 0


def _adapter_size(args: argparse.Namespace) -> int:
    spec = LoraSpec(r=args.r, alpha=args.alpha)
    total = spec.trainable_parameters(WHISPER_TINY)
    print(f"LoRA r={spec.r} alpha={spec.alpha} scaling={spec.scaling:g}")
    print(f"target modules: {', '.join(spec.target_modules)}")
    print(f"trainable parameters (whisper-tiny): {total:,}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whisper-sg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="score a checkpoint over a manifest")
    ev.add_argument("--manifest", type=Path, required=True)
    ev.add_argument("--model-id", default="openai/whisper-tiny")
    ev.add_argument("--adapter", type=Path, default=None)
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--split", default="test")
    ev.add_argument("--output", type=Path, default=None)
    ev.set_defaults(handler=_evaluate)

    sz = sub.add_parser("adapter-size", help="report LoRA parameter count")
    sz.add_argument("--r", type=int, default=8)
    sz.add_argument("--alpha", type=int, default=32)
    sz.set_defaults(handler=_adapter_size)

    args = parser.parse_args(argv)
    handler: object = args.handler
    assert callable(handler)
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())

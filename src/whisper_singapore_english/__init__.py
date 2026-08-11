"""LoRA adaptation and WER evaluation of Whisper for Singaporean-accented English."""

from whisper_singapore_english.lora import LoraSpec
from whisper_singapore_english.manifest import ManifestEntry, load_manifest
from whisper_singapore_english.wer import corpus_wer, utterance_wer

__all__ = [
    "LoraSpec",
    "ManifestEntry",
    "corpus_wer",
    "load_manifest",
    "utterance_wer",
]

"""Whisper-backed transcribers.

``torch``/``transformers``/``peft`` are imported lazily inside the constructors
so the evaluation core, the tests, and CI do not require them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16_000


@dataclass
class WhisperTranscriber:
    """A Whisper checkpoint, optionally with a LoRA adapter applied."""

    model_id: str = "openai/whisper-tiny"
    adapter_path: Path | None = None
    device: str = "cpu"
    language: str = "en"
    _model: Any = None
    _processor: Any = None

    @property
    def name(self) -> str:
        return self.model_id if self.adapter_path is None else f"{self.model_id}+lora"

    def load(self) -> WhisperTranscriber:
        """Load the model and processor. Requires the ``asr`` extra."""
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as error:  # pragma: no cover - exercised by install
            raise ImportError(
                "model execution needs the 'asr' extra: pip install '.[asr]'"
            ) from error

        processor = WhisperProcessor.from_pretrained(self.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        if self.adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as error:  # pragma: no cover
                raise ImportError(
                    "adapters need the 'asr' extra: pip install '.[asr]'"
                ) from error
            model = PeftModel.from_pretrained(model, str(self.adapter_path))
            model = model.merge_and_unload()
        self._model = model.to(self.device).eval()
        self._processor = processor
        self._torch = torch
        return self

    def transcribe(self, audio_path: Path) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("call load() before transcribe()")
        import librosa

        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE)
        features = self._processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device)
        with self._torch.no_grad():
            tokens = self._model.generate(
                features, language=self.language, task="transcribe"
            )
        text: str = self._processor.batch_decode(tokens, skip_special_tokens=True)[0]
        return text

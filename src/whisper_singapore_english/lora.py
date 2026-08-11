"""LoRA configuration and trainable-parameter accounting.

The point of parameter-efficient fine-tuning is the parameter count, so the
adapter size is computed from the architecture rather than quoted from memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Whisper attention projections are square (d_model -> d_model); the feed-forward
# pair maps d_model <-> ffn_dim.
_SQUARE_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj")
_FFN_MODULES = ("fc1", "fc2")


@dataclass(frozen=True)
class WhisperDimensions:
    """The architectural sizes needed to size a LoRA adapter."""

    d_model: int
    ffn_dim: int
    encoder_layers: int
    decoder_layers: int

    def __post_init__(self) -> None:
        for name in ("d_model", "ffn_dim", "encoder_layers", "decoder_layers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")


# openai/whisper-tiny
WHISPER_TINY = WhisperDimensions(
    d_model=384, ffn_dim=1536, encoder_layers=4, decoder_layers=4
)


@dataclass(frozen=True)
class LoraSpec:
    """The adapter configuration used for the reported run."""

    r: int = 8
    alpha: int = 32
    dropout: float = 0.1
    bias: str = "none"
    target_modules: tuple[str, ...] = field(
        default=("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
    )

    def __post_init__(self) -> None:
        if self.r < 1:
            raise ValueError("r must be at least 1")
        if self.alpha < 1:
            raise ValueError("alpha must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        unknown = set(self.target_modules) - set(_SQUARE_MODULES) - set(_FFN_MODULES)
        if unknown:
            raise ValueError(f"unknown target modules: {sorted(unknown)}")

    @property
    def scaling(self) -> float:
        """The alpha/r factor applied to the adapter output."""
        return self.alpha / self.r

    def _module_parameters(self, module: str, dims: WhisperDimensions) -> int:
        if module in _SQUARE_MODULES:
            return self.r * (dims.d_model + dims.d_model)
        return self.r * (dims.d_model + dims.ffn_dim)

    def trainable_parameters(self, dims: WhisperDimensions = WHISPER_TINY) -> int:
        """Return the total LoRA parameter count for ``dims``.

        Encoder layers carry self-attention only. Decoder layers additionally
        carry cross-attention, so attention projections there are counted twice.
        """
        attention = sum(
            self._module_parameters(m, dims)
            for m in self.target_modules
            if m in _SQUARE_MODULES
        )
        feed_forward = sum(
            self._module_parameters(m, dims)
            for m in self.target_modules
            if m in _FFN_MODULES
        )
        encoder = dims.encoder_layers * (attention + feed_forward)
        decoder = dims.decoder_layers * (2 * attention + feed_forward)
        return encoder + decoder

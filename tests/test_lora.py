from __future__ import annotations

import pytest

from whisper_singapore_english.lora import WHISPER_TINY, LoraSpec, WhisperDimensions


def test_reported_run_matches_the_published_adapter_size() -> None:
    # Known answer: the released whisper-tiny adapter contains 540,672 LoRA
    # parameters across 128 tensors, read from adapter_model.safetensors.
    assert LoraSpec().trainable_parameters(WHISPER_TINY) == 540_672


def test_scaling_is_alpha_over_r() -> None:
    assert LoraSpec(r=8, alpha=32).scaling == pytest.approx(4.0)


def test_decoder_layers_cost_more_than_encoder_layers() -> None:
    spec = LoraSpec()
    # Same total depth, weighted towards the decoder. Decoder layers carry
    # cross-attention on top of self-attention, so they cost strictly more.
    encoder_heavy = WhisperDimensions(384, 1536, encoder_layers=3, decoder_layers=1)
    decoder_heavy = WhisperDimensions(384, 1536, encoder_layers=1, decoder_layers=3)
    assert spec.trainable_parameters(decoder_heavy) > spec.trainable_parameters(
        encoder_heavy
    )


def test_layer_counts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="decoder_layers must be at least 1"):
        WhisperDimensions(384, 1536, encoder_layers=4, decoder_layers=0)


def test_rank_scales_the_adapter_linearly() -> None:
    assert (
        LoraSpec(r=16).trainable_parameters()
        == 2 * LoraSpec(r=8).trainable_parameters()
    )


def test_attention_only_adapter_is_smaller() -> None:
    attention = LoraSpec(target_modules=("q_proj", "k_proj", "v_proj", "out_proj"))
    assert attention.trainable_parameters() < LoraSpec().trainable_parameters()


def test_invalid_configurations_are_rejected() -> None:
    with pytest.raises(ValueError, match="r must be at least 1"):
        LoraSpec(r=0)
    with pytest.raises(ValueError, match="dropout"):
        LoraSpec(dropout=1.0)
    with pytest.raises(ValueError, match="unknown target modules"):
        LoraSpec(target_modules=("not_a_module",))
    with pytest.raises(ValueError, match="must not be empty"):
        LoraSpec(target_modules=())

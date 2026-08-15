"""CPU-only reference operators for the SpikingBrain cluster prototype."""

from .ops import (
    RMSNorm,
    dynamic_spikes,
    fake_quantize_weight,
    gla_recurrent,
    quant_linear,
    quantize_symmetric,
    sliding_window_attention,
    swiglu,
    swish,
)

__all__ = [
    "RMSNorm",
    "dynamic_spikes",
    "fake_quantize_weight",
    "gla_recurrent",
    "quant_linear",
    "quantize_symmetric",
    "sliding_window_attention",
    "swiglu",
    "swish",
]

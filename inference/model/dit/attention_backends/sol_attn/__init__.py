"""SM120 Sol-Attn backend adapted for MagiHuman's multimodal GQA layout."""

from .core import (
    SolAttnConfig,
    SolAttnSession,
    _dense_gqa,
    scheduled_sol_attn_attention,
    sol_attn_attention,
)

__all__ = [
    "SolAttnConfig",
    "SolAttnSession",
    "_dense_gqa",
    "scheduled_sol_attn_attention",
    "sol_attn_attention",
]

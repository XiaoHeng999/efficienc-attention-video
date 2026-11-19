"""MC-Sparse reference attention backend for MagiHuman."""

from .core import (
    MCSparseCache,
    MCSparseConfig,
    MCSparseCore,
    MCSparseSession,
    _dense_gqa,
    scheduled_mcsparse_attention,
)

__all__ = [
    "MCSparseCache",
    "MCSparseConfig",
    "MCSparseCore",
    "MCSparseSession",
    "_dense_gqa",
    "scheduled_mcsparse_attention",
]

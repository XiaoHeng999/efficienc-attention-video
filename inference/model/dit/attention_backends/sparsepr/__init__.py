"""Independent SparsePR backend for MagiHuman inference."""

from .core import (
    SparsePRConfig,
    SparsePRCore,
    SparsePRSession,
    _dense_gqa,
    scheduled_sparsepr_attention,
)

__all__ = [
    "SparsePRConfig",
    "SparsePRCore",
    "SparsePRSession",
    "scheduled_sparsepr_attention",
    "_dense_gqa",
]

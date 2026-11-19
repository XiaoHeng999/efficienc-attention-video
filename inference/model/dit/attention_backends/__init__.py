"""VC-Attention source backend and offline numerical helpers."""

from .vc_reference import value_permutation, vc_attention_reference


def vc_attention(*args, **kwargs):
    from .vc_attention import vc_attention as run

    return run(*args, **kwargs)


def nvfp4_attention(*args, **kwargs):
    from .vc_attention import nvfp4_attention as run

    return run(*args, **kwargs)


__all__ = [
    "value_permutation",
    "vc_attention_reference",
    "vc_attention",
    "nvfp4_attention",
]

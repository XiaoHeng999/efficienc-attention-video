"""MagiHuman scheduling and multimodal policy for the SM120 Sol-Attn backend."""

from __future__ import annotations

import math
import os
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable

import torch

from inference.common import Modality

from .reference import dense_gqa as _dense_gqa


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class SolAttnConfig:
    tau: float = -1.0
    dense_step_ratio: float = 0.20
    dense_layers: int = 1
    min_video_tokens: int = 8192
    executor: str = "sm120"
    strict: bool = False

    def __post_init__(self):
        if not math.isfinite(self.tau):
            raise ValueError("Sol-Attn tau must be finite")
        if not 0 <= self.dense_step_ratio <= 1:
            raise ValueError("Sol-Attn dense_step_ratio must be in [0, 1]")
        if self.dense_layers < 0 or self.min_video_tokens < 0:
            raise ValueError("Sol-Attn layer/token guards must be nonnegative")
        if self.executor not in {"sm120", "reference"}:
            raise ValueError("Sol-Attn executor must be 'sm120' or 'reference'")

    @classmethod
    def from_env(cls) -> "SolAttnConfig":
        return cls(
            tau=float(os.environ.get("MAGI_SOL_ATTN_TAU", "-1.0")),
            dense_step_ratio=float(
                os.environ.get("MAGI_SOL_ATTN_DENSE_STEP_RATIO", "0.20")
            ),
            dense_layers=int(os.environ.get("MAGI_SOL_ATTN_DENSE_LAYERS", "1")),
            min_video_tokens=int(
                os.environ.get("MAGI_SOL_ATTN_MIN_VIDEO_TOKENS", "8192")
            ),
            executor=os.environ.get("MAGI_SOL_ATTN_EXECUTOR", "sm120").lower(),
            strict=_env_bool("MAGI_SOL_ATTN_STRICT", False),
        )


_FALLBACK_WARNED = False


def _warn_fallback(reason: str):
    global _FALLBACK_WARNED
    if not _FALLBACK_WARNED:
        warnings.warn(f"Sol-Attn fell back to dense attention: {reason}", RuntimeWarning)
        _FALLBACK_WARNED = True


def _video_prefix_length(modality: torch.Tensor) -> int | None:
    if modality.ndim != 1:
        return None
    non_video = torch.nonzero(modality != int(Modality.VIDEO), as_tuple=True)[0]
    video_tokens = int(non_video[0].item()) if non_video.numel() else modality.numel()
    if video_tokens == 0:
        return 0
    if not bool(torch.all(modality[:video_tokens] == int(Modality.VIDEO)).item()):
        return None
    if bool(torch.any(modality[video_tokens:] == int(Modality.VIDEO)).item()):
        return None
    return video_tokens


def sol_attn_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    modality: torch.Tensor,
    config: SolAttnConfig,
    *,
    dense_attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply Sol-Attn to video Q and preserve exact multimodal anchor behavior."""
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[0] != 1:
        raise ValueError("Sol-Attn expects Q [1,S,Hq,D] and matching K/V")
    if q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1]:
        raise ValueError("MagiHuman Sol-Attn expects equal full Q/K sequence lengths")
    if q.shape[2] % k.shape[2]:
        raise ValueError("Sol-Attn requires divisible GQA heads")
    if modality.numel() != q.shape[1]:
        raise ValueError("Sol-Attn modality map must match the post-CP sequence")
    video_tokens = _video_prefix_length(modality)
    if video_tokens is None:
        _warn_fallback("video tokens are not a contiguous prefix")
        return dense_attention(q, k, v)
    if video_tokens < config.min_video_tokens or video_tokens == 0:
        return dense_attention(q, k, v)

    q_video = q[:, :video_tokens].contiguous()
    try:
        if config.executor == "reference":
            from .reference import sol_attn_reference

            video_output = sol_attn_reference(
                q_video,
                k,
                v,
                video_tokens=video_tokens,
                tau=config.tau,
            )
        else:
            from .interface import sol_attn_sm120_gqa

            video_output = sol_attn_sm120_gqa(
                q_video,
                k.contiguous(),
                v.contiguous(),
                video_tokens=video_tokens,
                tau=config.tau,
            )
    except (ImportError, RuntimeError) as error:
        if config.strict:
            raise
        _warn_fallback(str(error))
        return dense_attention(q, k, v)

    if video_tokens == q.shape[1]:
        return video_output
    anchor_output = dense_attention(q[:, video_tokens:], k, v)
    return torch.cat((video_output, anchor_output), dim=1)


@dataclass
class _StepScope:
    session: "SolAttnSession"
    index: int
    branch: str


_ACTIVE: ContextVar[_StepScope | None] = ContextVar("sol_attn_step", default=None)


class SolAttnSession:
    """Own the denoising schedule for one generation without cross-run state."""

    def __init__(self, total_steps: int, config: SolAttnConfig | None = None):
        if total_steps < 1:
            raise ValueError("Sol-Attn total_steps must be positive")
        self.total_steps = total_steps
        self.config = config or SolAttnConfig.from_env()

    @contextmanager
    def step(self, index: int, branch: str):
        if not 0 <= index < self.total_steps or not branch:
            raise ValueError("Expected an in-range Sol-Attn step and nonempty branch")
        token = _ACTIVE.set(_StepScope(self, index, branch))
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def run(self, q, k, v, modality, dense_attention, *, layer_idx: int):
        scope = _ACTIVE.get()
        assert scope is not None and scope.session is self
        if (
            scope.index < math.floor(self.total_steps * self.config.dense_step_ratio)
            or layer_idx < self.config.dense_layers
        ):
            return dense_attention(q, k, v)
        return sol_attn_attention(
            q,
            k,
            v,
            modality,
            self.config,
            dense_attention=dense_attention,
        )


def scheduled_sol_attn_attention(q, k, v, modality, dense_attention, *, layer_idx: int):
    if layer_idx < 0:
        raise ValueError("Sol-Attn layer_idx must be nonnegative")
    scope = _ACTIVE.get()
    if scope is not None:
        return scope.session.run(
            q, k, v, modality, dense_attention, layer_idx=layer_idx
        )
    config = SolAttnConfig.from_env()
    if layer_idx < config.dense_layers:
        return dense_attention(q, k, v)
    return sol_attn_attention(
        q,
        k,
        v,
        modality,
        config,
        dense_attention=dense_attention,
    )


__all__ = [
    "SolAttnConfig",
    "SolAttnSession",
    "_dense_gqa",
    "scheduled_sol_attn_attention",
    "sol_attn_attention",
]

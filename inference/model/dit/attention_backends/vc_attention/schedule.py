"""Amortized V-Smooth schedule for one MagiHuman denoising invocation."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch


@dataclass
class _StepScope:
    session: "VCAttentionSession"
    step: int
    branch: str
    call_index: int = 0


_ACTIVE: ContextVar[_StepScope | None] = ContextVar("vc_attention_step", default=None)


class VCAttentionSession:
    """Cache per-layer value permutations and warm-start centroids.

    V-Smooth is active only during the first quarter of denoising. Within that
    window one grouping is followed by three reuse steps. The final permutation
    is retained for the later plain FP8-PV calls.
    Conditional and unconditional CFG branches use independent caches.
    """

    refresh_interval = 4

    def __init__(self, total_steps: int):
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.total_steps = total_steps
        self.grouped_steps = max(1, (total_steps + 3) // 4)
        self.permutations: dict[tuple, torch.Tensor] = {}
        self.centers: dict[tuple, torch.Tensor] = {}

    @contextmanager
    def step(self, index: int, branch: str):
        if not 0 <= index < self.total_steps or not branch:
            raise ValueError("Expected an in-range denoising step and branch")
        token = _ACTIVE.set(_StepScope(self, index, branch))
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def _slot(self, q: torch.Tensor, k: torch.Tensor, scope: _StepScope):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("VC-Attention grouping cannot run during CUDA Graph capture")
        position = scope.call_index
        scope.call_index += 1
        key = (scope.branch, position, tuple(q.shape), tuple(k.shape), q.device)
        cached = self.permutations.get(key)
        smooth = scope.step < self.grouped_steps
        refresh = cached is None or (smooth and scope.step % self.refresh_interval == 0)
        return key, cached, smooth, refresh

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scope: _StepScope):
        from ..vc import _value_permutation, vc_attention_fa4_fp8

        key, permutation, smooth, refresh = self._slot(q, k, scope)
        if refresh:
            permutation, centers = _value_permutation(
                v[0].permute(1, 0, 2).contiguous(),
                initial_centers=self.centers.get(key),
                return_centers=True,
            )
            self.permutations[key] = permutation
            self.centers[key] = centers
        return vc_attention_fa4_fp8(q, k, v, permutation=permutation, smooth=smooth)

    def run_nvfp4(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scope: _StepScope):
        from ..vc import _value_permutation
        from .nvfp4 import vc_attention_nvfp4

        key, permutation, _smooth, refresh = self._slot(q, k, scope)
        if refresh:
            permutation, centers = _value_permutation(
                v[0].permute(1, 0, 2).contiguous(),
                initial_centers=self.centers.get(key),
                return_centers=True,
            )
            self.permutations[key] = permutation
            self.centers[key] = centers
        pv_mode = os.environ.get("MAGI_NVFP4_PV_MODE", "fp8").lower()
        return vc_attention_nvfp4(q, k, v, permutation=permutation, pv_mode=pv_mode)


def scheduled_vc_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Run the active amortized schedule or one standalone smoothed call."""
    scope = _ACTIVE.get()
    if scope is None:
        from ..vc import vc_attention_fa4_fp8

        return vc_attention_fa4_fp8(q, k, v, smooth=True)
    return scope.session.run(q, k, v, scope)


def scheduled_nvfp4_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Use the VC permutation schedule with the optional NVFP4 kernel."""
    scope = _ACTIVE.get()
    if scope is None:
        from ..vc import _value_permutation
        from .nvfp4 import vc_attention_nvfp4

        permutation = _value_permutation(v[0].permute(1, 0, 2).contiguous())
        pv_mode = os.environ.get("MAGI_NVFP4_PV_MODE", "fp8").lower()
        return vc_attention_nvfp4(q, k, v, permutation=permutation, pv_mode=pv_mode)
    return scope.session.run_nvfp4(q, k, v, scope)


__all__ = ["VCAttentionSession", "scheduled_vc_attention", "scheduled_nvfp4_attention"]

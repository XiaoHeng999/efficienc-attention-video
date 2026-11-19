"""SM120 Sol-Attn block summaries and native-GQA routing thresholds."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 64
HEAD_DIM = 128
_BLOCK_SIZE = tl.constexpr(64)
_HEAD_DIM = tl.constexpr(128)


@triton.jit
def _reduce_kv_kernel(
    k,
    v,
    kc,
    vc,
    tokens,
    blocks: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
):
    block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    token_offsets = block * _BLOCK_SIZE + tl.arange(0, _BLOCK_SIZE)
    dim_offsets = tl.arange(0, _HEAD_DIM)
    offsets = (
        ((batch * tokens + token_offsets[:, None]) * heads + head) * dim
        + dim_offsets[None, :]
    )
    mask = (token_offsets[:, None] < tokens) & (dim_offsets[None, :] < dim)
    k_values = tl.load(k + offsets, mask=mask, other=0.0).to(tl.float32)
    v_values = tl.load(v + offsets, mask=mask, other=0.0).to(tl.float32)
    block_len = tl.minimum(_BLOCK_SIZE, tokens - block * _BLOCK_SIZE).to(tl.float32)
    output_offsets = ((batch * blocks + block) * heads + head) * dim + dim_offsets
    tl.store(
        kc + output_offsets,
        tl.sum(k_values, axis=0) / block_len,
        mask=dim_offsets < dim,
    )
    tl.store(
        vc + output_offsets,
        tl.sum(v_values, axis=0),
        mask=dim_offsets < dim,
    )


@triton.jit
def _pool_q_kernel(
    q,
    q_bar,
    tokens,
    blocks: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
):
    block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    token_offsets = block * _BLOCK_SIZE + tl.arange(0, _BLOCK_SIZE)
    dim_offsets = tl.arange(0, _HEAD_DIM)
    offsets = (
        ((batch * tokens + token_offsets[:, None]) * heads + head) * dim
        + dim_offsets[None, :]
    )
    mask = (token_offsets[:, None] < tokens) & (dim_offsets[None, :] < dim)
    values = tl.load(q + offsets, mask=mask, other=0.0).to(tl.float32)
    block_len = tl.minimum(_BLOCK_SIZE, tokens - block * _BLOCK_SIZE).to(tl.float32)
    output_offsets = ((batch * blocks + block) * heads + head) * dim + dim_offsets
    tl.store(
        q_bar + output_offsets,
        tl.sum(values, axis=0) / block_len,
        mask=dim_offsets < dim,
    )


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    video_tokens: int,
    tau: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build K means, V sums and per-Q-block diagonal thresholds.

    Threshold statistics intentionally use video K blocks only.  The suffix
    audio/text blocks are forced exact by the mainloop and therefore must not
    skew the sparse video routing distribution.
    """
    batch, q_tokens, q_heads, head_dim = q.shape
    _, kv_tokens, kv_heads, _ = k.shape
    q_blocks = triton.cdiv(q_tokens, BLOCK_SIZE)
    kv_blocks = triton.cdiv(kv_tokens, BLOCK_SIZE)
    # A mixed boundary block is an exact sink in the mainloop.  Exclude it
    # from routing statistics so suffix anchor tokens cannot shift thresholds.
    video_stat_blocks = max(video_tokens // BLOCK_SIZE, 1)
    kc = torch.empty(
        (batch, kv_blocks, kv_heads, head_dim),
        device=k.device,
        dtype=torch.bfloat16,
    )
    vc = torch.empty_like(kc)
    q_bar = torch.empty(
        (batch, q_blocks, q_heads, head_dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    _reduce_kv_kernel[(kv_blocks, batch * kv_heads)](
        k,
        v,
        kc,
        vc,
        kv_tokens,
        kv_blocks,
        kv_heads,
        head_dim,
        num_warps=8,
        num_stages=2,
    )
    _pool_q_kernel[(q_blocks, batch * q_heads)](
        q,
        q_bar,
        q_tokens,
        q_blocks,
        q_heads,
        head_dim,
        num_warps=8,
        num_stages=2,
    )

    kc_video = kc[:, :video_stat_blocks].float()
    kc_mean = kc_video.mean(dim=1)
    kc_var = kc_video.var(dim=1, correction=0)
    kv_for_q = torch.arange(q_heads, device=q.device) // (q_heads // kv_heads)
    q_bar_fp32 = q_bar.float()
    selected_mean = kc_mean[:, kv_for_q].unsqueeze(1)
    selected_var = kc_var[:, kv_for_q].unsqueeze(1)
    log2_scale = float(scale) * math.log2(math.e)
    mean = (q_bar_fp32 * selected_mean).sum(dim=-1) * log2_scale
    variance = (
        (q_bar_fp32.square() * selected_var).sum(dim=-1)
        * (log2_scale * log2_scale)
    )
    threshold = mean + float(tau) * torch.sqrt(variance.clamp_min(0) + 1.0e-6)
    return kc, vc, threshold.contiguous()


__all__ = ["BLOCK_SIZE", "prepare"]

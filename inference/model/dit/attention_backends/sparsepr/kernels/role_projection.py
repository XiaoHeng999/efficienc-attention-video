# Derived from SparsePR revision e46646f.
# SPDX-License-Identifier: Apache-2.0
"""Fused response-coordinate projection and row RMS normalization."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _role_projection_rms_kernel(
    x,
    factor,
    output,
    tokens: tl.constexpr,
    input_dim: tl.constexpr,
    rank: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_r: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_head = tl.program_id(1)
    rows = pid_m * block_m + tl.arange(0, block_m)
    k_offsets = tl.arange(0, block_k)
    r_offsets = tl.arange(0, block_r)
    row_mask = rows < tokens
    k_mask = k_offsets < input_dim
    r_mask = r_offsets < rank

    values = tl.load(
        x
        + (pid_head * tokens + rows[:, None]) * input_dim
        + k_offsets[None, :],
        mask=row_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    transform = tl.load(
        factor
        + (pid_head * input_dim + k_offsets[:, None]) * rank
        + r_offsets[None, :],
        mask=k_mask[:, None] & r_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    projected = tl.dot(values, transform, input_precision="tf32")
    sum_square = tl.sum(
        tl.where(r_mask[None, :], projected * projected, 0.0), axis=1
    )
    inverse_rms = tl.rsqrt(tl.maximum(sum_square / rank, 1.0e-12))
    tl.store(
        output
        + (pid_head * tokens + rows[:, None]) * rank
        + r_offsets[None, :],
        projected * inverse_rms[:, None],
        mask=row_mask[:, None] & r_mask[None, :],
    )


def role_projection_rms_triton(
    x: torch.Tensor,
    factor: torch.Tensor,
) -> torch.Tensor:
    """Project `[H,N,D]` through `[H,D,R]` and RMS-normalize rows."""
    if not x.is_cuda or not factor.is_cuda:
        raise RuntimeError("SparsePR fused role projection requires CUDA")
    if x.ndim != 3 or factor.ndim != 3:
        raise ValueError("SparsePR role projection expects rank-three tensors")
    heads, tokens, dim = x.shape
    if factor.shape[:2] != (heads, dim):
        raise ValueError("SparsePR role projection received incompatible shapes")
    rank = factor.shape[2]
    if dim > 256 or rank > 128 or min(dim, rank) < 16:
        raise ValueError("SparsePR fused role projection supports 16<=D,R<=128/256")
    output = torch.empty(
        (heads, tokens, rank), device=x.device, dtype=torch.float32
    )
    block_m = 32
    _role_projection_rms_kernel[(triton.cdiv(tokens, block_m), heads)](
        x.contiguous(),
        factor.contiguous(),
        output,
        tokens=tokens,
        input_dim=dim,
        rank=rank,
        block_m=block_m,
        block_k=triton.next_power_of_2(dim),
        block_r=triton.next_power_of_2(rank),
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["role_projection_rms_triton"]

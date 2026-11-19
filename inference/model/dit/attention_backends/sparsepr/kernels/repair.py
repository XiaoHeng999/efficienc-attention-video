# Derived from SparsePR revision e46646f.
# SPDX-License-Identifier: Apache-2.0
"""Fused reduced-rank residual repair application."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _repair_apply_kernel(
    base,
    video_indices,
    x_mean,
    x_scale,
    residual_mean,
    left,
    basis,
    tokens: tl.constexpr,
    video_tokens: tl.constexpr,
    dim: tl.constexpr,
    rank: tl.constexpr,
    alpha: tl.constexpr,
    norm_cap: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    rows = pid_m * block_m + tl.arange(0, block_m)
    columns = tl.arange(0, block_d)
    row_mask = rows < video_tokens
    column_mask = columns < dim
    token = tl.load(video_indices + rows, mask=row_mask, other=0).to(tl.int64)
    base_offsets = (
        (head * tokens + token[:, None]) * dim + columns[None, :]
    )
    base_row = tl.load(
        base + base_offsets,
        mask=row_mask[:, None] & column_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(
        x_mean + head * dim + columns,
        mask=column_mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(
        x_scale + head * dim + columns,
        mask=column_mask,
        other=1.0,
    ).to(tl.float32)
    normalized = (base_row - mean[None, :]) / scale[None, :]
    rank_columns = tl.arange(0, rank)
    left_matrix = tl.load(
        left
        + (head * dim + columns[:, None]) * rank
        + rank_columns[None, :],
        mask=column_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    coordinates = tl.dot(normalized, left_matrix, input_precision="tf32")
    basis_transposed = tl.load(
        basis
        + (head * dim + columns[None, :]) * rank
        + rank_columns[:, None],
        mask=column_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    correction = tl.dot(coordinates, basis_transposed, input_precision="tf32")
    correction += tl.load(
        residual_mean + head * dim + columns,
        mask=column_mask,
        other=0.0,
    )[None, :]
    if norm_cap > 0.0:
        correction_norm = tl.sqrt(tl.sum(correction * correction, axis=1))
        base_norm = tl.sqrt(tl.sum(base_row * base_row, axis=1))
        repair_scale = tl.minimum(
            1.0,
            norm_cap * base_norm / tl.maximum(correction_norm, 1e-20),
        )
        correction *= repair_scale[:, None]
    tl.store(
        base + base_offsets,
        base_row + alpha * correction,
        mask=row_mask[:, None] & column_mask[None, :],
    )


def repair_apply_triton(
    base: torch.Tensor,
    video_indices: torch.Tensor,
    x_mean: torch.Tensor,
    x_scale: torch.Tensor,
    residual_mean: torch.Tensor,
    left: torch.Tensor,
    basis: torch.Tensor,
    alpha: float = 1.0,
    norm_cap: float = 0.0,
) -> torch.Tensor:
    """Apply low-rank repair in-place without full-sequence FP32 tensors."""
    if not base.is_cuda:
        raise RuntimeError("SparsePR fused repair requires CUDA")
    if base.ndim != 3:
        raise ValueError("SparsePR fused repair expects [H,N,D] base")
    heads, tokens, dim = base.shape
    if dim > 256 or dim < 16:
        raise ValueError("SparsePR fused repair supports 16<=D<=256")
    if left.ndim != 3 or left.shape[:2] != (heads, dim):
        raise ValueError("SparsePR fused repair left-factor shape mismatch")
    rank = left.shape[2]
    if rank < 16 or rank & (rank - 1):
        raise ValueError("SparsePR fused repair rank must be a power of two >=16")
    if basis.shape != (heads, dim, rank):
        raise ValueError("SparsePR fused repair basis shape mismatch")
    block_m = 32
    block_d = triton.next_power_of_2(dim)
    _repair_apply_kernel[(triton.cdiv(video_indices.numel(), block_m), heads)](
        base,
        video_indices.contiguous(),
        x_mean.contiguous(),
        x_scale.contiguous(),
        residual_mean.contiguous(),
        left.contiguous(),
        basis.contiguous(),
        tokens=tokens,
        video_tokens=video_indices.numel(),
        dim=dim,
        rank=rank,
        alpha=float(alpha),
        norm_cap=float(norm_cap),
        block_m=block_m,
        block_d=block_d,
        num_warps=4,
        num_stages=1,
    )
    return base


__all__ = ["repair_apply_triton"]

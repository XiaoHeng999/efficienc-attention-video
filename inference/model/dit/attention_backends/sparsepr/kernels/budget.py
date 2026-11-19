# Derived from SparsePR revision e46646f.
# SPDX-License-Identifier: Apache-2.0
"""Exact whole-cluster key-budget epilogue for SparsePR routing."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _budget_mask_kernel(
    order,
    sizes,
    budgets,
    output,
    q_clusters: tl.constexpr,
    k_clusters: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    head = row // q_clusters
    offsets = tl.arange(0, block_k)
    valid = offsets < k_clusters
    ordered_cluster = tl.load(
        order + row * k_clusters + offsets,
        mask=valid,
        other=0,
    ).to(tl.int32)
    costs = tl.load(
        sizes + head * k_clusters + ordered_cluster,
        mask=valid,
        other=0,
    ).to(tl.int32)
    positive = valid & (costs > 0)
    cumulative = tl.cumsum(tl.where(positive, costs, 0), axis=0)
    budget = tl.load(budgets + row).to(tl.int32)
    keep = positive & (cumulative <= budget)
    selected_cost = tl.sum(tl.where(keep, costs, 0), axis=0)
    selected_count = tl.sum(keep.to(tl.int32), axis=0)
    valid_count = tl.sum(positive.to(tl.int32), axis=0)
    include_crossing = (
        (budget > 0)
        & (selected_cost < budget)
        & (selected_count < valid_count)
    )
    keep |= (offsets == selected_count) & include_crossing
    tl.store(
        output + row * k_clusters + ordered_cluster,
        keep,
        mask=valid,
    )


def exact_budget_mask_triton(
    scores: torch.Tensor,
    cluster_sizes: torch.Tensor,
    budgets: torch.Tensor,
) -> torch.Tensor:
    """Sort scores and select the shortest whole-cluster budget prefix."""
    if not scores.is_cuda:
        raise RuntimeError("SparsePR exact budget kernel requires CUDA")
    heads, q_clusters, k_clusters = scores.shape
    if cluster_sizes.shape != (heads, k_clusters):
        raise ValueError("SparsePR budget key sizes do not match scores")
    if budgets.shape != (heads, q_clusters):
        raise ValueError("SparsePR budgets do not match scores")
    if k_clusters > 2048:
        raise ValueError("SparsePR exact budget supports at most 2048 K clusters")
    order = torch.argsort(scores, dim=2, descending=True).to(torch.int32)
    output = torch.zeros_like(scores, dtype=torch.bool)
    _budget_mask_kernel[(heads * q_clusters,)](
        order.contiguous(),
        cluster_sizes.to(torch.int32).contiguous(),
        budgets.to(torch.int32).contiguous(),
        output,
        q_clusters=q_clusters,
        k_clusters=k_clusters,
        block_k=triton.next_power_of_2(k_clusters),
        num_warps=8,
        num_stages=1,
    )
    return output


__all__ = ["exact_budget_mask_triton"]

# Derived from SparsePR revision e46646f.
# SPDX-License-Identifier: Apache-2.0
"""Exact attention-mass routing score for the private SparsePR backend.

Adapted from SparsePR's Triton selector at revision e46646f. The wrapper is
locally namespaced and has no runtime dependency on the reference repository.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_mass_score_kernel(
    q_centroids,
    k_sorted,
    k_offsets,
    scores,
    q_clusters: tl.constexpr,
    k_clusters: tl.constexpr,
    tokens: tl.constexpr,
    dim: tl.constexpr,
    gqa_group: tl.constexpr,
    block_q: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_kc = tl.program_id(1)
    pid_qh = tl.program_id(2)
    pid_kh = pid_qh // gqa_group

    q_idx = pid_q * block_q + tl.arange(0, block_q)
    d_idx = tl.arange(0, dim)
    n_idx = tl.arange(0, block_n)
    q_mask = q_idx < q_clusters
    q = tl.load(
        q_centroids
        + (pid_qh * q_clusters + q_idx[:, None]) * dim
        + d_idx[None, :],
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    start = tl.load(k_offsets + pid_kh * (k_clusters + 1) + pid_kc).to(
        tl.int32
    )
    end = tl.load(k_offsets + pid_kh * (k_clusters + 1) + pid_kc + 1).to(
        tl.int32
    )
    scale = 1.0 / tl.sqrt(tl.full((), dim, tl.float32))
    local_max = tl.full((block_q,), -float("inf"), tl.float32)
    exp_sum = tl.zeros((block_q,), tl.float32)

    token_start = start
    while token_start < end:
        token_idx = token_start + n_idx
        token_mask = token_idx < end
        keys = tl.load(
            k_sorted
            + (pid_kh * tokens + token_idx[:, None]) * dim
            + d_idx[None, :],
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        logits = tl.dot(q, tl.trans(keys), input_precision="tf32") * scale
        masked_logits = tl.where(
            q_mask[:, None] & token_mask[None, :], logits, -float("inf")
        )
        chunk_max = tl.max(masked_logits, axis=1)
        new_max = tl.maximum(local_max, chunk_max)
        exp_sum = exp_sum * tl.exp(local_max - new_max)
        exp_sum += tl.sum(
            tl.where(
                q_mask[:, None] & token_mask[None, :],
                tl.exp(logits - new_max[:, None]),
                0.0,
            ),
            axis=1,
        )
        local_max = new_max
        token_start += block_n

    score = tl.log(tl.maximum(exp_sum, 1.0e-30)) + local_max
    score = tl.where(q_mask & (end > start), score, -float("inf"))
    tl.store(
        scores + (pid_qh * q_clusters + q_idx) * k_clusters + pid_kc,
        score,
        mask=q_mask,
    )


def attention_mass_scores_triton(
    q_centroids: torch.Tensor,
    k_sorted: torch.Tensor,
    k_cluster_sizes: torch.Tensor,
    gqa_group: int,
) -> torch.Tensor:
    """Return exact per-cluster log attention numerators."""
    if not q_centroids.is_cuda or not k_sorted.is_cuda:
        raise RuntimeError("SparsePR Triton selector requires CUDA tensors")
    q_heads, q_clusters, dim = q_centroids.shape
    kv_heads, tokens, k_dim = k_sorted.shape
    if dim != 128 or k_dim != 128:
        raise ValueError("SparsePR Triton selector requires head_dim=128")
    if q_heads != kv_heads * gqa_group:
        raise ValueError("SparsePR Triton selector received incompatible GQA heads")
    k_clusters = k_cluster_sizes.shape[1]
    offsets = torch.cat(
        (
            torch.zeros(
                (kv_heads, 1),
                dtype=torch.int32,
                device=k_cluster_sizes.device,
            ),
            k_cluster_sizes.to(torch.int32).cumsum(1),
        ),
        dim=1,
    ).contiguous()
    scores = torch.empty(
        (q_heads, q_clusters, k_clusters),
        dtype=torch.float32,
        device=q_centroids.device,
    )
    block_q, block_n = 16, 32
    _attention_mass_score_kernel[
        (triton.cdiv(q_clusters, block_q), k_clusters, q_heads)
    ](
        q_centroids.contiguous(),
        k_sorted.contiguous(),
        offsets,
        scores,
        q_clusters=q_clusters,
        k_clusters=k_clusters,
        tokens=tokens,
        dim=dim,
        gqa_group=gqa_group,
        block_q=block_q,
        block_n=block_n,
        num_warps=4,
        num_stages=1,
    )
    return scores


@triton.jit
def _grouped_value_error_kernel(
    q_centroids,
    k_sorted,
    v_sorted,
    k_centroids,
    v_centroids,
    k_offsets,
    scores,
    q_clusters: tl.constexpr,
    k_clusters: tl.constexpr,
    tokens: tl.constexpr,
    dim: tl.constexpr,
    gqa_group: tl.constexpr,
    block_q: tl.constexpr,
    block_gq: tl.constexpr,
    block_n: tl.constexpr,
):
    """Evaluate all Q heads in one GQA group with one K/V traversal."""
    pid_q = tl.program_id(0)
    pid_k_cluster = tl.program_id(1)
    pid_k_head = tl.program_id(2)

    grouped_q = tl.arange(0, block_gq)
    group_index = grouped_q // block_q
    q_local = grouped_q - group_index * block_q
    q_index = pid_q * block_q + q_local
    dim_index = tl.arange(0, dim)
    token_local = tl.arange(0, block_n)
    q_mask = (group_index < gqa_group) & (q_index < q_clusters)
    q_head = pid_k_head * gqa_group + group_index
    q_row = q_head * q_clusters + q_index
    query = tl.load(
        q_centroids + q_row[:, None] * dim + dim_index[None, :],
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    key_center = tl.load(
        k_centroids
        + (pid_k_head * k_clusters + pid_k_cluster) * dim
        + dim_index
    ).to(tl.float32)
    value_center = tl.load(
        v_centroids
        + (pid_k_head * k_clusters + pid_k_cluster) * dim
        + dim_index
    ).to(tl.float32)

    scale = 1.0 / tl.sqrt(tl.full((), dim, tl.float32))
    center_logits = tl.sum(query * key_center[None, :], axis=1) * scale
    local_max = center_logits
    center_weight = tl.full((block_gq,), 1.0, tl.float32)
    center_value_norm = tl.sum(value_center * value_center, axis=0)
    start = tl.load(
        k_offsets + pid_k_head * (k_clusters + 1) + pid_k_cluster
    ).to(tl.int32)
    end = tl.load(
        k_offsets + pid_k_head * (k_clusters + 1) + pid_k_cluster + 1
    ).to(tl.int32)
    error_sum = tl.zeros((block_gq,), dtype=tl.float32)

    token_start = start
    while token_start < end:
        token_index = token_start + token_local
        token_mask = token_index < end
        offsets = (
            (pid_k_head * tokens + token_index[:, None]) * dim
            + dim_index[None, :]
        )
        keys = tl.load(
            k_sorted + offsets,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        values = tl.load(
            v_sorted + offsets,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        logits = tl.dot(query, tl.trans(keys), input_precision="tf32") * scale
        masked_logits = tl.where(
            q_mask[:, None] & token_mask[None, :], logits, -float("inf")
        )
        chunk_max = tl.max(masked_logits, axis=1)
        new_max = tl.maximum(local_max, chunk_max)
        alpha = tl.exp(local_max - new_max)
        center_weight *= alpha
        error_sum *= alpha * alpha
        token_weight = tl.where(
            q_mask[:, None] & token_mask[None, :],
            tl.exp(logits - new_max[:, None]),
            0.0,
        )
        value_norm = tl.sum(values * values, axis=1)
        center_dot_value = tl.sum(
            values * value_center[None, :], axis=1
        )
        error = (
            center_weight[:, None]
            * center_weight[:, None]
            * center_value_norm
            + token_weight * token_weight * value_norm[None, :]
            - 2.0
            * token_weight
            * center_weight[:, None]
            * center_dot_value[None, :]
        )
        error_sum += tl.sum(
            tl.where(token_mask[None, :], tl.maximum(error, 0.0), 0.0),
            axis=1,
        )
        local_max = new_max
        token_start += block_n

    cluster_size = tl.maximum(end - start, 1).to(tl.float32)
    score = (
        tl.log(tl.maximum(error_sum / cluster_size, 1.0e-30))
        + 2.0 * local_max
    )
    score = tl.where(q_mask & (end > start), score, -float("inf"))
    tl.store(
        scores
        + (q_head * q_clusters + q_index) * k_clusters
        + pid_k_cluster,
        score,
        mask=q_mask,
    )


def grouped_value_error_scores_triton(
    q_centroids: torch.Tensor,
    k_sorted: torch.Tensor,
    v_sorted: torch.Tensor,
    k_centroids: torch.Tensor,
    v_centroids: torch.Tensor,
    k_cluster_sizes: torch.Tensor,
    gqa_group: int,
) -> torch.Tensor:
    """Paper SVG-EAR value-error score with native-KV GQA traversal."""
    if not q_centroids.is_cuda:
        raise RuntimeError("SparsePR grouped value-error selector requires CUDA")
    q_heads, q_clusters, dim = q_centroids.shape
    kv_heads, tokens, key_dim = k_sorted.shape
    k_clusters = k_centroids.shape[1]
    if dim != 128 or key_dim != 128:
        raise ValueError("SparsePR grouped value-error selector requires D=128")
    if q_heads != kv_heads * gqa_group or v_sorted.shape != k_sorted.shape:
        raise ValueError("SparsePR grouped value-error selector GQA mismatch")
    offsets = torch.cat(
        (
            torch.zeros(
                (kv_heads, 1),
                dtype=torch.int32,
                device=k_cluster_sizes.device,
            ),
            k_cluster_sizes.to(torch.int32).cumsum(1),
        ),
        dim=1,
    ).contiguous()
    scores = torch.empty(
        (q_heads, q_clusters, k_clusters),
        dtype=torch.float32,
        device=q_centroids.device,
    )
    block_q = 8
    padded_group = triton.next_power_of_2(gqa_group)
    block_gq = padded_group * block_q
    _grouped_value_error_kernel[
        (triton.cdiv(q_clusters, block_q), k_clusters, kv_heads)
    ](
        q_centroids.contiguous(),
        k_sorted.contiguous(),
        v_sorted.contiguous(),
        k_centroids.contiguous(),
        v_centroids.contiguous(),
        offsets,
        scores,
        q_clusters=q_clusters,
        k_clusters=k_clusters,
        tokens=tokens,
        dim=dim,
        gqa_group=gqa_group,
        block_q=block_q,
        block_gq=block_gq,
        block_n=16,
        num_warps=8,
        num_stages=1,
    )
    return scores


__all__ = [
    "attention_mass_scores_triton",
    "grouped_value_error_scores_triton",
]

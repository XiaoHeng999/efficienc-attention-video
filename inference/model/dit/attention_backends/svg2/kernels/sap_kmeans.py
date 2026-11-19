# Copyright 2024 MIT HAN Lab / Sparse VideoGen contributors.
# SPDX-License-Identifier: Apache-2.0
"""SAP Euclidean assignment and sorted centroid reduction kernels.

Vendored from Sparse-VideoGen f89aeda, svg/kmeans_utils.py.
The selected functions are unchanged; unrelated algorithms/dependencies and
profiling decorators are excluded. See LICENSE-Sparse-VideoGen.txt.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _centroid_update_chunk_kernel(
    x_ptr,  # *f16 / *f32 [B, N, D] – ORIGINAL ORDER
    sorted_idx_ptr,  # *i32        [B, N]    – indices after sort
    sorted_cluster_ptr,  # *i32        [B, N]    – cluster ids in sorted order
    sum_ptr,  # *f32        [B, K, D]
    count_ptr,  # *i32        [B, K]
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,  # how many tokens (points) each program processes
):
    """Each program processes **BLOCK_N consecutive, already-sorted tokens**.

    Because the tokens are sorted by cluster id, identical ids appear in
    contiguous runs.  We therefore accumulate a local sum/count for the
    current run and perform **a single atomic update per run**, instead of
    per-token.
    """
    # program indices – 2-D launch grid: (chunk_id, batch_id)
    pid_chunk = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    b = pid_b
    chunk_start = pid_chunk * BLOCK_N  # position of the first token handled by this program

    # Nothing to do – out of range
    if chunk_start >= N:
        return

    # base pointers for this batch
    idx_batch_base = sorted_idx_ptr + b * N
    cid_batch_base = sorted_cluster_ptr + b * N
    x_batch_base = x_ptr + b * N * D  # for pointer arithmetic

    # helper aranges
    offs_token = tl.arange(0, BLOCK_N)
    offs_dim = tl.arange(0, D)

    # first token index & validity mask
    token_idx = chunk_start + offs_token
    valid_tok = token_idx < N
    first_token_idx = chunk_start
    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

    # Load first cluster id to initialise the running accumulator
    first_id = tl.load(cid_batch_base + first_token_idx)
    last_id = tl.load(cid_batch_base + last_token_idx)
    all_ids = tl.load(cid_batch_base + token_idx, mask=valid_tok, other=-1)

    all_tokens_idxs = tl.load(idx_batch_base + token_idx, mask=valid_tok, other=-1)  # [BLOCK_N]

    load_mask = all_tokens_idxs[:, None] * D + offs_dim[None, :]

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            cluster_feats = tl.load(x_batch_base + load_mask, mask=cluster_mask[:, None], other=0.0)  # [BLOCK_N, D]
            cluster_feats = cluster_feats.to(tl.float32)
            sum_feats = tl.sum(cluster_feats, axis=0)
            dest_ptr = sum_ptr + (b * K + cid) * D + offs_dim
            tl.atomic_add(dest_ptr, sum_feats)
            tl.atomic_add(count_ptr + b * K + cid, cluster_size)


def triton_centroid_update_sorted_euclid(
    x: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor, *, BLOCK_N: int = 256
):
    """Fast centroid update for *Euclidean* KMeans assuming cluster IDs are pre-sorted.

    Parameters
    ----------
    x : Tensor [B, N, D]
        Input feature vectors (no normalization assumed).
    cluster_ids : LongTensor [B, N]
        Cluster assignment for each point.
    old_centroids : Tensor [B, K, D]
        Previous centroids (used to fill empty clusters).
    BLOCK_N : int, optional
        Tokens per Triton program (affects occupancy/perf).
    """
    assert x.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA device"
    B, N, D = x.shape
    K = old_centroids.shape[1]

    # Batch-wise sort of cluster assignments
    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    centroid_cnts = torch.zeros((B, K), device=x.device, dtype=torch.int32)

    grid = (triton.cdiv(N, BLOCK_N), B)
    _centroid_update_chunk_kernel[grid](
        x,  # original features
        sorted_idx_int,  # gather indices
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_cnts,
        B,
        N,
        D,
        K,
        BLOCK_N=BLOCK_N,
    )

    # Convert sums to means; replace empty clusters with old centroids
    counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f
    empty_mask = (centroid_cnts == 0).unsqueeze(-1)
    centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
    return centroids.to(x.dtype), centroid_cnts


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


_TUNE_CONFIGS = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=4, num_warps=wp)
    for BN in [32, 64, 128]
    for BK in [32, 64, 128]
    for wp in [4, 8]
]


def _cfg_keep(conf):
    """Basic heuristic to prune unbalanced configs."""
    BN = conf.kwargs["BLOCK_N"]
    BK = conf.kwargs["BLOCK_K"]
    # Avoid tiny tiles on many warps
    if BN * BK < 32 * 32 and conf.num_warps > 4:
        return False
    return True


_TUNE_CONFIGS = list(filter(_cfg_keep, _TUNE_CONFIGS))


@triton.autotune(_TUNE_CONFIGS, key=["N", "K"])
@triton.jit
def _euclid_assign_kernel(
    x_ptr,  # *f16 / *f32 [B, N, D]
    c_ptr,  # *f16 / *f32 [B, K, D]
    x_sq_ptr,  # *f32         [B, N]
    out_ptr,  # *i32         [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Each program handles a tile of BLOCK_N points for a given batch element.

    The kernel iterates over the centroid dimension K in chunks of BLOCK_K and
    maintains the running minimum distance as well as the corresponding index
    for every point in the tile.
    """
    pid_n = tl.program_id(0)  # tile index along N dimension
    pid_b = tl.program_id(1)  # batch index

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # ------------------------------------------------------------------
    # Load x tile  (BLOCK_N, D)
    # ------------------------------------------------------------------
    offs_d = tl.arange(0, D)
    # Compute pointer for x block: base + b*stride_x_b + n*stride_x_n + d*stride_x_d
    x_ptrs = x_ptr + pid_b * stride_x_b + n_offsets[:, None] * stride_x_n + offs_d[None, :] * stride_x_d
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    x_tile = x_tile  # compute in f32

    # Pre-load x_sq for the tile  (BLOCK_N,)
    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    # Init best distance / index
    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)  # large number
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # ------------------------------------------------------------------
    # Iterate over centroids in chunks of BLOCK_K
    # ------------------------------------------------------------------
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load centroid tile  (D, BLOCK_K)
        c_ptrs = c_ptr + pid_b * stride_c_b + k_offsets[None, :] * stride_c_k + offs_d[:, None] * stride_c_d
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)
        c_tile = c_tile

        # Compute centroid squared norms (BLOCK_K,)
        cent_sq = tl.sum(c_tile * c_tile, axis=0).to(tl.float32)

        # Compute cross term (BLOCK_N, BLOCK_K) = x_tile @ c_tile
        cross = tl.dot(x_tile, c_tile).to(tl.float32)  # float32

        # Squared Euclidean distance
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)

        # Mask out invalid centroid columns before reduction
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


def euclid_assign_triton(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    out: torch.Tensor = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
) -> torch.Tensor:
    """Return nearest-centroid indices using Triton kernel.

    Args:
        x         : (B, N, D) float16 / float32 (on CUDA)
        centroids : (B, K, D) same dtype/device as x
        x_sq      : (B, N)    float32 – ||x||^2 per point (on CUDA)

    Returns:
        cluster_ids (B, N) int32 (callers can cast to int64 if desired)
    """
    assert x.is_cuda and centroids.is_cuda and x_sq.is_cuda, "All tensors must be on CUDA"
    # assert x.dtype in (torch.float16, torch.float32), "x must be fp16/fp32"
    assert centroids.dtype == x.dtype, "centroids dtype mismatch"

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D), "centroids shape mismatch"
    assert x_sq.shape == (B, N), "x_sq shape mismatch"

    # x = x.contiguous()
    # centroids = centroids.contiguous()
    # x_sq = x_sq.contiguous()

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int64)

    # Strides (in elements)
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    _euclid_assign_kernel[grid](
        x,
        centroids,
        x_sq,
        out,
        B,
        N,
        K,
        D,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_c_b,
        stride_c_k,
        stride_c_d,
        stride_xsq_b,
        stride_xsq_n,
        stride_out_b,
        stride_out_n,
    )
    return out


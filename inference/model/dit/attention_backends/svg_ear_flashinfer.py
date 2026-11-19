"""FlashInfer + Triton executor for the standalone SVG-EAR backend.

The selected Q/K cluster pairs are evaluated exactly by FlashInfer variable-
block attention.  Omitted video K clusters are represented by one K/V
centroid and merged into FlashInfer's output/LSE with a stable online softmax
Triton kernel.  This module owns its workspace and metadata; it neither imports
nor shares executor state with SVG2.

MagiHuman routes every Q head independently while using native GQA K/V heads.
FlashInfer requires the sparse topology to be shared inside one GQA group, so
the executor runs one Q lane from every native KV group at a time.  This avoids
materialising five full K/V copies for the usual 40Q/8KV layout.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

import torch


_WORKSPACE_BYTES = 128 * 1024 * 1024


@contextmanager
def _stage(profile, name: str):
    if profile is None:
        yield
    else:
        with profile(name):
            yield


def _centroid_merge_kernel():
    """Create the Triton kernel lazily so CPU-only reference use stays light."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        Q,
        K_CENTERS,
        V_CENTERS,
        ROUTE,
        K_COUNTS,
        Q_INDPTR,
        EXACT_OUT,
        EXACT_LSE_LOG2,
        OUT,
        stride_qh: tl.constexpr,
        stride_qs: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kk: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_rh: tl.constexpr,
        stride_rq: tl.constexpr,
        stride_rk: tl.constexpr,
        stride_ch: tl.constexpr,
        stride_ck: tl.constexpr,
        stride_ih: tl.constexpr,
        stride_iq: tl.constexpr,
        stride_lh: tl.constexpr,
        stride_ls: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_os: tl.constexpr,
        stride_od: tl.constexpr,
        Q_HEADS: tl.constexpr,
        KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        K_CLUSTERS: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        q_cluster = tl.program_id(0)
        q_head = tl.program_id(1)
        kv_head = q_head // (Q_HEADS // KV_HEADS)
        start = tl.load(Q_INDPTR + q_head * stride_ih + q_cluster * stride_iq)
        end = tl.load(Q_INDPTR + q_head * stride_ih + (q_cluster + 1) * stride_iq)
        dims = tl.arange(0, HEAD_DIM)

        for row_start in range(start, end, BLOCK_M):
            rows = row_start + tl.arange(0, BLOCK_M)
            row_mask = rows < end
            q_ptr = Q + q_head * stride_qh + rows[:, None] * stride_qs + dims[None] * stride_qd
            query = tl.load(q_ptr, mask=row_mask[:, None], other=0.0)

            # FlashInfer 0.7.x VSA exposes LSE in log2 space.  The exact output
            # is already normalized, so it is an online-softmax state with
            # denominator one at max=exact_lse.
            exact_lse = tl.load(
                EXACT_LSE_LOG2 + q_head * stride_lh + rows * stride_ls,
                mask=row_mask,
                other=-float("inf"),
            ) * 0.6931471805599453
            exact_ptr = (
                EXACT_OUT
                + q_head * stride_oh
                + rows[:, None] * stride_os
                + dims[None] * stride_od
            )
            accumulator = tl.load(exact_ptr, mask=row_mask[:, None], other=0.0).to(tl.float32)
            running_max = exact_lse
            denominator = tl.where(row_mask, 1.0, 0.0)

            for key_start in range(0, K_CLUSTERS, BLOCK_N):
                keys = key_start + tl.arange(0, BLOCK_N)
                key_mask = keys < K_CLUSTERS
                center_ptr = (
                    K_CENTERS
                    + kv_head * stride_kh
                    + keys[:, None] * stride_kk
                    + dims[None] * stride_kd
                )
                center_k = tl.load(center_ptr, mask=key_mask[:, None], other=0.0)
                logits = tl.dot(query, tl.trans(center_k)) * SCALE
                counts = tl.load(
                    K_COUNTS + kv_head * stride_ch + keys * stride_ck,
                    mask=key_mask,
                    other=0,
                )
                selected = tl.load(
                    ROUTE
                    + q_head * stride_rh
                    + q_cluster * stride_rq
                    + keys * stride_rk,
                    mask=key_mask,
                    other=1,
                )
                centroid_mask = (selected == 0) & (counts > 0) & key_mask
                logits += tl.log(counts[None, :].to(tl.float32))
                logits = tl.where(centroid_mask[None, :], logits, -float("inf"))

                tile_max = tl.max(logits, axis=1)
                next_max = tl.maximum(running_max, tile_max)
                old_scale = tl.exp(running_max - next_max)
                probability = tl.exp(logits - next_max[:, None])
                tile_sum = tl.sum(probability, axis=1)

                value_ptr = (
                    V_CENTERS
                    + kv_head * stride_kh
                    + keys[:, None] * stride_kk
                    + dims[None] * stride_kd
                )
                center_v = tl.load(value_ptr, mask=key_mask[:, None], other=0.0)
                accumulator = (
                    accumulator * old_scale[:, None]
                    + tl.dot(probability.to(center_v.dtype), center_v).to(tl.float32)
                )
                denominator = denominator * old_scale + tile_sum
                running_max = next_max

            result = accumulator / denominator[:, None]
            out_ptr = OUT + q_head * stride_oh + rows[:, None] * stride_os + dims[None] * stride_od
            tl.store(out_ptr, result, mask=row_mask[:, None])

    return kernel


@dataclass
class _State:
    wrapper: object
    workspace: torch.Tensor
    q_sorted: torch.Tensor
    k_sorted: torch.Tensor
    v_sorted: torch.Tensor
    exact_out: torch.Tensor
    exact_lse: torch.Tensor
    merged_out: torch.Tensor
    result_by_head: torch.Tensor
    run_out: torch.Tensor
    run_lse: torch.Tensor
    q_sizes: torch.Tensor
    k_sizes: torch.Tensor
    block_map: torch.Tensor
    q_permutation: torch.Tensor
    k_permutation: torch.Tensor
    q_indptr: torch.Tensor
    has_anchors: bool


class SVGEARFlashInferExecutor:
    """Generation-local exact-block/centroid SVG-EAR executor."""

    def __init__(self, max_states: int | None = None):
        self.max_states = max_states if max_states is not None else int(
            os.environ.get("MAGI_SVG_EAR_FLASHINFER_CACHE_STATES", "2")
        )
        if self.max_states < 1:
            raise ValueError("MAGI_SVG_EAR_FLASHINFER_CACHE_STATES must be positive")
        self._states: OrderedDict[tuple, _State] = OrderedDict()
        self._kernel = None

    @staticmethod
    def _key(q, k, anchors: int, q_clusters: int, k_clusters: int):
        return (
            q.device.type,
            q.device.index,
            q.dtype,
            q.shape[1],
            q.shape[2],
            k.shape[2],
            q.shape[-1],
            anchors,
            q_clusters,
            k_clusters,
        )

    def _state(self, q, k, anchors: int, q_clusters: int, k_clusters: int):
        key = self._key(q, k, anchors, q_clusters, k_clusters)
        state = self._states.get(key)
        if state is not None:
            self._states.move_to_end(key)
            return state
        try:
            from flashinfer.sparse import VariableBlockSparseAttentionWrapper
        except ImportError as exc:
            raise RuntimeError(
                "SVG-EAR FlashInfer executor requires flashinfer-python"
            ) from exc

        _, sequence, q_heads, dim = q.shape
        kv_heads = k.shape[2]
        group = q_heads // kv_heads
        has_anchors = anchors > 0
        row_blocks = q_clusters + int(has_anchors)
        col_blocks = k_clusters + int(has_anchors)
        workspace = torch.empty(
            _WORKSPACE_BYTES, dtype=torch.uint8, device=q.device
        )
        state = _State(
            wrapper=VariableBlockSparseAttentionWrapper(workspace, backend="auto"),
            workspace=workspace,
            q_sorted=torch.empty((q_heads, sequence, dim), dtype=q.dtype, device=q.device),
            k_sorted=torch.empty((kv_heads, sequence, dim), dtype=k.dtype, device=k.device),
            v_sorted=torch.empty((kv_heads, sequence, dim), dtype=k.dtype, device=k.device),
            exact_out=torch.empty((q_heads, sequence, dim), dtype=q.dtype, device=q.device),
            exact_lse=torch.empty((q_heads, sequence), dtype=torch.float32, device=q.device),
            merged_out=torch.empty((q_heads, sequence, dim), dtype=q.dtype, device=q.device),
            result_by_head=torch.empty((q_heads, sequence, dim), dtype=q.dtype, device=q.device),
            run_out=torch.empty((kv_heads * sequence, 1, dim), dtype=q.dtype, device=q.device),
            run_lse=torch.empty((kv_heads * sequence, 1), dtype=torch.float32, device=q.device),
            q_sizes=torch.empty((kv_heads, row_blocks), dtype=torch.int32, device=q.device),
            k_sizes=torch.empty((kv_heads, col_blocks), dtype=torch.int32, device=q.device),
            block_map=torch.empty(
                (kv_heads, row_blocks, col_blocks), dtype=torch.bool, device=q.device
            ),
            q_permutation=torch.empty((q_heads, sequence), dtype=torch.long, device=q.device),
            k_permutation=torch.empty((kv_heads, sequence), dtype=torch.long, device=q.device),
            q_indptr=torch.empty((q_heads, q_clusters + 1), dtype=torch.int32, device=q.device),
            has_anchors=has_anchors,
        )
        if has_anchors:
            state.q_sizes[:, -1].fill_(anchors)
            state.k_sizes[:, -1].fill_(anchors)
            state.block_map[:, -1, :].fill_(True)
            state.block_map[:, :, -1].fill_(True)
        # Allocate for the largest native-GQA lane only; every lane has exactly
        # one Q head per KV head.
        if group < 1:
            raise ValueError("SVG-EAR FlashInfer requires valid GQA grouping")
        self._states[key] = state
        while len(self._states) > self.max_states:
            self._states.popitem(last=False)
        return state

    @staticmethod
    def _permute(source: torch.Tensor, order: torch.Tensor, out: torch.Tensor):
        torch.gather(
            source,
            1,
            order[..., None].expand(-1, -1, source.shape[-1]),
            out=out,
        )

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        video_idx: torch.Tensor,
        anchor_idx: torch.Tensor,
        q_labels: torch.Tensor,
        k_labels: torch.Tensor,
        q_counts: torch.Tensor,
        k_counts: torch.Tensor,
        k_centers: torch.Tensor,
        v_centers: torch.Tensor,
        route: torch.Tensor,
        *,
        profile=None,
    ) -> torch.Tensor:
        if not q.is_cuda:
            raise RuntimeError("SVG-EAR FlashInfer executor requires CUDA tensors")
        if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("SVG-EAR FlashInfer requires matching FP16/BF16 Q/K/V")
        if q.shape[-1] not in (16, 32, 64, 128):
            raise ValueError("SVG-EAR Triton merge supports head dimensions 16/32/64/128")
        q_heads, kv_heads = q.shape[2], k.shape[2]
        if q_heads % kv_heads:
            raise ValueError("SVG-EAR FlashInfer requires divisible GQA heads")
        q_clusters, k_clusters = route.shape[-2:]
        state = self._state(q, k, anchor_idx.numel(), q_clusters, k_clusters)
        video_tokens = video_idx.numel()

        with _stage(profile, "flashinfer_pack"):
            q_order = torch.argsort(q_labels, dim=1, stable=True)
            k_order = torch.argsort(k_labels, dim=1, stable=True)
            torch.gather(
                video_idx.expand(q_heads, -1),
                1,
                q_order,
                out=state.q_permutation[:, :video_tokens],
            )
            torch.gather(
                video_idx.expand(kv_heads, -1),
                1,
                k_order,
                out=state.k_permutation[:, :video_tokens],
            )
            if anchor_idx.numel():
                state.q_permutation[:, video_tokens:].copy_(anchor_idx.expand(q_heads, -1))
                state.k_permutation[:, video_tokens:].copy_(anchor_idx.expand(kv_heads, -1))
            self._permute(q[0].transpose(0, 1), state.q_permutation, state.q_sorted)
            self._permute(k[0].transpose(0, 1), state.k_permutation, state.k_sorted)
            self._permute(v[0].transpose(0, 1), state.k_permutation, state.v_sorted)

        with _stage(profile, "flashinfer_exact"):
            state.k_sizes[:, :k_clusters].copy_(k_counts)
            group = q_heads // kv_heads
            for lane in range(group):
                q_head_slice = slice(lane, q_heads, group)
                state.q_sizes[:, :q_clusters].copy_(q_counts[q_head_slice])
                state.block_map[:, :q_clusters, :k_clusters].copy_(route[q_head_slice])
                state.wrapper.plan(
                    block_mask_map=state.block_map,
                    block_row_sz=state.q_sizes,
                    block_col_sz=state.k_sizes,
                    num_qo_heads=kv_heads,
                    num_kv_heads=kv_heads,
                    head_dim=q.shape[-1],
                    q_data_type=q.dtype,
                    kv_data_type=k.dtype,
                )
                exact, lse = state.wrapper.run(
                    state.q_sorted[q_head_slice],
                    state.k_sorted,
                    state.v_sorted,
                    out=state.run_out,
                    lse=state.run_lse,
                    return_lse=True,
                )
                state.exact_out[q_head_slice].copy_(exact)
                state.exact_lse[q_head_slice].copy_(lse)

        with _stage(profile, "centroid_merge"):
            state.q_indptr[:, 0].zero_()
            torch.cumsum(q_counts, dim=-1, dtype=torch.int32, out=state.q_indptr[:, 1:])
            state.merged_out.copy_(state.exact_out)
            if self._kernel is None:
                self._kernel = _centroid_merge_kernel()
            self._kernel[(q_clusters, q_heads)](
                state.q_sorted,
                k_centers,
                v_centers,
                route,
                k_counts,
                state.q_indptr,
                state.exact_out,
                state.exact_lse,
                state.merged_out,
                state.q_sorted.stride(0),
                state.q_sorted.stride(1),
                state.q_sorted.stride(2),
                k_centers.stride(0),
                k_centers.stride(1),
                k_centers.stride(2),
                route.stride(0),
                route.stride(1),
                route.stride(2),
                k_counts.stride(0),
                k_counts.stride(1),
                state.q_indptr.stride(0),
                state.q_indptr.stride(1),
                state.exact_lse.stride(0),
                state.exact_lse.stride(1),
                state.exact_out.stride(0),
                state.exact_out.stride(1),
                state.exact_out.stride(2),
                Q_HEADS=q_heads,
                KV_HEADS=kv_heads,
                HEAD_DIM=q.shape[-1],
                K_CLUSTERS=k_clusters,
                SCALE=q.shape[-1] ** -0.5,
                BLOCK_M=16,
                BLOCK_N=64,
                num_warps=4,
                num_stages=2,
            )

        with _stage(profile, "flashinfer_scatter"):
            state.result_by_head.scatter_(
                1,
                state.q_permutation[..., None].expand(-1, -1, q.shape[-1]),
                state.merged_out,
            )
            # The returned model tensor must not alias persistent storage that
            # is overwritten by the next attention layer.
            return state.result_by_head.transpose(0, 1).unsqueeze(0).contiguous()


__all__ = ["SVGEARFlashInferExecutor"]

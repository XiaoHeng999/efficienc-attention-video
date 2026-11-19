"""FlashInfer variable-block executor for SVG2 inference.

The sparsity map is dynamic, so FlashInfer must still be planned for every
attention call.  This module deliberately keeps everything else persistent:
the wrapper, workspaces, Q/K/V packing buffers, output buffer and the static
parts of variable-block metadata.  A session owns the cache, avoiding global
state across concurrent generations.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass

import torch

from .kernels.permutation import permute_tokens
from .planning import plan_variable_blocks
from .profile import stage


_WORKSPACE_BYTES = 128 * 1024 * 1024


@dataclass
class _FlashInferState:
    wrapper: object
    workspace: torch.Tensor
    q_sorted: torch.Tensor
    k_for_q: torch.Tensor
    v_for_q: torch.Tensor
    # FlashInfer's public run API takes this packed NHD output layout.
    run_out: torch.Tensor
    row_sizes: torch.Tensor
    col_sizes: torch.Tensor
    block_map: torch.Tensor
    kv_for_q: torch.Tensor
    q_permutation: torch.Tensor
    k_permutation: torch.Tensor
    has_anchor: bool


class FlashInferSVG2Executor:
    """Session-local persistent storage for dynamic FlashInfer SVG2 calls.

    A state is shape-specialized, not sparsity-specialized.  ``plan`` is still
    invoked each time because labels and top-p routing may change, while every
    allocation whose shape does not depend on that route is reused.
    """

    def __init__(self, max_states: int | None = None):
        # A state owns roughly 128 MiB workspace plus packed GQA buffers.  A
        # bounded LRU preserves the usual cond/uncond shapes without allowing
        # a long variable-resolution process to retain every old shape.
        self.max_states = max_states if max_states is not None else int(
            os.environ.get("MAGI_SVG2_FLASHINFER_CACHE_STATES", "2")
        )
        if self.max_states < 1:
            raise ValueError("MAGI_SVG2_FLASHINFER_CACHE_STATES must be positive")
        self._states: OrderedDict[tuple, _FlashInferState] = OrderedDict()

    @staticmethod
    def _key(q, k, anchor_count, q_clusters, k_clusters):
        return (
            q.device.type, q.device.index, q.dtype,
            q.shape[1], q.shape[2], k.shape[2], q.shape[-1],
            anchor_count, q_clusters, k_clusters,
        )

    def _get_state(self, q, k, anchor_count, q_clusters, k_clusters):
        key = self._key(q, k, anchor_count, q_clusters, k_clusters)
        state = self._states.get(key)
        if state is not None:
            self._states.move_to_end(key)
            return state

        try:
            from flashinfer.sparse import VariableBlockSparseAttentionWrapper
        except ImportError as exc:
            raise RuntimeError(
                "SVG2 FlashInfer executor requires flashinfer-python in the active environment"
            ) from exc

        _, sequence_length, q_heads, head_dim = q.shape
        kv_heads = k.shape[2]
        if q_heads % kv_heads:
            raise ValueError("SVG2 FlashInfer executor requires divisible GQA heads")
        device, dtype = q.device, q.dtype
        has_anchor = bool(anchor_count)
        workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        wrapper = VariableBlockSparseAttentionWrapper(workspace, backend="auto")
        row_blocks = q_clusters + int(has_anchor)
        col_blocks = k_clusters + int(has_anchor)
        q_group = q_heads // kv_heads
        state = _FlashInferState(
            wrapper=wrapper,
            workspace=workspace,
            q_sorted=torch.empty((q_heads, sequence_length, head_dim), dtype=dtype, device=device),
            k_for_q=torch.empty((q_heads, sequence_length, head_dim), dtype=dtype, device=device),
            v_for_q=torch.empty((q_heads, sequence_length, head_dim), dtype=dtype, device=device),
            run_out=torch.empty((q_heads * sequence_length, 1, head_dim), dtype=dtype, device=device),
            row_sizes=torch.empty((q_heads, row_blocks), dtype=torch.int32, device=device),
            col_sizes=torch.empty((q_heads, col_blocks), dtype=torch.int32, device=device),
            block_map=torch.empty((q_heads, row_blocks, col_blocks), dtype=torch.bool, device=device),
            kv_for_q=torch.arange(q_heads, device=device, dtype=torch.long) // q_group,
            q_permutation=torch.empty((q_heads, sequence_length), dtype=torch.long, device=device),
            k_permutation=torch.empty((kv_heads, sequence_length), dtype=torch.long, device=device),
            has_anchor=has_anchor,
        )
        # Anchor-to-anything blocks are static and always dense.  Dynamic video
        # routing overwrites only the top-left SVG2 region below.
        if has_anchor:
            state.block_map.fill_(True)
            state.row_sizes[:, -1].fill_(anchor_count)
            state.col_sizes[:, -1].fill_(anchor_count)
        self._states[key] = state
        while len(self._states) > self.max_states:
            self._states.popitem(last=False)
        return state

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        video_idx: torch.Tensor,
        anchor_idx: torch.Tensor,
        q_order: torch.Tensor,
        k_order: torch.Tensor,
        q_counts: torch.Tensor,
        k_counts: torch.Tensor,
        cluster_map: torch.Tensor,
        *,
        profile=None,
    ) -> torch.Tensor:
        if not q.is_cuda:
            raise RuntimeError("The SVG2 FlashInfer executor requires CUDA tensors")
        if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("The SVG2 FlashInfer executor requires matching FP16/BF16 Q/K/V")

        q_heads, kv_heads = q.shape[2], k.shape[2]
        q_clusters, k_clusters = cluster_map.shape[-2:]
        state = self._get_state(q, k, anchor_idx.numel(), q_clusters, k_clusters)
        with stage(profile, "flashinfer_pack"):
            # These permutations are genuinely dynamic.  Their destination
            # storage is not, and is reused from the session-local state.
            video_length = video_idx.numel()
            torch.gather(
                video_idx.expand(q_heads, -1), 1, q_order,
                out=state.q_permutation[:, :video_length],
            )
            torch.gather(
                video_idx.expand(kv_heads, -1), 1, k_order,
                out=state.k_permutation[:, :video_length],
            )
            if anchor_idx.numel():
                state.q_permutation[:, video_length:].copy_(anchor_idx.expand(q_heads, -1))
                state.k_permutation[:, video_length:].copy_(anchor_idx.expand(kv_heads, -1))
            q_by_head = q[0].transpose(0, 1)
            k_by_head = k[0].transpose(0, 1)
            v_by_head = v[0].transpose(0, 1)
            permute_tokens(q_by_head, state.q_permutation, out=state.q_sorted)
            # Present one KV copy per Q head: the public variable-block API
            # indexes a sparse pattern by KV head, whereas SVG2 routes are
            # distinct for every Q head inside a GQA group.
            repeat = q_heads // kv_heads
            permute_tokens(k_by_head, state.k_permutation, repeat=repeat, out=state.k_for_q)
            permute_tokens(v_by_head, state.k_permutation, repeat=repeat, out=state.v_for_q)

        with stage(profile, "flashinfer_metadata"):
            state.row_sizes[:, :q_clusters].copy_(q_counts)
            # Use the already-allocated metadata region rather than materialize
            # k_counts.index_select(...) and concatenated anchor columns.
            torch.index_select(k_counts, 0, state.kv_for_q, out=state.col_sizes[:, :k_clusters])
            state.block_map[:, :q_clusters, :k_clusters].copy_(cluster_map)

        with stage(profile, "flashinfer_plan"):
            plan_variable_blocks(
                state.wrapper,
                block_mask_map=state.block_map,
                block_row_sz=state.row_sizes,
                block_col_sz=state.col_sizes,
                num_qo_heads=q_heads,
                num_kv_heads=q_heads,
                head_dim=q.shape[-1],
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
            )
        with stage(profile, "flashinfer_attention"):
            output_sorted = state.wrapper.run(
                state.q_sorted, state.k_for_q, state.v_for_q, out=state.run_out
            )
        with stage(profile, "flashinfer_scatter"):
            # Do not reuse the returned tensor: it becomes the model-visible
            # layer output and must remain valid after the next SVG2 call.
            output = permute_tokens(output_sorted, state.q_permutation, inverse=True)
            output = output.transpose(0, 1).unsqueeze(0).contiguous()
        return output


def svg2_flashinfer_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    video_idx: torch.Tensor,
    anchor_idx: torch.Tensor,
    q_order: torch.Tensor,
    k_order: torch.Tensor,
    q_counts: torch.Tensor,
    k_counts: torch.Tensor,
    cluster_map: torch.Tensor,
    *,
    executor: FlashInferSVG2Executor | None = None,
    profile=None,
) -> torch.Tensor:
    """Run exact variable-cluster attention over sorted multimodal Q/K/V.

    Direct callers get a correct one-shot executor.  Real MagiHuman inference
    passes a session-owned executor so allocations persist across layers and
    denoising steps.
    """
    executor = executor or FlashInferSVG2Executor()
    return executor.run(
        q, k, v, video_idx, anchor_idx, q_order, k_order,
        q_counts, k_counts, cluster_map, profile=profile,
    )

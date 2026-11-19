"""Standalone FlashInfer variable-block executor for SparsePR."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


_WORKSPACE_BYTES = 128 * 1024 * 1024


@dataclass
class _State:
    wrapper: object
    workspace: torch.Tensor
    q_sorted: torch.Tensor
    k_sorted: torch.Tensor
    v_sorted: torch.Tensor
    run_out: torch.Tensor
    q_sizes: torch.Tensor
    k_sizes: torch.Tensor
    block_map: torch.Tensor


class SparsePRFlashInferExecutor:
    """Session-local executor; no workspace or plan is shared with other backends."""

    def __init__(self, max_states: int = 2):
        if max_states < 1:
            raise ValueError("SparsePR FlashInfer max_states must be positive")
        self.max_states = max_states
        self._states: OrderedDict[tuple, _State] = OrderedDict()

    @staticmethod
    def _key(q, k, anchors, q_clusters, k_clusters):
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

    def _state(self, q, k, anchors, q_clusters, k_clusters):
        key = self._key(q, k, anchors, q_clusters, k_clusters)
        state = self._states.get(key)
        if state is not None:
            self._states.move_to_end(key)
            return state
        try:
            from flashinfer.sparse import VariableBlockSparseAttentionWrapper
        except ImportError as exc:
            raise RuntimeError(
                "SparsePR requires flashinfer-python for its CUDA executor"
            ) from exc
        _, length, q_heads, dim = q.shape
        kv_heads = k.shape[2]
        workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device)
        state = _State(
            wrapper=VariableBlockSparseAttentionWrapper(workspace, backend="auto"),
            workspace=workspace,
            q_sorted=torch.empty((q_heads, length, dim), dtype=q.dtype, device=q.device),
            k_sorted=torch.empty((kv_heads, length, dim), dtype=k.dtype, device=k.device),
            v_sorted=torch.empty((kv_heads, length, dim), dtype=k.dtype, device=k.device),
            run_out=torch.empty((kv_heads * length, 1, dim), dtype=q.dtype, device=q.device),
            q_sizes=torch.empty((kv_heads, q_clusters + 1), dtype=torch.int32, device=q.device),
            k_sizes=torch.empty((kv_heads, k_clusters + 1), dtype=torch.int32, device=q.device),
            block_map=torch.empty(
                (kv_heads, q_clusters + 1, k_clusters + 1), dtype=torch.bool, device=q.device
            ),
        )
        self._states[key] = state
        while len(self._states) > self.max_states:
            self._states.popitem(last=False)
        return state

    @staticmethod
    def _permute(source: torch.Tensor, order: torch.Tensor, out: torch.Tensor):
        torch.gather(source, 1, order[..., None].expand(-1, -1, source.shape[-1]), out=out)

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
        cluster_map: torch.Tensor,
    ) -> torch.Tensor:
        if not q.is_cuda:
            raise RuntimeError("SparsePR FlashInfer executor requires CUDA tensors")
        if (
            q.dtype not in (torch.float16, torch.bfloat16)
            or k.dtype != q.dtype
            or v.dtype != q.dtype
        ):
            raise ValueError("SparsePR FlashInfer requires matching FP16/BF16 Q/K/V")
        q_heads, kv_heads = q.shape[2], k.shape[2]
        if q_heads % kv_heads:
            raise ValueError("SparsePR requires divisible GQA heads")
        q_clusters, k_clusters = cluster_map.shape[-2:]
        state = self._state(q, k, anchor_idx.numel(), q_clusters, k_clusters)
        q_order = torch.argsort(q_labels, dim=1, stable=True)
        k_order = torch.argsort(k_labels, dim=1, stable=True)
        q_permutation = torch.cat((video_idx[q_order], anchor_idx.expand(q_heads, -1)), dim=1)
        k_permutation_native = torch.cat(
            (video_idx[k_order], anchor_idx.expand(kv_heads, -1)), dim=1
        )
        group = q_heads // kv_heads

        q_by_head = q[0].transpose(0, 1)
        k_by_head = k[0].transpose(0, 1)
        v_by_head = v[0].transpose(0, 1)
        self._permute(q_by_head, q_permutation, state.q_sorted)
        self._permute(k_by_head, k_permutation_native, state.k_sorted)
        self._permute(v_by_head, k_permutation_native, state.v_sorted)

        state.k_sizes[:, :k_clusters].copy_(k_counts)
        state.k_sizes[:, -1].fill_(anchor_idx.numel())
        state.block_map[:, -1, :].fill_(True)
        state.block_map[:, :, -1].fill_(True)

        # FlashInfer native GQA requires every Q head in a KV group to share
        # one sparse topology. SparsePR deliberately routes each Q head
        # independently, so process one Q lane from every KV group at a time.
        # This preserves the paper semantics while keeping K/V at native heads
        # instead of materializing an 8->40 expansion.
        output = torch.empty_like(state.q_sorted)
        for lane in range(group):
            q_head_slice = slice(lane, q_heads, group)
            state.q_sizes[:, :q_clusters].copy_(q_counts[q_head_slice])
            state.q_sizes[:, -1].fill_(anchor_idx.numel())
            state.block_map[:, :q_clusters, :k_clusters].copy_(
                cluster_map[q_head_slice]
            )
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
            output_sorted = state.wrapper.run(
                state.q_sorted[q_head_slice],
                state.k_sorted,
                state.v_sorted,
                out=state.run_out,
            ).view(kv_heads, q.shape[1], q.shape[-1])
            output[q_head_slice].scatter_(
                1,
                q_permutation[q_head_slice, :, None].expand(
                    -1, -1, q.shape[-1]
                ),
                output_sorted,
            )
        return output.transpose(0, 1).unsqueeze(0).contiguous()


__all__ = ["SparsePRFlashInferExecutor"]

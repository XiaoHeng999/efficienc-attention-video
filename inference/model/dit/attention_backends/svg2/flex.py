"""FlexAttention executor for SVG2's semantic cluster map.

The algorithm owns clustering and top-p selection. This executor maps its
variable cluster boundaries to fixed FlexAttention tiles, marking only border
tiles as partial. A future CuTe/FlashInfer executor can consume the same map.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from .profile import stage
from .kernels.permutation import permute_tokens


@lru_cache(maxsize=1)
def _compiled_flex_attention():
    return torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def _tile_membership(labels: torch.Tensor, cluster_count: int, block_size: int):
    """Return cluster presence and endpoint labels for each fixed-size tile."""
    length = labels.shape[1]
    starts = torch.arange(0, length, block_size, device=labels.device)
    ends = (starts + block_size - 1).clamp(max=length - 1)
    first, last = labels[:, starts], labels[:, ends]
    clusters = torch.arange(cluster_count, device=labels.device)
    membership = (clusters[None, None, :] >= first[..., None]) & (
        clusters[None, None, :] <= last[..., None]
    )
    return membership, first, last


def build_svg2_block_mask(
    q_labels: torch.Tensor,
    k_labels: torch.Tensor,
    cluster_map: torch.Tensor,
    block_size: int,
) -> BlockMask:
    """Build sparse tile indices on-device; use exact masks on boundary tiles.

    q_labels/k_labels are the sorted, full-sequence labels. The last label value
    is reserved for audio/text anchors, which always attend to every token.
    """
    q_heads, length = q_labels.shape
    kv_heads = k_labels.shape[0]
    q_clusters, k_clusters = cluster_map.shape[-2:]
    group = q_heads // kv_heads
    kv_for_q = torch.arange(q_heads, device=q_labels.device) // group
    q_members, q_first, q_last = _tile_membership(q_labels, q_clusters, block_size)
    k_members, k_first, k_last = _tile_membership(k_labels, k_clusters, block_size)
    k_members = k_members[kv_for_q]
    k_first, k_last = k_first[kv_for_q], k_last[kv_for_q]

    # A tile is present when any pair of clusters in its Q/K spans is selected.
    selected_k = torch.matmul(q_members.float(), cluster_map.float())
    active = torch.einsum("hmc,hnc->hmn", selected_k, k_members.float()) > 0
    q_has_anchor = q_last == q_clusters
    k_has_anchor = k_last == k_clusters
    active |= q_has_anchor[:, :, None] | k_has_anchor[:, None, :]

    # Pure selected cluster pairs and pure anchor tiles need no token mask.
    q_pure_video = (q_first == q_last) & (q_first < q_clusters)
    k_pure_video = (k_first == k_last) & (k_first < k_clusters)
    head = torch.arange(q_heads, device=q_labels.device)[:, None, None]
    q_cluster = q_first.clamp(max=q_clusters - 1)[:, :, None]
    k_cluster = k_first.clamp(max=k_clusters - 1)[:, None, :]
    selected_pure_pair = cluster_map[head, q_cluster, k_cluster]
    full = (q_first == q_clusters)[:, :, None] | (k_first == k_clusters)[:, None, :]
    full |= q_pure_video[:, :, None] & k_pure_video[:, None, :] & selected_pure_pair
    full &= active
    partial = active & ~full

    # The last index dimension is fixed-capacity; each count selects its prefix.
    partial_counts = partial.sum(-1, dtype=torch.int32).unsqueeze(0)
    partial_indices = torch.argsort(~partial, dim=-1, stable=True).to(torch.int32).unsqueeze(0)
    full_counts = full.sum(-1, dtype=torch.int32).unsqueeze(0)
    full_indices = torch.argsort(~full, dim=-1, stable=True).to(torch.int32).unsqueeze(0)

    # FlexAttention may call mask_mod for padded edge positions.
    padded_length = ((length + block_size - 1) // block_size) * block_size
    q_padded = torch.nn.functional.pad(q_labels, (0, padded_length - length), value=q_clusters)
    k_padded = torch.nn.functional.pad(k_labels, (0, padded_length - length), value=k_clusters)

    def mask_mod(batch, head_idx, q_idx, k_idx):
        q_cluster_idx = q_padded[head_idx, q_idx]
        kv_head_idx = head_idx // group
        k_cluster_idx = k_padded[kv_head_idx, k_idx]
        selected = cluster_map[
            head_idx,
            q_cluster_idx.clamp(max=q_clusters - 1),
            k_cluster_idx.clamp(max=k_clusters - 1),
        ]
        return ((q_cluster_idx == q_clusters) | (k_cluster_idx == k_clusters) | selected) & (
            (q_idx < length) & (k_idx < length)
        )

    return BlockMask.from_kv_blocks(
        partial_counts,
        partial_indices,
        full_kv_num_blocks=full_counts,
        full_kv_indices=full_indices,
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=mask_mod,
        seq_lengths=(length, length),
        compute_q_blocks=False,
    )


def svg2_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    video_idx: torch.Tensor,
    anchor_idx: torch.Tensor,
    q_order: torch.Tensor,
    q_labels: torch.Tensor,
    k_order: torch.Tensor,
    k_labels: torch.Tensor,
    cluster_map: torch.Tensor,
    block_size: int,
    *,
    profile=None,
) -> torch.Tensor:
    """One FlexAttention call for all Q heads, with GQA and exact sparse edges."""
    q_heads, kv_heads = q.shape[2], k.shape[2]
    with stage(profile, "flex_pack"):
        q_by_head = q[0].transpose(0, 1)
        k_by_head = k[0].transpose(0, 1)
        v_by_head = v[0].transpose(0, 1)
        q_permutation = torch.cat(
            [video_idx[q_order], anchor_idx.expand(q_heads, -1)], dim=1
        )
        k_permutation = torch.cat(
            [video_idx[k_order], anchor_idx.expand(kv_heads, -1)], dim=1
        )
        q_permuted = permute_tokens(q_by_head, q_permutation).unsqueeze(0)
        k_permuted = permute_tokens(k_by_head, k_permutation).unsqueeze(0)
        v_permuted = permute_tokens(v_by_head, k_permutation).unsqueeze(0)

        q_cluster_count, k_cluster_count = cluster_map.shape[-2:]
        q_sorted_labels = q_labels.gather(1, q_order)
        k_sorted_labels = k_labels.gather(1, k_order)
        q_full_labels = torch.cat(
            [q_sorted_labels, q_sorted_labels.new_full((q_heads, anchor_idx.numel()), q_cluster_count)], dim=1
        )
        k_full_labels = torch.cat(
            [k_sorted_labels, k_sorted_labels.new_full((kv_heads, anchor_idx.numel()), k_cluster_count)], dim=1
        )
    with stage(profile, "flex_block_mask"):
        block_mask = build_svg2_block_mask(q_full_labels, k_full_labels, cluster_map, block_size)
    with stage(profile, "flex_attention"):
        attention = _compiled_flex_attention() if q.is_cuda else flex_attention
        out_permuted = attention(
            q_permuted, k_permuted, v_permuted,
            block_mask=block_mask, enable_gqa=q_heads != kv_heads,
            kernel_options={"BACKEND": "TRITON"} if q.is_cuda else None,
        )
    with stage(profile, "flex_scatter"):
        out = permute_tokens(out_permuted[0], q_permutation, inverse=True)
        out = out.transpose(0, 1).unsqueeze(0).contiguous()
    return out

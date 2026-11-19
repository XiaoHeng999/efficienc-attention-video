"""Thin context-parallel adapter around the AB video backend."""

from __future__ import annotations

from typing import List

import torch
from einops import rearrange
from magi_compiler.api import magi_register_custom_op

from .core import scheduled_ab_attention


def _infer_output(q: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return torch.empty_like(q, dtype=torch.bfloat16).squeeze(0)


@magi_register_custom_op(
    name="infra::ab_attention_with_cp",
    mutates_args=(),
    infer_output_meta_fn=_infer_output,
    is_subgraph_boundary=True,
)
def ab_attention_with_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_split_sizes: List[int],
    modality: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    from inference.infra.distributed import get_cp_group, get_cp_world_size
    from inference.infra.parallelism.all_to_all_primitive import (
        batch_scatter_head_gather_seqlen, scatter_seqlen_gather_head,
    )
    from inference.model.dit.dit_module import _flash_attn_forward

    q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
    if get_cp_world_size() > 1:
        q, k, v = batch_scatter_head_gather_seqlen(
            [q.squeeze(0), k.squeeze(0), v.squeeze(0)], cp_split_sizes, get_cp_group()
        )
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    out = scheduled_ab_attention(q, k, v, modality, _flash_attn_forward, layer_idx=layer_idx).squeeze(0)
    if get_cp_world_size() > 1:
        out = scatter_seqlen_gather_head(out, cp_split_sizes, get_cp_group(), async_op=False)
        out = rearrange(out, "(cp sq) hn hd -> sq (cp hn) hd", cp=get_cp_world_size())
    return out

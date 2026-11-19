"""Slow reference for numerical checks of SIP/GQA/mixed modalities."""

from __future__ import annotations

import math

import torch


def weighted_attention_reference(q, k, v, video_idx, anchor_idx, plan):
    """Input [1,S,H,D]; returns full-sequence output with exact anchor rows."""
    if q.shape[0] != 1 or q.shape[2] % k.shape[2]:
        raise ValueError("Expected batch one and divisible GQA heads")
    _, _, q_heads, dim = q.shape
    kv_heads = k.shape[2]
    output = torch.empty_like(q)
    video = video_idx.tolist()
    anchors = anchor_idx.tolist()
    for head in range(q_heads):
        kh = head // (q_heads // kv_heads)
        for qb in range(plan.indices.shape[1]):
            query_ids = video[qb * 128:(qb + 1) * 128]
            if not query_ids:
                continue
            small = plan.indices[head, qb].tolist()
            key_ids = [video[block * 4 + offset] for block in small for offset in range(4)] + anchors
            shift = plan.log_weights[head, qb].repeat_interleave(4).to(q.dtype)
            shift = torch.cat((shift, torch.zeros(len(anchors), device=q.device, dtype=q.dtype)))
            logits = q[0, query_ids, head].float() @ k[0, key_ids, kh].float().T / math.sqrt(dim)
            logits += shift.float()
            output[0, query_ids, head] = (logits.softmax(-1) @ v[0, key_ids, kh].float()).to(q.dtype)
        if anchors:
            logits = q[0, anchors, head].float() @ k[0, :, kh].float().T / math.sqrt(dim)
            output[0, anchors, head] = (logits.softmax(-1) @ v[0, :, kh].float()).to(q.dtype)
    return output

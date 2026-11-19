"""Small PyTorch numerical reference for MagiHuman Sol-Attn tests."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def dense_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_heads, kv_heads = q.shape[2], k.shape[2]
    if q_heads % kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        enable_gqa=q_heads != kv_heads,
    ).transpose(1, 2)


def sol_attn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    video_tokens: int,
    tau: float = 1.0,
    block_size: int = 64,
) -> torch.Tensor:
    """Readable block approximation; intended for tests, not inference speed."""
    batch, q_tokens, q_heads, dim = q.shape
    kv_heads = k.shape[2]
    if batch != 1 or q_tokens != video_tokens or q_heads % kv_heads:
        raise ValueError("Reference expects B=1 video Q and divisible GQA heads")
    scale = dim**-0.5
    group = q_heads // kv_heads
    output = torch.empty_like(q)
    video_blocks = math.ceil(video_tokens / block_size)
    kv_blocks = math.ceil(k.shape[1] / block_size)
    for q_start in range(0, q_tokens, block_size):
        q_end = min(q_start + block_size, q_tokens)
        q_tile = q[:, q_start:q_end]
        for q_head in range(q_heads):
            kv_head = q_head // group
            qh = q_tile[0, :, q_head].float()
            k_centers = []
            for block in range(video_blocks):
                start, end = block * block_size, min((block + 1) * block_size, video_tokens)
                k_centers.append(k[0, start:end, kv_head].float().mean(dim=0))
            k_centers = torch.stack(k_centers)
            route_scores = qh.mean(dim=0) @ k_centers.T * scale
            threshold = route_scores.mean() + float(tau) * route_scores.std(correction=0)
            numerator = torch.zeros((qh.shape[0], dim), device=q.device, dtype=torch.float32)
            denominator = torch.zeros(qh.shape[0], device=q.device, dtype=torch.float32)
            block_logits: list[torch.Tensor] = []
            block_payloads: list[tuple[torch.Tensor, torch.Tensor, bool]] = []
            for block in range(kv_blocks):
                start, end = block * block_size, min((block + 1) * block_size, k.shape[1])
                kb = k[0, start:end, kv_head].float()
                vb = v[0, start:end, kv_head].float()
                exact = block >= video_blocks or (
                    block < video_blocks and route_scores[block] >= threshold
                )
                logits = qh @ kb.T * scale if exact else qh @ kb.mean(dim=0)[:, None] * scale
                block_logits.append(logits.amax(dim=-1))
                block_payloads.append((logits, vb, exact))
            row_max = torch.stack(block_logits).amax(dim=0)
            for logits, vb, exact in block_payloads:
                weights = torch.exp(logits - row_max[:, None])
                if exact:
                    numerator += weights @ vb
                    denominator += weights.sum(dim=-1)
                else:
                    numerator += weights * vb.sum(dim=0)
                    denominator += weights[:, 0] * vb.shape[0]
            output[0, q_start:q_end, q_head] = (
                numerator / denominator[:, None].clamp_min(1.0e-20)
            ).to(output.dtype)
    return output


__all__ = ["dense_gqa", "sol_attn_reference"]

"""SIP-weighted video attention in Triton, including exact non-video keys.

FlashInfer's ordinary paged API has no per-KV4 logit bias. The upstream AB
implementation supplies that bias with a C++/CUDA variant; this kernel keeps
the same weighted online-softmax semantics without any CUDA-language source.
"""

from __future__ import annotations

import math

import torch


def weighted_video_attention(q, k, v, video_idx, anchor_idx, plan, *, query_tile_rows=16):
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(Q, K, V, VIDEO, ANCHOR, IDS, LOGW, OUT,
               SQ: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
               NV: tl.constexpr, NA: tl.constexpr, NS: tl.constexpr,
               D: tl.constexpr, Q_BLOCKS: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
        sub = tl.program_id(0)
        qb = tl.program_id(1)
        head = tl.program_id(2)
        kh = head // (HQ // HK)
        qi = qb * 128 + sub * BM + tl.arange(0, BM)
        dd = tl.arange(0, D)
        q_global = tl.load(VIDEO + qi, qi < NV, other=0)
        q_tile = tl.load(Q + q_global[:, None] * HQ * D + head * D + dd[None, :],
                         (qi[:, None] < NV), other=0)
        max_logit = tl.full((BM,), float("-inf"), tl.float32)
        denom = tl.zeros((BM,), tl.float32)
        numerator = tl.zeros((BM, D), tl.float32)
        row = head * Q_BLOCKS + qb
        lane = tl.arange(0, BN)
        for start in range(tl.cdiv(NS * 4, BN)):
            slot = start * BN + lane
            small = slot // 4
            block = tl.load(IDS + row * NS + small, small < NS, other=0)
            shift = tl.load(LOGW + row * NS + small, small < NS, other=0)
            vi = block * 4 + slot % 4
            kg = tl.load(VIDEO + vi, (slot < NS * 4) & (vi < NV), other=0)
            kk = tl.load(K + kg[:, None] * HK * D + kh * D + dd[None, :],
                         slot[:, None] < NS * 4, other=0)
            vv = tl.load(V + kg[:, None] * HK * D + kh * D + dd[None, :],
                         slot[:, None] < NS * 4, other=0)
            logits = tl.dot(q_tile, tl.trans(kk)).to(tl.float32) * 0.08838834764831845
            logits = logits + shift[None, :]
            logits = tl.where(slot[None, :] < NS * 4, logits, float("-inf"))
            new_max = tl.maximum(max_logit, tl.max(logits, 1))
            old_scale = tl.exp(max_logit - new_max)
            prob = tl.exp(logits - new_max[:, None])
            numerator = numerator * old_scale[:, None] + tl.dot(prob.to(vv.dtype), vv).to(tl.float32)
            denom = denom * old_scale + tl.sum(prob, 1)
            max_logit = new_max
        for start in range(tl.cdiv(NA, BN)):
            ai = start * BN + lane
            kg = tl.load(ANCHOR + ai, ai < NA, other=0)
            kk = tl.load(K + kg[:, None] * HK * D + kh * D + dd[None, :], ai[:, None] < NA, other=0)
            vv = tl.load(V + kg[:, None] * HK * D + kh * D + dd[None, :], ai[:, None] < NA, other=0)
            logits = tl.dot(q_tile, tl.trans(kk)).to(tl.float32) * 0.08838834764831845
            logits = tl.where(ai[None, :] < NA, logits, float("-inf"))
            new_max = tl.maximum(max_logit, tl.max(logits, 1))
            old_scale = tl.exp(max_logit - new_max)
            prob = tl.exp(logits - new_max[:, None])
            numerator = numerator * old_scale[:, None] + tl.dot(prob.to(vv.dtype), vv).to(tl.float32)
            denom = denom * old_scale + tl.sum(prob, 1)
            max_logit = new_max
        result = numerator / denom[:, None]
        tl.store(OUT + q_global[:, None] * HQ * D + head * D + dd[None, :],
                 result, qi[:, None] < NV)

    if q.shape[0] != 1 or q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise ValueError("Triton AB executor expects BF16 [1,S,H,128]")
    _, length, q_heads, dim = q.shape
    kv_heads = k.shape[2]
    q_blocks = math.ceil(video_idx.numel() / 128)
    samples = plan.indices.shape[-1]
    if query_tile_rows not in (16, 32, 64, 128):
        raise ValueError("query_tile_rows must divide the 128-query SIP tile")
    output = torch.empty_like(q)
    kernel[(128 // query_tile_rows, q_blocks, q_heads)](
        q, k, v, video_idx, anchor_idx, plan.indices, plan.log_weights, output,
        length, q_heads, kv_heads, video_idx.numel(), anchor_idx.numel(), samples,
        dim, q_blocks, query_tile_rows, 64,
        num_warps=8 if query_tile_rows >= 64 else 4,
    )
    return output

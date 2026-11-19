"""IBCS and SIP in PyTorch; no CUDA-language extension is required.

Reference: Xu et al., arXiv:2610.39663, equations 3--6. The selector is
deliberately separate from the attention executor so its cost is measurable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SamplePlan:
    indices: torch.Tensor  # [Hq, ceil(Nq/128), samples], KV4 indices
    log_weights: torch.Tensor  # natural log(1/pi), same shape


def _centroids(q: torch.Tensor, clusters: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """One farthest-first/Lloyd pass, independently per query tile and head."""
    heads, tokens, dim = q.shape
    blocks = math.ceil(tokens / 128)
    padded = F.pad(q.float(), (0, 0, 0, blocks * 128 - tokens))
    rows = padded.reshape(heads * blocks, 128, dim)
    valid = (torch.arange(blocks * 128, device=q.device) < tokens).reshape(1, blocks, 128)
    valid = valid.expand(heads, -1, -1).reshape(-1, 128)
    # The AB CUDA selector starts with the query farthest from the tile mean,
    # then uses farthest-first centers. Starting from row zero changes the
    # clusters and, consequently, the SIP distribution.
    mean = (rows * valid[..., None]).sum(1) / valid.sum(-1).clamp_min(1)[:, None]
    first_distance = ((rows - mean[:, None]) ** 2).sum(-1).masked_fill(~valid, -1)
    centers = [rows[torch.arange(rows.shape[0], device=q.device), first_distance.argmax(-1)]]
    minimum = torch.full((rows.shape[0], 128), float("inf"), device=q.device)
    for _ in range(1, clusters):
        distance = ((rows - centers[-1][:, None]) ** 2).sum(-1)
        minimum = torch.minimum(minimum, distance).masked_fill(~valid, -1)
        centers.append(rows[torch.arange(rows.shape[0], device=q.device), minimum.argmax(-1)])
    seed = torch.stack(centers, dim=1)
    labels = torch.cdist(rows, seed).argmin(-1)
    members = F.one_hot(labels, clusters).to(rows.dtype) * valid[..., None]
    count = members.sum(1)
    centroid = torch.bmm(members.transpose(1, 2), rows) / count.clamp_min(1)[..., None]
    centroid = torch.where(count[..., None] > 0, centroid, seed)
    if q.dtype == torch.bfloat16:
        centroid = centroid.to(torch.bfloat16).float()
    weights = count / valid.sum(-1).clamp_min(1)[:, None]
    return centroid.reshape(heads, blocks, clusters, dim), weights.reshape(heads, blocks, clusters)


def ibcs_scores(q: torch.Tensor, k: torch.Tensor, *, chunk_rows: int = 16) -> torch.Tensor:
    """Score video KV4 blocks for GQA Q heads; input [H,N,D]."""
    heads, _, dim = q.shape
    kv_heads, key_tokens, _ = k.shape
    if heads % kv_heads or key_tokens % 4:
        raise ValueError("IBCS needs divisible GQA heads and video length divisible by four")
    centroid, weights = _centroids(q)
    q_blocks = centroid.shape[1]
    pooled = k.float().reshape(kv_heads, key_tokens // 4, 2, 2, dim).mean(3)
    if k.dtype == torch.bfloat16:
        pooled = pooled.to(torch.bfloat16).float()
    pooled = pooled.reshape(kv_heads, -1, dim)
    rows = centroid.reshape(heads * q_blocks, 8, dim)
    row_weights = weights.reshape(heads * q_blocks, 8)
    result = torch.empty((heads * q_blocks, key_tokens // 4), device=q.device)
    rows_per_group = (heads // kv_heads) * q_blocks
    for kv_head in range(kv_heads):
        group_begin = kv_head * rows_per_group
        group_end = group_begin + rows_per_group
        key_matrix = pooled[kv_head].T.contiguous()
        for start in range(group_begin, group_end, chunk_rows):
            end = min(start + chunk_rows, group_end)
            logits = torch.matmul(rows[start:end], key_matrix) / math.sqrt(dim)
            mass = logits.softmax(-1).reshape(end - start, 8, key_tokens // 4, 2).sum(-1)
            result[start:end] = (mass * row_weights[start:end, :, None]).sum(1)
    return result.reshape(heads, q_blocks, key_tokens // 4)


def sip_sample(scores: torch.Tensor, samples: int, *, seed: int) -> SamplePlan:
    """Fixed-size systematic sample with positive inclusion probabilities."""
    if scores.ndim != 3 or not 0 < samples < scores.shape[-1]:
        raise ValueError("Sample count must be between zero and the number of KV4 blocks")
    shape = scores.shape
    rows, blocks = shape[0] * shape[1], shape[2]
    indices_out = torch.empty((rows, samples), device=scores.device, dtype=torch.int32)
    weights_out = torch.empty((rows, samples), device=scores.device, dtype=torch.float32)
    generator = torch.Generator(device=scores.device).manual_seed(seed)
    offsets = torch.rand((rows, 1), generator=generator, device=scores.device, dtype=torch.float64)
    positions = torch.arange(samples, device=scores.device, dtype=torch.float64)[None]
    flat = scores.reshape(rows, blocks)
    # Sorting solves the capped-proportional water filling exactly up to FP64
    # rounding, and chunking bounds scratch memory on long videos.
    for start in range(0, rows, 64):
        end = min(start + 64, rows)
        values = flat[start:end].to(torch.float64).clamp_min(1e-30)
        ordered = values.sort(dim=-1, descending=True).values
        # Reverse cumulative sum avoids cancelling a huge saturated score
        # against the total when the remaining tail is tiny.
        remaining = ordered.flip(-1).cumsum(-1).flip(-1).clamp_min(1e-300)
        counts = torch.arange(blocks, device=scores.device, dtype=torch.float64)[None]
        candidates = (samples - counts) / remaining
        first_valid = ((candidates * ordered <= 1) & (counts < samples)).to(torch.int32).argmax(-1)
        scale = candidates.gather(-1, first_valid[:, None])
        pi = (values * scale).clamp(min=1e-300, max=1)
        cdf = pi.cumsum(-1)
        threshold = positions + offsets[start:end]
        chosen = torch.searchsorted(cdf.contiguous(), threshold.contiguous()).clamp_max(blocks - 1)
        indices_out[start:end] = chosen.to(torch.int32)
        weights_out[start:end] = -pi.gather(-1, chosen).log().float()
    return SamplePlan(indices_out.reshape(*shape[:2], samples),
                      weights_out.reshape(*shape[:2], samples))

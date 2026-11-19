"""Research-only numerical model of reusable VC-Attention ideas.

This module is deliberately not connected to MagiHuman's inference path. It
keeps the useful, testable pieces of the experiment: matched Q/K Hadamard
rotation, exact K centering, E4M3 Q/K/P/V arithmetic, value clustering, and
blockwise value smoothing/mean recovery. Products are evaluated in PyTorch, so
this is an analysis aid rather than a performance backend.
"""

import math

import torch
import torch.nn.functional as F

BLOCK = 128
__all__ = ["value_permutation", "vc_attention_reference"]


def value_permutation(values: torch.Tensor, clusters: int = 32) -> torch.Tensor:
    """Group similar value tokens independently for each KV head.

    Args:
        values: Float tensor shaped ``[heads, sequence, channels]``.
        clusters: Maximum number of clusters per head. The effective count is
            capped by ``sequence // 128`` so clusters contain useful blocks.

    Returns:
        Stable gather indices shaped ``[heads, sequence]``.
    """
    if values.ndim != 3 or not values.is_floating_point():
        raise ValueError("values must be a floating-point [heads, sequence, channels] tensor")
    heads, sequence, dim = values.shape
    if sequence == 0 or clusters < 1:
        raise ValueError("sequence and clusters must be non-zero")
    count = min(clusters, max(1, sequence // BLOCK))
    data = values.float()
    samples = torch.linspace(0, sequence - 1, count, device=values.device).long()
    centers = data[:, samples].contiguous()
    for _ in range(2):
        offset = 0.5 * centers.square().sum(-1)[:, None, :]
        labels = (torch.bmm(data, centers.transpose(1, 2)) - offset).argmax(-1)
        sums = torch.zeros_like(centers)
        sums.scatter_add_(1, labels[:, :, None].expand(-1, -1, dim), data)
        counts = torch.zeros(heads, count, device=values.device, dtype=torch.float32)
        counts.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float32))
        centers = torch.where(
            counts[:, :, None] > 0,
            sums / counts.clamp_min(1)[:, :, None],
            centers,
        )
    offset = 0.5 * centers.square().sum(-1)[:, None, :]
    labels = (torch.bmm(data, centers.transpose(1, 2)) - offset).argmax(-1)
    return labels.argsort(dim=1, stable=True)


def _e4m3(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale


def _hadamard(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim < 1 or dim & (dim - 1):
        raise ValueError("Hadamard rotation requires a power-of-two head dimension")
    y, width = x.float(), 1
    while width < dim:
        shape = y.shape
        pair = y.reshape(*shape[:-1], -1, 2, width)
        lo, hi = pair[..., 0, :], pair[..., 1, :]
        y = torch.cat((lo + hi, lo - hi), -1).reshape(shape)
        width *= 2
    return y / math.sqrt(dim)


def _quantize_blocks(x: torch.Tensor, block: int = 64) -> torch.Tensor:
    heads, sequence, dim = x.shape
    padding = (-sequence) % block
    blocks = F.pad(x, (0, 0, 0, padding)).reshape(heads, -1, block, dim)
    scale = (blocks.abs().amax((2, 3), keepdim=True) / 448).clamp_min(1e-8)
    return _e4m3(blocks, scale).reshape(heads, -1, dim)[:, :sequence]


def _quantize_residual(v: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    heads, sequence, dim = v.shape
    padding = (-sequence) % BLOCK
    padded = F.pad(v, (0, 0, 0, padding)).reshape(heads, -1, BLOCK, dim)
    length = torch.full((padded.shape[1],), BLOCK, device=v.device, dtype=torch.float32)
    if padding:
        length[-1] -= padding
    mean = (padded.sum(2) / length[None, :, None]).to(torch.bfloat16).float()
    residual = padded - mean[:, :, None, :]
    if padding:
        residual[:, -1, -padding:] = 0
    if bits != 8:
        raise ValueError("Only E4M3 residual quantization is retained")
    scale = (residual.abs().amax(2, keepdim=True) / 448).clamp_min(1e-8)
    quantized = _e4m3(residual, scale)
    return quantized.reshape(heads, -1, dim)[:, :sequence], mean


@torch.no_grad()
def vc_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           *, bits: int = 4, permutation: torch.Tensor | None = None) -> torch.Tensor:
    """Compute a VC-style online recurrence for small global GQA inputs.

    Supports the MagiHuman [V,A,T] order because K/V receive the same
    per-KV-head permutation while Q remains in its original order.
    """
    if bits not in (0, 8) or q.ndim != 4 or q.shape[0] != 1 or k.shape != v.shape:
        raise ValueError("Expected bits=0/8 and matching batch-one Q/K/V")
    if q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1] or q.shape[2] % k.shape[2]:
        raise ValueError("Expected self-attention with an integral GQA ratio")
    sequence = q.shape[1]
    if sequence > 1024:
        raise ValueError("Reference implementation is capped at 1024 tokens")
    ratio = q.shape[2] // k.shape[2]
    values = v[0].permute(1, 0, 2).float().contiguous()
    if permutation is None:
        permutation = value_permutation(values)
    indices = permutation.long()[:, :, None].expand_as(values)
    ordered_v = values.gather(1, indices)
    ordered_k = k[0].permute(1, 0, 2).float().gather(1, indices)
    q_heads = q[0].permute(1, 0, 2).float()

    q_quant, k_quant = q_heads, ordered_k
    if bits == 0:
        value_pad = (-sequence) % BLOCK
        blocks = F.pad(ordered_v, (0, 0, 0, value_pad)).reshape(k.shape[2], -1, BLOCK, q.shape[-1])
        lengths = torch.full((blocks.shape[1],), BLOCK, device=q.device, dtype=torch.float32)
        if value_pad:
            lengths[-1] -= value_pad
        means = blocks.sum(2) / lengths[None, :, None]
        residual = ordered_v - means[:, torch.arange(sequence, device=q.device) // BLOCK]
    else:
        k_quant = k_quant - k_quant.mean(1, keepdim=True)
        q_quant = _quantize_blocks(_hadamard(q_quant))
        k_quant = _quantize_blocks(_hadamard(k_quant))
        residual, means = _quantize_residual(ordered_v, bits)
    k_quant = k_quant.repeat_interleave(ratio, 0)
    residual = residual.repeat_interleave(ratio, 0)
    means = means.repeat_interleave(ratio, 0)
    output = torch.empty_like(q_heads)
    heads = q_heads.shape[0]
    for q_start in range(0, sequence, 64):
        q_end = min(q_start + 64, sequence)
        queries = q_quant[:, q_start:q_end]
        rows = q_end - q_start
        row_max = torch.full((heads, rows, 1), -torch.inf, device=q.device)
        row_sum = torch.zeros((heads, rows, 1), device=q.device)
        numerator = torch.zeros((heads, rows, q.shape[-1]), device=q.device)
        for k_start in range(0, sequence, BLOCK):
            k_end = min(k_start + BLOCK, sequence)
            scores = torch.bmm(queries, k_quant[:, k_start:k_end].transpose(1, 2)) / math.sqrt(q.shape[-1])
            new_max = torch.maximum(row_max, scores.amax(-1, keepdim=True))
            alpha = torch.exp(row_max - new_max)
            probs = torch.exp(scores - new_max)
            mma_probs = probs.to(torch.float8_e4m3fn).float() if bits == 8 else probs
            # The low-bit probability is consumed only by PV. Online
            # normalization and exact mean restoration use the FP32 row sum.
            tile_sum = probs.sum(-1, keepdim=True)
            numerator = alpha * numerator + torch.bmm(mma_probs, residual[:, k_start:k_end])
            numerator = numerator + tile_sum * means[:, k_start // BLOCK, None, :]
            row_sum = alpha * row_sum + tile_sum
            row_max = new_max
        output[:, q_start:q_end] = numerator / row_sum
    return output.permute(1, 0, 2)[None].to(q.dtype)

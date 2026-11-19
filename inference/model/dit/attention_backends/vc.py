"""Preparation for the copied-FA4 all-E4M3 VC-Attention path."""

import torch


def _hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """Apply the same normalized Walsh-Hadamard transform on the last axis."""
    dim = x.shape[-1]
    if dim < 1 or dim & (dim - 1):
        raise ValueError("Hadamard rotation requires a power-of-two head dimension")
    y = x.float()
    width = 1
    while width < dim:
        shape = y.shape
        pairs = y.reshape(*shape[:-1], -1, 2, width)
        lo, hi = pairs[..., 0, :], pairs[..., 1, :]
        y = torch.cat((lo + hi, lo - hi), dim=-1).reshape(shape)
        width *= 2
    return y * (dim ** -0.5)


def _quantize_qk_fp8(q, k, permutation):
    """Dispatch the production path to the fused SM120 CuTe implementation."""
    from .vc_attention.prepare_qk_fp8 import prepare_fp8_qk

    return prepare_fp8_qk(q, k, permutation)


def _value_permutation(
    values: torch.Tensor, clusters: int = 32, *,
    initial_centers: torch.Tensor | None = None,
    return_centers: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Online K-means per KV head; warm starts use one update instead of two."""
    heads, sequence, dim = values.shape
    count = min(clusters, max(1, sequence // 128))
    if (
        values.device.type == "cuda"
        and torch.cuda.get_device_capability(values.device) == (12, 0)
        and values.dtype == torch.bfloat16
        and dim == 128
        and clusters == 32
    ):
        from .vc_attention.group_values import group_values_cute

        if initial_centers is not None and initial_centers.shape != (heads, count, dim):
            raise ValueError("Warm-start centers must have shape [Hkv, clusters, D]")
        permutation, centers = group_values_cute(
            values, initial_centers=initial_centers
        )
        return (permutation, centers) if return_centers else permutation

    # CPU/reference fallback; production SM120 grouping stays in CuTe kernels.
    data = values.float()
    if initial_centers is None:
        sample = torch.linspace(0, sequence - 1, count, device=values.device).long()
        centers = data[:, sample].contiguous()
    else:
        if (initial_centers.shape != (heads, count, dim)
                or initial_centers.device != values.device
                or initial_centers.dtype != torch.float32):
            raise ValueError("Warm-start centers must be FP32 [Hkv, clusters, D] on the input GPU")
        centers = initial_centers
    for _ in range(1 if initial_centers is not None else 2):
        similarity = torch.bmm(data, centers.transpose(1, 2))
        similarity -= 0.5 * (centers * centers).sum(-1)[:, None, :]
        labels = similarity.argmax(-1)
        sums = torch.zeros_like(centers)
        sums.scatter_add_(1, labels[:, :, None].expand(-1, -1, dim), data)
        counts = torch.zeros(heads, count, device=values.device, dtype=torch.float32)
        counts.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float32))
        centers = torch.where(counts[:, :, None] > 0, sums / counts.clamp_min(1)[:, :, None], centers)
    similarity = torch.bmm(data, centers.transpose(1, 2))
    similarity -= 0.5 * (centers * centers).sum(-1)[:, None, :]
    permutation = similarity.argmax(-1).argsort(dim=1, stable=True).to(torch.int32)
    return (permutation, centers) if return_centers else permutation


@torch.no_grad()
def prepare_fa4_fp8_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    permutation: torch.Tensor | None = None,
    *, smooth: bool = True,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Prepare FP8 Q/K and the permuted, demeaned FP8 V residual."""
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("VC FP8 source backend requires an SM120 GPU")
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[0] != 1:
        raise ValueError("Expected batch-one [B, S, H, D] Q/K/V")
    if q.shape[1] != k.shape[1] or q.shape[-1] != 128 or k.shape[-1] != 128:
        raise ValueError("Expected self-attention with head dimension 128")
    if q.shape[2] % k.shape[2] or any(t.dtype != torch.bfloat16 or t.device != q.device for t in (q, k, v)):
        raise ValueError("Expected BF16 operands on one GPU and an integral GQA ratio")

    from .vc_attention.prepare_fp8 import prepare_fp8_v
    sequence, kv_heads = k.shape[1:3]
    if permutation is None:
        permutation = _value_permutation(v[0].permute(1, 0, 2).contiguous())
    elif permutation.shape != (kv_heads, sequence) or permutation.device != q.device:
        raise ValueError("Expected the cached permutation with shape [Hkv, S]")
    value_codes, means, value_scales = prepare_fp8_v(v, permutation, smooth=smooth)
    # K gather is fused into centering, matched Hadamard rotation, block
    # scaling, and E4M3 quantization; no ordered BF16 K reaches HBM.
    q_codes, k_codes, q_scales, k_scales = _quantize_qk_fp8(q, k, permutation)
    v_codes = value_codes.reshape(kv_heads, -1, 128)[:, :sequence]
    v_codes = v_codes.permute(1, 0, 2)[None].contiguous()
    return (
        q_codes.contiguous(), k_codes.contiguous(), v_codes,
        q_scales.contiguous(), k_scales.contiguous(),
        means[None], value_scales[None],
    ), permutation


@torch.no_grad()
def vc_attention_fa4_fp8(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *, permutation: torch.Tensor | None = None, smooth: bool = True,
) -> torch.Tensor:
    """Run the copied FA4 SM120 FP8 mainloop, optionally smoothing V."""
    from .vc_attention import flash_attn_v_smooth_fp8_func
    inputs, _ = prepare_fa4_fp8_inputs(q, k, v, permutation, smooth=smooth)
    output, _ = flash_attn_v_smooth_fp8_func(
        *inputs, smooth=smooth,
    )
    return output

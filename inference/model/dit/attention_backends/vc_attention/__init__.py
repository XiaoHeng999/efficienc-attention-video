"""Minimal FA4 source fork for all-E4M3 VC-Attention."""


def flash_attn_func(*args, **kwargs):
    """Run the unchanged-dtype copied FA4 baseline."""
    from .interface import flash_attn_func as copied_flash_attn_func

    return copied_flash_attn_func(*args, **kwargs)


def flash_attn_v_smooth_fp8_func(
    q,
    k,
    v_residual_codes,
    q_scales,
    k_scales,
    means,
    value_scales=None,
    *,
    smooth=True,
):
    """Run FP8 QK and residual-V PV in copied FA4.

    Q/K and the demeaned V residual are E4M3. The kernel applies Q/K descales,
    runs standard online softmax before its E4M3 probability cast, and restores
    the V block mean inside the online recurrence. PV consumes UE8M0 value
    scales directly through SM120 MXFP8 block-scaled MMA.
    """
    import torch

    from .interface import _flash_attn_fwd

    v = v_residual_codes
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, means)):
        raise ValueError("The VC-Attention source path is inference-only")
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (12, 0):
        raise ValueError("The VC-Attention source path requires SM120")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[:2] != k.shape[:2]:
        raise ValueError("Expected self-attention [B, S, H, D] Q/K/V")
    if q.shape[2] % k.shape[2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Expected matching head dimension and integral GQA ratio")
    if any(t.dtype != torch.float8_e4m3fn for t in (q, k, v)):
        raise ValueError("Q, K, and the V residual must be E4M3")
    if any(t.device != q.device for t in (k, v)):
        raise ValueError("Expected Q/K/V on one device")
    shape = (k.shape[0], k.shape[2], (k.shape[1] + 127) // 128, v.shape[-1])
    q_scale_shape = (
        k.shape[0], k.shape[2],
        ((q.shape[1] * (q.shape[2] // k.shape[2]) + 63) // 64) * 2,
    )
    k_scale_shape = (k.shape[0], k.shape[2], (k.shape[1] + 63) // 64)
    for scale, name, expected in (
        (q_scales, "Q", q_scale_shape),
        (k_scales, "K", k_scale_shape),
    ):
        if scale.shape != expected or scale.dtype != torch.float32 or scale.device != q.device:
            raise ValueError(f"Expected FP32 {name} scales with shape {expected}")
    if means.shape != shape or means.dtype != torch.bfloat16 or means.device != q.device:
        raise ValueError(f"Expected BF16 V block means with shape {shape}")
    if value_scales is None:
        value_scales = torch.ones(shape, device=q.device, dtype=torch.float8_e8m0fnu)
    if (
        value_scales.shape != shape
        or value_scales.dtype != torch.float8_e8m0fnu
        or value_scales.device != q.device
    ):
        raise ValueError(f"Expected UE8M0 V scales with shape {shape}")
    output, lse, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        tile_mn=(128, 64),
        aux_tensors=[
            q_scales.contiguous(), k_scales.contiguous(),
            means.contiguous(), value_scales.contiguous(),
        ],
        v_smooth=smooth,
    )
    return output, lse


def vc_attention(q, k, v, *, permutation=None, smooth=True):
    """Prepare V-Smooth operands and run the fused copied-FA4 kernel."""
    from ..vc import vc_attention_fa4_fp8

    return vc_attention_fa4_fp8(q, k, v, permutation=permutation, smooth=smooth)


def nvfp4_attention(q, k, v, *, permutation=None, pv_mode="fp8"):
    """Run the optional grouped SM120 NVFP4 comparison path."""
    from .nvfp4 import vc_attention_nvfp4

    return vc_attention_nvfp4(q, k, v, permutation=permutation, pv_mode=pv_mode)


__all__ = [
    "flash_attn_func",
    "flash_attn_v_smooth_fp8_func",
    "vc_attention",
    "nvfp4_attention",
]

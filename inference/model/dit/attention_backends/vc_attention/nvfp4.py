"""Optional SM120 NVFP4 attention experiment.

This route intentionally stays separate from the copied-FA4 FP8 V-Smooth
kernel.  It uses FlashInfer's native SM120 block-scaled E2M1 kernel and keeps
the value-guided K/V permutation, but it does not yet restore a subtracted V
block mean inside the online recurrence.
"""

from __future__ import annotations

import torch


_cu_seqlens: dict[tuple[torch.device, int], torch.Tensor] = {}


def _ordered_kv(
    k: torch.Tensor, v: torch.Tensor, permutation: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous [B, Hkv, S, D] K/V with a shared token order."""
    k_hsd = k.permute(0, 2, 1, 3)
    v_hsd = v.permute(0, 2, 1, 3)
    if permutation is None:
        return k_hsd.contiguous(), v_hsd.contiguous()
    if permutation.shape != k_hsd.shape[1:3]:
        raise ValueError("permutation must have shape [Hkv, S]")
    if permutation.device != k.device or permutation.dtype != torch.int32:
        raise ValueError("permutation must be an int32 tensor on the Q/K/V GPU")
    index = permutation.long()[None, :, :, None].expand_as(k_hsd)
    return k_hsd.gather(2, index).contiguous(), v_hsd.gather(2, index).contiguous()


def _rotate_qk(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the normalized D=128 Hadamard used by the FP8 preparation path."""
    try:
        from quack.transform.hadamard import hadamard_transform_fwd
    except ImportError as exc:
        raise RuntimeError("NVFP4 Q/K smoothing requires the FA4 QuACK dependency") from exc
    scale = 128**-0.5
    return hadamard_transform_fwd(q, scale), hadamard_transform_fwd(k, scale)


@torch.no_grad()
def vc_attention_nvfp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    permutation: torch.Tensor | None = None,
    rotate_qk: bool = True,
    center_v: bool = True,
    pv_mode: str = "fp8",
) -> torch.Tensor:
    """Run grouped NVFP4 attention on SM120.

    The quality-first ``pv_mode="fp8"`` keeps NVFP4 Q/K but uses E4M3 P/V.
    ``pv_mode="fp4"`` uses E2M1 for Q/K/P/V. Both use one E4M3 scale per 16
    NVFP4 values. Q/K receive a normalized Hadamard rotation before K is
    channel-centred and Q is globally centred; the compact correction is added
    to the scores, so both transforms preserve full-precision attention.
    A global Q mean is used instead of a mean per 128 rows to keep the
    correction buffer O(H*S), rather than O(H*S^2/128), at video lengths.

    V subtracts one BF16 channel mean per KV head before FP4 quantization and
    adds it to the normalized output. Unlike a per-block mean, this restoration
    is exact without probability row sums because softmax rows sum to one.

    This is not the FP8 per-block V-Smooth kernel: the global V mean is restored,
    but there is no per-128-token rank-one mean term inside the recurrence.
    """
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise ValueError("NVFP4 attention is inference-only")
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("NVFP4 attention requires an SM120 GPU")
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[0] != 1:
        raise ValueError("Expected batch-one [B, S, H, D] Q/K/V")
    if q.shape[1] != k.shape[1] or q.shape[-1] != 128 or k.shape[-1] != 128:
        raise ValueError("Expected self-attention with head dimension 128")
    if q.shape[2] % k.shape[2]:
        raise ValueError("Q heads must be an integral multiple of KV heads")
    if any(t.dtype != torch.bfloat16 or t.device != q.device for t in (q, k, v)):
        raise ValueError("Expected contiguous-compatible BF16 operands on one GPU")
    if pv_mode not in {"fp4", "fp8"}:
        raise ValueError("pv_mode must be 'fp4' or 'fp8'")

    if pv_mode == "fp8":
        return _nvfp4_qk_fp8_pv(q, k, v, permutation, center_v=center_v)

    try:
        import flashinfer
    except ImportError as exc:
        raise RuntimeError(
            "The optional NVFP4 path requires a FlashInfer build containing "
            "nvfp4_attention_sm120_quantize_qkv"
        ) from exc
    if not all(
        hasattr(flashinfer, name)
        for name in ("nvfp4_attention_sm120_quantize_qkv", "nvfp4_attention_sm120_fwd")
    ):
        raise RuntimeError("Installed FlashInfer does not provide SM120 NVFP4 attention")

    sequence = q.shape[1]
    # The fused CuTe preparation wins once launch overhead is amortized; the
    # smaller QuACK/PyTorch path remains slightly faster at short sequences.
    if rotate_qk and center_v and permutation is not None and sequence >= 8192:
        from .prepare_nvfp4 import prepare_nvfp4_bf16

        q_hsd, k_hsd, v_hsd, value_mean = prepare_nvfp4_bf16(q, k, v, permutation)
    else:
        q_hsd = q.permute(0, 2, 1, 3).contiguous()
        k_hsd, v_hsd = _ordered_kv(k, v, permutation)
        if rotate_qk:
            q_hsd, k_hsd = _rotate_qk(q_hsd, k_hsd)
        value_mean = None
        if center_v:
            value_mean = v_hsd.float().mean(dim=2, keepdim=True).to(torch.bfloat16)
            v_hsd = (v_hsd - value_mean).contiguous()
    operands = flashinfer.nvfp4_attention_sm120_quantize_qkv(
        q_hsd, k_hsd, v_hsd, per_block_mean=False
    )
    output = flashinfer.nvfp4_attention_sm120_fwd(
        *operands,
        per_block_mean=False,
        return_lse=False,
        unpadded_k_len=sequence,
    )
    if value_mean is not None:
        output = output + value_mean.repeat_interleave(q.shape[2] // k.shape[2], dim=1)
    return output[:, :, :sequence].permute(0, 2, 1, 3).contiguous()


def _nvfp4_qk_fp8_pv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    permutation: torch.Tensor | None,
    *,
    center_v: bool,
) -> torch.Tensor:
    """Quality-first NVFP4-QK/E4M3-PV route for non-causal self-attention."""
    try:
        from flashinfer.diffusion_ops import minimax_h3_sm120_varlen_attention_nvfp4
    except ImportError as exc:
        raise RuntimeError("Installed FlashInfer lacks the SM120 NVFP4/FP8-PV kernel") from exc

    sequence = q.shape[1]
    k_hsd, v_hsd = _ordered_kv(k, v, permutation)
    value_mean = None
    if center_v:
        value_mean = v_hsd.float().mean(dim=2, keepdim=True).to(torch.bfloat16)
        v_hsd = (v_hsd - value_mean).contiguous()

    ratio = q.shape[2] // k.shape[2]
    k_thd = k_hsd.repeat_interleave(ratio, dim=1).permute(0, 2, 1, 3).squeeze(0).contiguous()
    v_thd = v_hsd.repeat_interleave(ratio, dim=1).permute(0, 2, 1, 3).squeeze(0).contiguous()
    key = (q.device, sequence)
    cu = _cu_seqlens.get(key)
    if cu is None:
        cu = torch.tensor((0, sequence), device=q.device, dtype=torch.int32)
        _cu_seqlens[key] = cu
    output = minimax_h3_sm120_varlen_attention_nvfp4(
        q.squeeze(0).contiguous(),
        k_thd,
        v_thd,
        cu,
        cu_seqlens_host=(0, sequence),
        pv_mode="fp8",
    )
    if value_mean is not None:
        output = output + value_mean.repeat_interleave(ratio, dim=1).permute(0, 2, 1, 3).squeeze(0)
    return output[None]


__all__ = ["vc_attention_nvfp4"]

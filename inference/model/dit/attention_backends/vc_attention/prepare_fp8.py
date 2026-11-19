"""CuTe V-Smooth preparation for the SM120 FP8 experiment.

One CTA's threads each own a value channel of one 128-token block. The kernel
gathers V under the permutation, computes the block mean and its per-channel
FP8 scale, and writes the residual in the column-major layout expected by PV.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.runtime import from_dlpack
import torch


BLOCK = 128
_compiled = {}


@cute.kernel
def _prepare_kernel(
    v: cute.Tensor,
    permutation: cute.Tensor,
    value_codes: cute.Tensor,
    means: cute.Tensor,
    scales: cute.Tensor,
    sequence: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    smooth: cutlass.Constexpr,
):
    thread, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    blocks = (sequence + BLOCK - 1) // BLOCK
    linear = cta * 256 + thread
    if linear < heads * blocks * dim:
        channel = linear % dim
        value_block = (linear // dim) % blocks
        head = linear // (dim * blocks)
        start = value_block * BLOCK
        count = min(BLOCK, sequence - start)

        mean = cutlass.Float32(0.0)
        if const_expr(smooth):
            total = cutlass.Float32(0.0)
            for offset in range(BLOCK):
                if offset < count:
                    source = permutation[head, start + offset]
                    total += v[0, source, head, channel].to(cutlass.Float32)
            mean = total / count
        # Store the paper's 16-bit mean and quantize the residual against that
        # exact stored value, so mean rounding is absorbed by the residual.
        mean_code = mean.to(cutlass.BFloat16)
        mean = mean_code.to(cutlass.Float32)
        amax = cutlass.Float32(0.0)
        for offset in range(BLOCK):
            if offset < count:
                source = permutation[head, start + offset]
                residual = v[0, source, head, channel].to(cutlass.Float32) - mean
                amax = cutlass.max(amax, cute.math.absf(residual))
        scale = cutlass.max(amax / cutlass.Float32(448.0), cutlass.Float32(1e-8))
        # MXFP8 requires one UE8M0 power-of-two scale per 32 K elements. The
        # 128-token VC block repeats this channel scale over its four K groups.
        scale_code = scale.to(cutlass.Float8E8M0FNU)
        scale = scale_code.to(cutlass.Float32)
        scales[head, value_block, channel] = scale_code
        means[head, value_block, channel] = mean_code

        for offset in range(BLOCK):
            # The SM120 QK-C -> PV-A register pack uses this fixed order in
            # each 16-key group. Store V in the matching order so the register
            # conversion remains exact without a shared-memory transpose.
            group = offset // 16 * 16
            quad = (offset % 16) // 4
            element = offset % 4
            ordered_offset = group + 8 * (element // 2) + 2 * quad + element % 2
            if ordered_offset < count:
                source = permutation[head, start + ordered_offset]
                residual = v[0, source, head, channel].to(cutlass.Float32) - mean
                code = cutlass.min(
                    cutlass.max(residual / scale, cutlass.Float32(-448.0)),
                    cutlass.Float32(448.0),
                ).to(cutlass.Float8E4M3FN)
                value_codes[head, value_block, offset, channel] = code
            else:
                value_codes[head, value_block, offset, channel] = cutlass.Float8E4M3FN(0)


@cute.jit
def _prepare(
    v: cute.Tensor,
    permutation: cute.Tensor,
    value_codes: cute.Tensor,
    means: cute.Tensor,
    scales: cute.Tensor,
    sequence: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    smooth: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    blocks = (sequence + BLOCK - 1) // BLOCK
    total = heads * blocks * dim
    _prepare_kernel(
        v, permutation, value_codes, means, scales,
        sequence, heads, dim, smooth,
    ).launch(grid=((total + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


def prepare_fp8_v(
    v: torch.Tensor, permutation: torch.Tensor, *, smooth: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP8 V residual codes, BF16 means, and UE8M0 channel scales."""
    if (
        v.ndim != 4
        or v.shape[0] != 1
        or v.dtype != torch.bfloat16
        or v.device.type != "cuda"
        or torch.cuda.get_device_capability(v.device) != (12, 0)
    ):
        raise ValueError("Expected batch-one SM120 CUDA BF16 V")
    sequence, heads, dim = v.shape[1:]
    if permutation.shape != (heads, sequence):
        raise ValueError("Expected permutation shape [Hkv, S]")
    v = v.contiguous()
    permutation = permutation.to(torch.int32).contiguous()
    blocks = (sequence + BLOCK - 1) // BLOCK
    value_codes = torch.empty_strided(
        (heads, blocks, BLOCK, dim),
        (blocks * BLOCK * dim, BLOCK * dim, 1, BLOCK),
        device=v.device,
        dtype=torch.float8_e4m3fn,
    )
    means = torch.empty((heads, blocks, dim), device=v.device, dtype=torch.bfloat16)
    scales = torch.empty(
        (heads, blocks, dim), device=v.device, dtype=torch.float8_e8m0fnu
    )
    stream = cuda.CUstream(torch.cuda.current_stream(v.device).cuda_stream)
    args = (
        from_dlpack(v),
        from_dlpack(permutation),
        from_dlpack(value_codes),
        from_dlpack(means),
        from_dlpack(scales),
        sequence,
        heads,
        dim,
        smooth,
        stream,
    )
    key = (v.device.index, sequence, heads, dim, smooth)
    if key not in _compiled:
        _compiled[key] = cute.compile(_prepare, *args)
    _compiled[key](*args[:5], stream)
    return value_codes, means, scales

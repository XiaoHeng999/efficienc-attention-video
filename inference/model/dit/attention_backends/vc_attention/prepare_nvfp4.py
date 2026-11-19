"""Fused BF16 preparation feeding the optional SM120 NVFP4 quantizer."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cute.runtime import from_dlpack
import torch


DIM = 128
ROWS = 64
THREADS = 128
_compiled = {}


@cute.kernel
def _rotate_kernel(
    source: cute.Tensor,
    permutation: cute.Tensor,
    output: cute.Tensor,
    sequence: cutlass.Constexpr,
    kv_heads: cutlass.Constexpr,
    ratio: cutlass.Constexpr,
    is_query: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    warp = tidx // 32
    total_rows = sequence * ratio if const_expr(is_query) else sequence
    blocks = (total_rows + ROWS - 1) // ROWS
    kv_head = cta // blocks
    block = cta - kv_head * blocks

    for row_iter in cutlass.range_constexpr(ROWS // 4):
        row = warp + row_iter * 4
        packed_row = block * ROWS + row
        valid = packed_row < total_rows
        values = cute.make_rmem_tensor((4,), Float32)
        token = cutlass.Int32(0)
        out_head = cutlass.Int32(0)
        if valid:
            if const_expr(is_query):
                token = packed_row // ratio
                out_head = kv_head * ratio + packed_row - token * ratio
            else:
                token = permutation[kv_head, packed_row]
                out_head = kv_head
        for item in cutlass.range_constexpr(4):
            channel = lane + item * 32
            values[item] = (
                source[0, token, out_head, channel].to(Float32)
                if valid
                else Float32(0.0)
            )

        for width in (1, 2, 4, 8, 16):
            for item in cutlass.range_constexpr(4):
                value = values[item]
                peer = cute.arch.shuffle_sync_bfly(value, offset=width)
                values[item] = peer - value if lane & width else value + peer
        a0, a1, a2, a3 = values[0], values[1], values[2], values[3]
        values[0], values[1] = a0 + a1, a0 - a1
        values[2], values[3] = a2 + a3, a2 - a3
        a0, a1, a2, a3 = values[0], values[1], values[2], values[3]
        values[0], values[2] = a0 + a2, a0 - a2
        values[1], values[3] = a1 + a3, a1 - a3

        if valid:
            for item in cutlass.range_constexpr(4):
                channel = lane + item * 32
                rotated = (values[item] * Float32(0.08838834764831845)).to(
                    cutlass.BFloat16
                )
                if const_expr(is_query):
                    output[0, out_head, token, channel] = rotated
                else:
                    output[0, out_head, block * ROWS + row, channel] = rotated


@cute.kernel
def _value_mean_kernel(
    v: cute.Tensor,
    means: cute.Tensor,
    sequence: cutlass.Constexpr,
    heads: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    linear = cta * 256 + tidx
    if linear < heads * DIM:
        head = linear // DIM
        channel = linear - head * DIM
        total = Float32(0.0)
        for token in range(sequence):
            total += v[0, token, head, channel].to(Float32)
        means[0, head, 0, channel] = (total / sequence).to(cutlass.BFloat16)


@cute.kernel
def _value_residual_kernel(
    v: cute.Tensor,
    permutation: cute.Tensor,
    means: cute.Tensor,
    residual: cute.Tensor,
    sequence: cutlass.Constexpr,
    heads: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    linear = cta * 256 + tidx
    total = heads * sequence * DIM
    if linear < total:
        channel = linear % DIM
        token = (linear // DIM) % sequence
        head = linear // (DIM * sequence)
        source = permutation[head, token]
        residual[0, head, token, channel] = (
            v[0, source, head, channel].to(Float32)
            - means[0, head, 0, channel].to(Float32)
        ).to(cutlass.BFloat16)


@cute.jit
def _prepare(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    permutation: cute.Tensor,
    q_rotated: cute.Tensor,
    k_rotated: cute.Tensor,
    v_residual: cute.Tensor,
    value_mean: cute.Tensor,
    sequence: cutlass.Constexpr,
    kv_heads: cutlass.Constexpr,
    ratio: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    q_blocks = (sequence * ratio + ROWS - 1) // ROWS
    k_blocks = (sequence + ROWS - 1) // ROWS
    _rotate_kernel(q, permutation, q_rotated, sequence, kv_heads, ratio, True).launch(
        grid=(kv_heads * q_blocks, 1, 1), block=(THREADS, 1, 1), stream=stream
    )
    _rotate_kernel(k, permutation, k_rotated, sequence, kv_heads, ratio, False).launch(
        grid=(kv_heads * k_blocks, 1, 1), block=(THREADS, 1, 1), stream=stream
    )
    _value_mean_kernel(v, value_mean, sequence, kv_heads).launch(
        grid=((kv_heads * DIM + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
    )
    total = kv_heads * sequence * DIM
    _value_residual_kernel(
        v, permutation, value_mean, v_residual, sequence, kv_heads
    ).launch(grid=((total + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


def prepare_nvfp4_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rotated Q/K, demeaned V and its BF16 head mean in BHSD layout."""
    sequence, query_heads, dim = q.shape[1:]
    kv_heads = k.shape[2]
    ratio = query_heads // kv_heads
    q_rotated = torch.empty((1, query_heads, sequence, dim), device=q.device, dtype=q.dtype)
    k_rotated = torch.empty((1, kv_heads, sequence, dim), device=q.device, dtype=q.dtype)
    v_residual = torch.empty_like(k_rotated)
    value_mean = torch.empty((1, kv_heads, 1, dim), device=q.device, dtype=q.dtype)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    args = (
        from_dlpack(q.contiguous()),
        from_dlpack(k.contiguous()),
        from_dlpack(v.contiguous()),
        from_dlpack(permutation.to(torch.int32).contiguous()),
        from_dlpack(q_rotated),
        from_dlpack(k_rotated),
        from_dlpack(v_residual),
        from_dlpack(value_mean),
        sequence,
        kv_heads,
        ratio,
        stream,
    )
    key = (q.device.index, sequence, query_heads, kv_heads, dim)
    if key not in _compiled:
        _compiled[key] = cute.compile(_prepare, *args)
    _compiled[key](*args[:8], stream)
    return q_rotated, k_rotated, v_residual, value_mean


__all__ = ["prepare_nvfp4_bf16"]

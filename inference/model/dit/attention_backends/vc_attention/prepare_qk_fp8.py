"""Fused SM120 CuTe preparation for E4M3 Q/K operands.

K's sequence-wide channel mean needs a separate reduction launch because CUDA
CTAs have no grid-wide synchronization inside an ordinary kernel. The two
quantization launches then fuse gather/centering, the 128-point Hadamard,
block amax, scaling, and E4M3 storage without an FP32 HBM intermediate.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cute.runtime import from_dlpack
import torch


DIM = 128
THREADS = 128
_compiled = {}


@cute.kernel
def _key_mean_kernel(k: cute.Tensor, means: cute.Tensor, sequence: cutlass.Constexpr,
                     heads: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    linear = bidx * 256 + tidx
    if linear < heads * DIM:
        head = linear // DIM
        channel = linear % DIM
        total = Float32(0.0)
        for token in range(sequence):
            total += k[0, token, head, channel].to(Float32)
        means[head, channel] = total / sequence


@cute.kernel
def _quantize_kernel(
    source: cute.Tensor,
    permutation: cute.Tensor,
    key_means: cute.Tensor,
    codes: cute.Tensor,
    scales: cute.Tensor,
    sequence: cutlass.Constexpr,
    kv_heads: cutlass.Constexpr,
    ratio: cutlass.Constexpr,
    block_rows: cutlass.Constexpr,
    is_query: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    warp_idx = tidx // 32
    total_rows = sequence * ratio if const_expr(is_query) else sequence
    blocks = (total_rows + block_rows - 1) // block_rows
    head = cta // blocks
    block = cta - head * blocks

    @cute.struct
    class SharedStorage:
        values: cute.struct.Align[cute.struct.MemRange[Float32, 64 * DIM], 16]
        warp_max: cute.struct.Align[cute.struct.MemRange[Float32, 4], 16]

    allocator = cutlass.utils.SmemAllocator()
    storage = allocator.allocate(SharedStorage)
    values = storage.values.get_tensor(cute.make_layout((64, DIM), stride=(DIM, 1)))
    warp_max = storage.warp_max.get_tensor(cute.make_layout(4))

    local_max = Float32(0.0)
    rows_per_warp = block_rows // 4
    for row_iter in cutlass.range_constexpr(rows_per_warp):
        row = warp_idx + row_iter * 4
        packed_row = block * block_rows + row
        valid = packed_row < total_rows
        transformed = cute.make_rmem_tensor((4,), Float32)
        for item in cutlass.range_constexpr(4):
            channel = lane + item * 32
            value = Float32(0.0)
            if valid:
                if const_expr(is_query):
                    token = packed_row // ratio
                    query_in_group = packed_row - token * ratio
                    query_head = head * ratio + query_in_group
                    value = source[0, token, query_head, channel].to(Float32)
                else:
                    token = permutation[head, packed_row]
                    value = (
                        source[0, token, head, channel].to(Float32)
                        - key_means[head, channel]
                    )
            transformed[item] = value

        # Widths 1..16 exchange channels held by different lanes.
        for width in (1, 2, 4, 8, 16):
            for item in cutlass.range_constexpr(4):
                value = transformed[item]
                peer = cute.arch.shuffle_sync_bfly(value, offset=width)
                transformed[item] = peer - value if lane & width else value + peer

        # Widths 32 and 64 exchange the four channel groups held by each lane.
        a0, a1, a2, a3 = (
            transformed[0], transformed[1], transformed[2], transformed[3]
        )
        transformed[0], transformed[1] = a0 + a1, a0 - a1
        transformed[2], transformed[3] = a2 + a3, a2 - a3
        a0, a1, a2, a3 = (
            transformed[0], transformed[1], transformed[2], transformed[3]
        )
        transformed[0], transformed[2] = a0 + a2, a0 - a2
        transformed[1], transformed[3] = a1 + a3, a1 - a3

        for item in cutlass.range_constexpr(4):
            channel = lane + item * 32
            value = transformed[item] * Float32(0.08838834764831845)
            values[row, channel] = value
            local_max = cutlass.max(local_max, cute.math.absf(value))

    for width in (1, 2, 4, 8, 16):
        local_max = cutlass.max(
            local_max, cute.arch.shuffle_sync_bfly(local_max, offset=width)
        )
    if lane == 0:
        warp_max[warp_idx] = local_max
    cute.arch.barrier()

    if warp_idx == 0:
        block_max = warp_max[lane] if lane < 4 else Float32(0.0)
        for width in (1, 2, 4, 8, 16):
            block_max = cutlass.max(
                block_max, cute.arch.shuffle_sync_bfly(block_max, offset=width)
            )
        if lane == 0:
            scale = cutlass.max(block_max / Float32(448.0), Float32(1e-8))
            warp_max[0] = scale
            scales[0, head, block] = scale
    cute.arch.barrier()
    scale = warp_max[0]

    for row_iter in cutlass.range_constexpr(rows_per_warp):
        row = warp_idx + row_iter * 4
        packed_row = block * block_rows + row
        if packed_row < total_rows:
            for item in cutlass.range_constexpr(4):
                channel = lane + item * 32
                code = cutlass.min(
                    cutlass.max(values[row, channel] / scale, Float32(-448.0)),
                    Float32(448.0),
                ).to(cutlass.Float8E4M3FN)
                if const_expr(is_query):
                    token = packed_row // ratio
                    query_in_group = packed_row - token * ratio
                    query_head = head * ratio + query_in_group
                    codes[0, token, query_head, channel] = code
                else:
                    codes[0, packed_row, head, channel] = code


@cute.jit
def _prepare_qk(
    q: cute.Tensor,
    k: cute.Tensor,
    permutation: cute.Tensor,
    q_codes: cute.Tensor,
    k_codes: cute.Tensor,
    q_scales: cute.Tensor,
    k_scales: cute.Tensor,
    key_means: cute.Tensor,
    sequence: cutlass.Constexpr,
    kv_heads: cutlass.Constexpr,
    ratio: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _key_mean_kernel(k, key_means, sequence, kv_heads).launch(
        grid=((kv_heads * DIM + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
    )
    q_blocks = ((sequence * ratio + 63) // 64) * 2
    _quantize_kernel(
        q, permutation, key_means, q_codes, q_scales,
        sequence, kv_heads, ratio, 32, True,
    ).launch(grid=(kv_heads * q_blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)
    k_blocks = (sequence + 63) // 64
    _quantize_kernel(
        k, permutation, key_means, k_codes, k_scales,
        sequence, kv_heads, ratio, 64, False,
    ).launch(grid=(kv_heads * k_blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)


def prepare_fp8_qk(q: torch.Tensor, k: torch.Tensor, permutation: torch.Tensor):
    """Return fused-prepared E4M3 Q/K and their FP32 block scales."""
    sequence, query_heads, dim = q.shape[1:]
    kv_heads = k.shape[2]
    ratio = query_heads // kv_heads
    q_codes = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    k_codes = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    q_scales = torch.empty(
        (1, kv_heads, ((sequence * ratio + 63) // 64) * 2),
        device=q.device, dtype=torch.float32,
    )
    k_scales = torch.empty(
        (1, kv_heads, (sequence + 63) // 64),
        device=q.device, dtype=torch.float32,
    )
    key_means = torch.empty((kv_heads, dim), device=q.device, dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    args = (
        from_dlpack(q.contiguous()), from_dlpack(k.contiguous()),
        from_dlpack(permutation.to(torch.int32).contiguous()),
        from_dlpack(q_codes), from_dlpack(k_codes),
        from_dlpack(q_scales), from_dlpack(k_scales), from_dlpack(key_means),
        sequence, kv_heads, ratio, stream,
    )
    key = (q.device.index, sequence, query_heads, kv_heads, dim)
    if key not in _compiled:
        _compiled[key] = cute.compile(_prepare_qk, *args)
    _compiled[key](*args[:8], stream)
    return q_codes, k_codes, q_scales, k_scales


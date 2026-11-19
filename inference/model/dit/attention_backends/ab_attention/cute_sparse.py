"""CuTe DSL reference executor for AB's weighted video attention.

One CTA owns one 128-query tile and one Q head. Four warps process four
queries at a time, using the same SIP plan for the whole tile. This deliberately
favors an auditable translation over tensor-core performance; it does not edit
FlashAttention or FlashInfer. Exact non-video query rows remain the caller's
responsibility, while every video query sees all non-video keys here.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack
import torch


_compiled = {}


def weighted_video_attention_cute(q, k, v, video_idx, anchor_idx, plan):
    @cute.kernel
    def _tile(
        query: cute.Tensor, key: cute.Tensor, value: cute.Tensor,
        video: cute.Tensor, anchor: cute.Tensor,
        indices: cute.Tensor, shifts: cute.Tensor, output: cute.Tensor,
        nvideo: cutlass.Constexpr, nanchor: cutlass.Constexpr,
        samples: cutlass.Constexpr, qblocks: cutlass.Constexpr,
        qheads: cutlass.Constexpr, kvheads: cutlass.Constexpr,
    ):
        thread, _, _ = cute.arch.thread_idx()
        qb, head, _ = cute.arch.block_idx()
        lane = thread % 32
        warp = thread // 32
        kh = head // (qheads // kvheads)
        scale = Float32(0.08838834764831845)

        for group in range(32):
            row = qb * 128 + group * 4 + warp
            if row < nvideo:
                qpos = video[row]
                qreg = cute.make_rmem_tensor((4,), Float32)
                acc = cute.make_rmem_tensor((4,), Float32)
                for part in cutlass.range_constexpr(4):
                    channel = lane + part * 32
                    qreg[part] = query[0, qpos, head, channel].to(Float32)
                    acc[part] = Float32(0.0)
                maximum = -Float32.inf
                denominator = Float32(0.0)

                for sample in range(samples):
                    kv4 = indices[head, qb, sample]
                    shift = shifts[head, qb, sample].to(Float32)
                    for member in cutlass.range_constexpr(4):
                        kpos = video[kv4 * 4 + member]
                        dot = Float32(0.0)
                        for part in cutlass.range_constexpr(4):
                            channel = lane + part * 32
                            dot += qreg[part] * key[0, kpos, kh, channel].to(Float32)
                        logit = cute.arch.warp_reduction_sum(dot) * scale + shift
                        next_max = cutlass.max(maximum, logit)
                        old = cute.math.exp(maximum - next_max)
                        probability = cute.math.exp(logit - next_max)
                        denominator = denominator * old + probability
                        for part in cutlass.range_constexpr(4):
                            channel = lane + part * 32
                            acc[part] = acc[part] * old + probability * value[0, kpos, kh, channel].to(Float32)
                        maximum = next_max

                for ai in range(nanchor):
                    kpos = anchor[ai]
                    dot = Float32(0.0)
                    for part in cutlass.range_constexpr(4):
                        channel = lane + part * 32
                        dot += qreg[part] * key[0, kpos, kh, channel].to(Float32)
                    logit = cute.arch.warp_reduction_sum(dot) * scale
                    next_max = cutlass.max(maximum, logit)
                    old = cute.math.exp(maximum - next_max)
                    probability = cute.math.exp(logit - next_max)
                    denominator = denominator * old + probability
                    for part in cutlass.range_constexpr(4):
                        channel = lane + part * 32
                        acc[part] = acc[part] * old + probability * value[0, kpos, kh, channel].to(Float32)
                    maximum = next_max

                for part in cutlass.range_constexpr(4):
                    channel = lane + part * 32
                    output[0, qpos, head, channel] = (acc[part] / denominator).to(cutlass.BFloat16)

    @cute.jit
    def _launch(
        query: cute.Tensor, key: cute.Tensor, value: cute.Tensor,
        video: cute.Tensor, anchor: cute.Tensor,
        indices: cute.Tensor, shifts: cute.Tensor, output: cute.Tensor,
        nvideo: cutlass.Constexpr, nanchor: cutlass.Constexpr,
        samples: cutlass.Constexpr, qblocks: cutlass.Constexpr,
        qheads: cutlass.Constexpr, kvheads: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        _tile(
            query, key, value, video, anchor, indices, shifts, output,
            nvideo, nanchor, samples, qblocks, qheads, kvheads,
        ).launch(grid=(qblocks, qheads, 1), block=(128, 1, 1), stream=stream)

    if q.ndim != 4 or q.shape[0] != 1 or q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise ValueError("CuTe AB executor requires BF16 [1,S,Hq,128]")
    if k.shape != v.shape or k.shape[1] != q.shape[1] or q.shape[2] % k.shape[2]:
        raise ValueError("CuTe AB executor requires matching K/V and divisible GQA heads")
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("CuTe AB executor requires Q/K/V on one CUDA device")
    qblocks = math.ceil(video_idx.numel() / 128)
    samples = plan.indices.shape[-1]
    if plan.indices.shape != (q.shape[2], qblocks, samples) or plan.log_weights.shape != plan.indices.shape:
        raise ValueError("SIP plan must be [Hq, ceil(video/128), samples]")
    output = torch.empty_like(q)
    tensors = (
        q.contiguous(), k.contiguous(), v.contiguous(), video_idx.to(torch.int32).contiguous(),
        anchor_idx.to(torch.int32).contiguous(), plan.indices.to(torch.int32).contiguous(),
        plan.log_weights.to(torch.float32).contiguous(), output,
    )
    runtime = tuple(from_dlpack(t) for t in tensors)
    constants = (video_idx.numel(), anchor_idx.numel(), samples, qblocks, q.shape[2], k.shape[2])
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    cache_key = (q.device.index, *constants)
    if cache_key not in _compiled:
        _compiled[cache_key] = cute.compile(_launch, *runtime, *constants, stream)
    _compiled[cache_key](*runtime, stream)
    return output

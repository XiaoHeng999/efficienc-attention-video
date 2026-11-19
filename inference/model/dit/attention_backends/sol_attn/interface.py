"""MagiHuman-native GQA interface for the SM120 Sol-Attn kernel."""

from __future__ import annotations

import functools
import math

import torch

from .preprocess import BLOCK_SIZE, prepare


_COMPILED: dict[tuple, object] = {}


@functools.lru_cache(maxsize=1)
def runtime_available() -> bool:
    """Return whether the optional CUDA/CuTe runtime can be imported."""
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass.cute  # noqa: F401
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, video_tokens: int):
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("Sol-Attn expects Q and matching K/V in BTHD layout")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Sol-Attn requires matching Q/K batch and head dimensions")
    if q.shape[-1] != 128 or q.shape[1] != video_tokens or video_tokens < 1:
        raise ValueError("SM120 Sol-Attn requires nonempty video Q with D=128")
    if q.shape[2] % k.shape[2]:
        raise ValueError("Sol-Attn requires Q heads divisible by KV heads")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise TypeError("SM120 Sol-Attn requires BF16 Q/K/V")
    if not q.is_cuda or k.device != q.device or v.device != q.device:
        raise ValueError("SM120 Sol-Attn requires Q/K/V on the same CUDA device")
    if not all(t.is_contiguous() for t in (q, k, v)):
        raise ValueError("SM120 Sol-Attn requires contiguous BTHD tensors")
    capability = tuple(torch.cuda.get_device_capability(q.device))
    if capability != (12, 0):
        raise RuntimeError(f"This Sol-Attn backend targets SM120, got SM{capability[0]}{capability[1]}")
    if not runtime_available():
        raise RuntimeError(
            "Sol-Attn needs cuda-python, NVIDIA CUTLASS DSL, and Triton; "
            "install requirements-sol-attn-sm120.txt"
        )


def _stream(device: torch.device):
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _to_cute_tensors(tensors):
    from .common import to_cute_tensor

    return [to_cute_tensor(tensor) for tensor in tensors]


def sol_attn_sm120_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    video_tokens: int,
    scale: float | None = None,
    tau: float = 1.0,
) -> torch.Tensor:
    """Run noncausal Sol-Attn for video Q against full multimodal K/V.

    Q may have more heads than K/V.  Audio/text K/V form an exact suffix sink;
    their query rows are handled by the caller's dense attention implementation.
    """
    _validate(q, k, v, video_tokens)
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    batch, q_tokens, q_heads, _ = q.shape
    kv_tokens, kv_heads = k.shape[1:3]
    with torch.cuda.device(q.device):
        kc, vc, threshold = prepare(
            q,
            k,
            v,
            video_tokens=video_tokens,
            tau=float(tau),
            scale=scale,
        )
        output = torch.empty_like(q)
        lse = torch.empty(
            (batch, q_tokens, q_heads),
            device=q.device,
            dtype=torch.float32,
        )
        stream = _stream(q.device)
        sink_start_block = video_tokens // BLOCK_SIZE
        sink_end_block = math.ceil(kv_tokens / BLOCK_SIZE)
        tensors = [q, k, v, output, kc, vc, threshold, lse]
        key = (
            q.device.index,
            batch,
            q_tokens,
            kv_tokens,
            q_heads,
            kv_heads,
        )
        compiled = _COMPILED.get(key)
        args = _to_cute_tensors(tensors)
        if compiled is None:
            import cutlass.cute as cute

            from .sm120 import make_kernel

            compiled = cute.compile(
                make_kernel(),
                *args,
                scale,
                sink_start_block,
                sink_end_block,
                stream=stream,
                options="--enable-tvm-ffi",
            )
            _COMPILED[key] = compiled
        compiled(
            *args,
            scale,
            sink_start_block,
            sink_end_block,
            stream=stream,
        )
    return output


__all__ = ["runtime_available", "sol_attn_sm120_gqa"]

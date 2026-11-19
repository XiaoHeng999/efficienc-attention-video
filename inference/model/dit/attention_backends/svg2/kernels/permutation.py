"""Strided Triton gather/scatter, including GQA KV replication during packing."""
import torch
import triton
import triton.language as tl


@triton.jit
def _permute(X, ORDER, Y, S: tl.constexpr, D: tl.constexpr,
             XH: tl.constexpr, XS: tl.constexpr, XD: tl.constexpr,
             OH: tl.constexpr, OS: tl.constexpr,
             REPEAT: tl.constexpr, INVERSE: tl.constexpr,
             BS: tl.constexpr, BD: tl.constexpr):
    h = tl.program_id(0)
    row = tl.program_id(1) * BS + tl.arange(0, BS)
    d = tl.arange(0, BD)
    source_h = h // REPEAT
    idx = tl.load(ORDER + source_h * OH + row * OS, row < S, other=0)
    src = row if INVERSE else idx
    dst = idx if INVERSE else row
    value = tl.load(X + source_h * XH + src[:, None] * XS + d[None, :] * XD,
                    (row[:, None] < S) & (d[None, :] < D), other=0)
    tl.store(Y + (h * S + dst[:, None]) * D + d[None, :], value,
             (row[:, None] < S) & (d[None, :] < D))


def permute_tokens(x, order, *, repeat=1, inverse=False, out=None):
    """Map [H,S,D] with a bijective [H,S] order; inverse restores Q positions.

    ``out`` is intentionally supported for the FlashInfer path.  The semantic
    permutation is rebuilt every denoising call, but its storage shape is not;
    keeping the large Q/K/V packing buffers alive avoids allocator work without
    retaining an output visible to the model.
    """
    h, s, d = x.shape
    if order.shape != (h, s) or repeat < 1 or (inverse and repeat != 1):
        raise ValueError("Invalid token permutation shape or head repetition")
    expected_shape = (h * repeat, s, d)
    if out is not None and (out.shape != expected_shape or out.dtype != x.dtype or out.device != x.device):
        raise ValueError("Invalid reusable permutation output buffer")
    if not x.is_cuda:
        if inverse:
            result = torch.empty_like(x) if out is None else out
            result.scatter_(1, order[..., None].expand_as(x), x)
        else:
            if repeat == 1:
                index = order[..., None].expand_as(x)
                result = x.gather(1, index) if out is None else torch.gather(x, 1, index, out=out)
            else:
                gathered = x.gather(1, order[..., None].expand_as(x))
                result = gathered.repeat_interleave(repeat, 0, output_size=h * repeat)
                if out is not None:
                    out.copy_(result)
                    result = out
        return result
    result = torch.empty(expected_shape, dtype=x.dtype, device=x.device) if out is None else out
    _permute[(h * repeat, triton.cdiv(s, 32))](
        x, order, result, s, d, *x.stride(), *order.stride(), repeat, inverse,
        32, triton.next_power_of_2(d), num_warps=4)
    return result

# Copyright (c) 2025, Tri Dao.
# SPDX-License-Identifier: BSD-3-Clause
"""Minimal FlashAttention CuTe helpers required by the SM120 Sol kernel."""

from __future__ import annotations

import inspect
import math
from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T, dsl_user_op


_NVVM_FMAX_REQUIRES_RESULT_TYPE = sum(
    parameter.kind
    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    for parameter in inspect.signature(nvvm.fmax).parameters.values()
) >= 3


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if const_expr(isinstance(val, cute.TensorSSA)):
        result = cute.make_rmem_tensor(val.shape, val.dtype)
        result.store(val)
        for index in cutlass.range_constexpr(cute.size(val.shape)):
            result[index] = warp_reduce(result[index], op, width)
        return result.load()
    for index in cutlass.range_constexpr(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << index))
    return val


@dsl_user_op
def fmax(
    a: float | Float32,
    b: float | Float32,
    c: float | Float32 | None = None,
    *,
    loc=None,
    ip=None,
) -> Float32:
    args = (
        Float32(a).ir_value(loc=loc, ip=ip),
        Float32(b).ir_value(loc=loc, ip=ip),
    )
    kwargs = {
        "c": Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
        "loc": loc,
        "ip": ip,
    }
    if _NVVM_FMAX_REQUIRES_RESULT_TYPE:
        return Float32(nvvm.fmax(T.f32(), *args, **kwargs))
    return Float32(nvvm.fmax(*args, **kwargs))


@cute.jit
def fmax_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
    arch: cutlass.Constexpr[int] = 80,
) -> Float32:
    values = cute.make_rmem_tensor(x.shape, Float32)
    values.store(x)
    local_max = [values[0], values[1], values[2], values[3]]
    for index in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        local_max[0] = fmax(local_max[0], values[index])
        local_max[1] = fmax(local_max[1], values[index + 1])
        local_max[2] = fmax(local_max[2], values[index + 2])
        local_max[3] = fmax(local_max[3], values[index + 3])
    local_max[0] = fmax(local_max[0], local_max[1])
    local_max[2] = fmax(local_max[2], local_max[3])
    local_max[0] = fmax(local_max[0], local_max[2])
    return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)


@cute.jit
def fadd_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
    arch: cutlass.Constexpr[int] = 80,
) -> Float32:
    if const_expr(init_val is None):
        init_val = Float32.zero
    return x.reduce(cute.ReductionOp.ADD, init_val, 0)


__all__ = ["fadd_reduce", "fmax_reduce", "warp_reduce"]

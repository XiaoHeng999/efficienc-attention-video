"""Scope the reference SAP GPU-index planner to our wrapper instances only."""
import os
import warnings
from functools import lru_cache


@lru_cache(maxsize=4)
def _gpu_plan(original):
    from .kernels.flashinfer_plan import _build_patched_plan
    patched = _build_patched_plan(original)
    if patched is None:
        warnings.warn("SAP GPU index planner does not match this FlashInfer version; using its original plan",
                      RuntimeWarning, stacklevel=2)
    return original if patched is None else patched


def plan_variable_blocks(wrapper, **kwargs):
    original = type(wrapper).plan
    if os.environ.get("MAGI_SVG2_FLASHINFER_GPU_PLAN", "1") == "1":
        return _gpu_plan(original)(wrapper, **kwargs)
    return original(wrapper, **kwargs)

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) forward pass.
#
# FA4's SM120 forward path retains the SM80-style warp-level pipeline rather
# than B200's tcgen05/TMEM pipeline. This VC fork substitutes SM120 FP8
# m16n8k32 atoms in that pipeline and observes the device's 99 KB SMEM limit.

import cutlass
import cutlass.utils as utils_basic
from cutlass.base_dsl.arch import Arch

from .flash_fwd_vc import FlashAttentionForwardSm80


class FlashAttentionForwardSm120(FlashAttentionForwardSm80):
    def __init__(self, *args, **kwargs):
        """Reuse FA4's warp pipeline while the DSL targets the resident SM120 GPU."""
        super().__init__(*args, **kwargs)
        self.arch = Arch.sm_80

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs=False,
    ) -> bool:
        """Check if the kernel can be implemented on SM120.

        Same logic as SM80 but uses SM120's shared memory capacity (99 KB).
        """
        if dtype != cutlass.Float8E4M3FN:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        # Shared memory usage: Q tile + (K tile + V tile)
        smem_usage_Q = tile_m * head_dim
        smem_usage_K = tile_n * head_dim * num_stages
        smem_usage_V = tile_n * head_dim_v * num_stages  # E4M3 residual
        smem_usage_QV = (
            (smem_usage_Q + smem_usage_V) if not Q_in_regs else max(smem_usage_Q, smem_usage_V)
        )
        smem_usage = smem_usage_QV + smem_usage_K
        # BF16 output staging is live; P stays in registers.
        smem_usage += tile_m * head_dim_v * 2
        # SM120 has 99 KB shared memory (vs 163 KB on SM80)
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        if smem_usage > smem_capacity:
            return False
        if (tile_m * 2) % num_threads != 0:
            return False
        return True

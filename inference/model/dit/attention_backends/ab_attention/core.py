"""MagiHuman adapter and per-generation schedule for AB-Attention."""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from inference.common import Modality
from .planning import SamplePlan, ibcs_scores, sip_sample


@dataclass(frozen=True)
class ABAttentionConfig:
    sample_ratio: float = 0.75
    dense_steps: int = 10
    dense_layers: int = 1
    refresh_steps: int = 4
    min_video_tokens: int = 2048
    min_sample_keys: int = 128
    max_sample_keys: int = 8192
    cpu_plan_threshold: int = 8192
    executor: str = "triton"

    @classmethod
    def from_env(cls):
        return cls(
            executor=os.getenv("MAGI_AB_EXECUTOR", "triton").lower(),
            sample_ratio=float(os.getenv("MAGI_AB_SAMPLE_RATIO", "0.75")),
            dense_steps=int(os.getenv("MAGI_AB_DENSE_STEPS", "10")),
            dense_layers=int(os.getenv("MAGI_AB_DENSE_LAYERS", "1")),
            refresh_steps=int(os.getenv("MAGI_AB_REFRESH_STEPS", "4")),
            min_video_tokens=int(os.getenv("MAGI_AB_MIN_VIDEO_TOKENS", "2048")),
            min_sample_keys=int(os.getenv("MAGI_AB_MIN_SAMPLE_KEYS", "128")),
            max_sample_keys=int(os.getenv("MAGI_AB_MAX_SAMPLE_KEYS", "8192")),
            cpu_plan_threshold=int(os.getenv("MAGI_AB_CPU_PLAN_THRESHOLD", "8192")),
        )

    def __post_init__(self):
        if self.executor not in {"triton", "triton_tile", "cute"}:
            raise ValueError("AB executor must be triton, triton_tile, or cute")
        if not 0 < self.sample_ratio < 1 or self.refresh_steps < 1:
            raise ValueError("AB needs 0 < sample_ratio < 1 and positive refresh_steps")
        if min(self.dense_steps, self.dense_layers, self.min_video_tokens) < 0:
            raise ValueError("AB guard values cannot be negative")
        if self.min_sample_keys < 128:
            raise ValueError("AB needs at least one 128-key execution tile")
        if self.max_sample_keys < self.min_sample_keys or self.max_sample_keys % 128:
            raise ValueError("AB maximum sample keys must be a multiple of 128")


_ACTIVE: ContextVar[tuple["ABAttentionSession", int, str] | None] = ContextVar("ab_step", default=None)


class ABAttentionSession:
    def __init__(self, total_steps: int, config: ABAttentionConfig | None = None):
        self.total_steps = total_steps
        self.config = config or ABAttentionConfig.from_env()
        # One refresh cycle per layer/branch/shape; old plans are replaced.
        self.plans: dict[tuple, tuple[int, SamplePlan]] = {}
        self.sparse_calls = 0

    @contextmanager
    def step(self, index: int, branch: str):
        if not 0 <= index < self.total_steps or not branch:
            raise ValueError("invalid AB denoising step or branch")
        token = _ACTIVE.set((self, index, branch))
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def run(self, q, k, v, modality, dense_attention, *, layer_idx: int):
        scope = _ACTIVE.get()
        if scope is None or scope[0] is not self:
            raise RuntimeError("AB attention requires an active generation step")
        _, step, branch = scope
        config = self.config
        if step < config.dense_steps or layer_idx < config.dense_layers:
            return dense_attention(q, k, v)
        if q.ndim != 4 or q.shape[0] != 1 or k.shape != v.shape or q.shape[1] != k.shape[1]:
            raise ValueError("AB expects Q [1,S,Hq,D] and K/V [1,S,Hkv,D]")
        if q.shape[2] % k.shape[2] or q.shape[-1] != k.shape[-1] or modality.numel() != q.shape[1]:
            raise ValueError("AB requires divisible GQA heads and a full-sequence modality map")
        video_idx = torch.nonzero(modality == int(Modality.VIDEO), as_tuple=True)[0]
        anchor_idx = torch.nonzero(modality != int(Modality.VIDEO), as_tuple=True)[0]
        nvideo = video_idx.numel()
        # The paper's 8192-key budget is invalid on short MagiHuman videos.
        selected_keys = min(
            int(math.floor(nvideo * config.sample_ratio / 128)) * 128,
            config.max_sample_keys,
        )
        samples = selected_keys // 4
        if (nvideo < config.min_video_tokens or nvideo % 4 or q.shape[-1] != 128
                or samples * 4 < config.min_sample_keys or samples >= nvideo // 4
                or not q.is_cuda or q.dtype != torch.bfloat16):
            return dense_attention(q, k, v)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("AB dynamic sampling cannot run inside CUDA graph capture")
        anchor = config.dense_steps + (step - config.dense_steps) // config.refresh_steps * config.refresh_steps
        key = (branch, layer_idx, q.shape, k.shape, nvideo, q.device)
        cached = self.plans.get(key)
        if cached is None or cached[0] != anchor:
            qv = q[0].index_select(0, video_idx).transpose(0, 1).contiguous()
            kv = k[0].index_select(0, video_idx).transpose(0, 1).contiguous()
            scores = ibcs_scores(qv, kv)
            plan = sip_sample(scores, samples, seed=42 + layer_idx * 1000003 + anchor * 97 + (branch == "uncond"))
            if nvideo >= config.cpu_plan_threshold:
                host_plan = SamplePlan(*(
                    torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
                    for t in (plan.indices, plan.log_weights)
                ))
                host_plan.indices.copy_(plan.indices)
                host_plan.log_weights.copy_(plan.log_weights)
                self.plans[key] = (anchor, host_plan)
            else:
                self.plans[key] = (anchor, plan)
        else:
            stored = cached[1]
            plan = (SamplePlan(stored.indices.to(q.device, non_blocking=True),
                               stored.log_weights.to(q.device, non_blocking=True))
                    if stored.indices.device.type == "cpu" else stored)
        self.sparse_calls += 1
        if self.sparse_calls == 1 and os.getenv("MAGI_AB_TRACE") == "1":
            print(f"AB sparse active: step={step}, layer={layer_idx}, video_tokens={nvideo}, "
                  f"sampled_keys={selected_keys}, anchors={anchor_idx.numel()}, "
                  f"executor={config.executor}", flush=True)
        if config.executor == "cute":
            from .cute_sparse import weighted_video_attention_cute
            out = weighted_video_attention_cute(
                q.contiguous(), k.contiguous(), v.contiguous(), video_idx, anchor_idx, plan
            )
        else:
            from .triton_sparse import weighted_video_attention
            out = weighted_video_attention(
                q.contiguous(), k.contiguous(), v.contiguous(), video_idx, anchor_idx, plan,
                query_tile_rows=128 if config.executor == "triton_tile" else 16,
            )
        if anchor_idx.numel():
            # FlashInfer handles exact anchor-query rows with native GQA. Its
            # FA3 prefill API accepts [Q,Hq,D] and [K,Hkv,D]. On SM120 the
            # existing CuTe DSL FA4 path is the tested dense primitive.
            if torch.cuda.get_device_capability(q.device)[0] >= 12:
                exact = dense_attention(q[:, anchor_idx], k, v)[0]
            else:
                import flashinfer
                exact = flashinfer.single_prefill_with_kv_cache(
                    q[0].index_select(0, anchor_idx).contiguous(), k[0].contiguous(), v[0].contiguous(),
                    causal=False, backend="fa3",
                )
            out[0, anchor_idx] = exact
        return out


def scheduled_ab_attention(q, k, v, modality, dense_attention, *, layer_idx: int):
    scope = _ACTIVE.get()
    if scope is None:
        raise RuntimeError("MAGI_ATTN_BACKEND=ab_attention requires ABAttentionSession")
    return scope[0].run(q, k, v, modality, dense_attention, layer_idx=layer_idx)

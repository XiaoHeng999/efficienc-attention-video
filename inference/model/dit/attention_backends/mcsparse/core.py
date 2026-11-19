"""MC-Sparse reference backend adapted for MagiHuman.

The implementation follows the official MC-Sparse algorithm while making the
model-specific contracts explicit: native 40Q/8KV GQA, mixed video/audio/text
tokens, serial CFG branches, and post-Ulysses full-sequence execution.  This is
the correctness/reference executor.  It deliberately uses PyTorch operations
instead of pretending that the official Hopper-only gather-KV kernel is an
SM120 production kernel.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from inference.common import Modality

from .pddp import pddp_group_perm


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


def _env_int_tuple(name: str) -> tuple[int, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    return tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))


def _env_layer_int_map(name: str) -> tuple[tuple[int, int], ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    values: dict[int, int] = {}
    for item in raw.split(","):
        layer, value = item.split(":", 1)
        values[int(layer.strip())] = int(value.strip())
    return tuple(sorted(values.items()))


@dataclass(frozen=True)
class MCSparseConfig:
    density: float = 0.25
    query_tile: int = 128
    kv_tile: int = 128
    pddp_power_iters: int = 4
    selector_kv_chunk: int = 4096
    dense_step_ratio: float = 0.20
    refresh_steps: int = 8
    anchor_steps: tuple[int, ...] = ()
    layer_refresh_steps: tuple[tuple[int, int], ...] = ()
    dense_layers: int = 1
    dense_layer_ids: tuple[int, ...] = ()
    min_video_tokens: int = 4096
    max_reference_tokens: int = 16384
    preserve_nonvideo: bool = True
    video_sink_tokens: int = 0
    residual: bool = True
    pddp: bool = True
    cache_offload: bool = False

    def __post_init__(self):
        if not 0 < self.density <= 1:
            raise ValueError("MC-Sparse density must be in (0, 1]")
        if not 0 <= self.dense_step_ratio <= 1:
            raise ValueError("MC-Sparse dense_step_ratio must be in [0, 1]")
        if min(
            self.query_tile,
            self.kv_tile,
            self.pddp_power_iters,
            self.selector_kv_chunk,
            self.refresh_steps,
        ) < 1:
            raise ValueError("MC-Sparse tile/iteration/chunk/refresh values must be positive")
        if min(
            self.dense_layers,
            self.min_video_tokens,
            self.max_reference_tokens,
            self.video_sink_tokens,
        ) < 0:
            raise ValueError("MC-Sparse layer/token guards must be nonnegative")
        if any(step < 0 for step in self.anchor_steps):
            raise ValueError("MC-Sparse anchor steps must be nonnegative")
        if any(layer < 0 for layer in self.dense_layer_ids):
            raise ValueError("MC-Sparse dense layer ids must be nonnegative")
        if any(layer < 0 or refresh < 1 for layer, refresh in self.layer_refresh_steps):
            raise ValueError("MC-Sparse layer refresh entries must be layer>=0, refresh>=1")

    @classmethod
    def from_env(cls) -> "MCSparseConfig":
        return cls(
            density=float(os.getenv("MAGI_MCSPARSE_DENSITY", "0.25")),
            query_tile=int(os.getenv("MAGI_MCSPARSE_QUERY_TILE", "128")),
            kv_tile=int(os.getenv("MAGI_MCSPARSE_KV_TILE", "128")),
            pddp_power_iters=int(os.getenv("MAGI_MCSPARSE_PDDP_ITERS", "4")),
            selector_kv_chunk=int(os.getenv("MAGI_MCSPARSE_KV_CHUNK", "4096")),
            dense_step_ratio=float(os.getenv("MAGI_MCSPARSE_DENSE_STEP_RATIO", "0.20")),
            refresh_steps=int(os.getenv("MAGI_MCSPARSE_REFRESH_STEPS", "8")),
            anchor_steps=_env_int_tuple("MAGI_MCSPARSE_ANCHOR_STEPS"),
            layer_refresh_steps=_env_layer_int_map(
                "MAGI_MCSPARSE_LAYER_REFRESH_STEPS"
            ),
            dense_layers=int(os.getenv("MAGI_MCSPARSE_DENSE_LAYERS", "1")),
            dense_layer_ids=_env_int_tuple("MAGI_MCSPARSE_DENSE_LAYER_IDS"),
            min_video_tokens=int(os.getenv("MAGI_MCSPARSE_MIN_VIDEO_TOKENS", "4096")),
            max_reference_tokens=int(os.getenv("MAGI_MCSPARSE_MAX_REFERENCE_TOKENS", "16384")),
            preserve_nonvideo=_env_bool("MAGI_MCSPARSE_PRESERVE_NONVIDEO", True),
            video_sink_tokens=int(
                os.getenv("MAGI_MCSPARSE_VIDEO_SINK_TOKENS", "0")
            ),
            residual=_env_bool("MAGI_MCSPARSE_RESIDUAL", True),
            pddp=_env_bool("MAGI_MCSPARSE_PDDP", True),
            cache_offload=_env_bool("MAGI_MCSPARSE_CACHE_OFFLOAD", False),
        )


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[0] != 1:
        raise ValueError("MC-Sparse expects Q/K/V in [1, tokens, heads, dim]")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("MC-Sparse requires matching Q/K head dimensions")
    if q.shape[2] % k.shape[2]:
        raise ValueError("MC-Sparse requires the Q head count to be divisible by KV heads")


def _dense_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Portable dense GQA reference for BSHD tensors."""
    _validate_qkv(q, k, v)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        enable_gqa=q.shape[2] != k.shape[2],
        dropout_p=0.0,
    ).transpose(1, 2)


def _gqa_kv_map(q_heads: int, kv_heads: int, device: torch.device) -> torch.Tensor:
    return torch.arange(q_heads, device=device) // (q_heads // kv_heads)


@torch.no_grad()
def _gqa_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    key_chunk: int,
) -> torch.Tensor:
    """Recompute exact row LSE without materializing the attention matrix."""
    _validate_qkv(q, k, k)
    qh = q[0].transpose(0, 1).float()
    kh = k[0].transpose(0, 1).float()
    q_heads, query_tokens, dim = qh.shape
    key_tokens = kh.shape[1]
    kv_map = _gqa_kv_map(q_heads, kh.shape[0], q.device)
    result = torch.full(
        (q_heads, query_tokens), -torch.inf, device=q.device, dtype=torch.float32
    )
    scale = dim**-0.5
    # Query chunking bounds the temporary QK tile on short and long sequences.
    query_chunk = min(256, query_tokens)
    for q_start in range(0, query_tokens, query_chunk):
        q_part = qh[:, q_start : q_start + query_chunk]
        lse_part = torch.full(
            (q_heads, q_part.shape[1]),
            -torch.inf,
            device=q.device,
            dtype=torch.float32,
        )
        for k_start in range(0, key_tokens, key_chunk):
            k_part = kh[:, k_start : k_start + key_chunk]
            # Advanced indexing is limited to one key chunk instead of expanding
            # all eight MagiHuman KV heads to forty full-length copies.
            scores = torch.bmm(q_part, k_part[kv_map].transpose(1, 2)) * scale
            lse_part = torch.logaddexp(lse_part, torch.logsumexp(scores, dim=-1))
        result[:, q_start : q_start + q_part.shape[1]] = lse_part
    return result


def _normalize_lse(lse: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Normalize FlashAttention LSE to ``[Hq, Sq]`` float32 layout."""
    if lse.ndim == 3:
        if lse.shape[0] != 1:
            raise ValueError(f"Expected batch-one LSE, got {tuple(lse.shape)}")
        lse = lse[0]
    if lse.ndim != 2:
        raise ValueError(f"Expected 2D/3D attention LSE, got {tuple(lse.shape)}")
    expected = (q.shape[2], q.shape[1])
    if tuple(lse.shape) == (q.shape[1], q.shape[2]):
        lse = lse.transpose(0, 1)
    elif tuple(lse.shape) != expected:
        raise ValueError(f"Expected LSE {expected}, got {tuple(lse.shape)}")
    return lse.float().contiguous()


def _video_and_anchor_indices(modality: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    video = torch.nonzero(modality == int(Modality.VIDEO), as_tuple=True)[0]
    anchors = torch.nonzero(modality != int(Modality.VIDEO), as_tuple=True)[0]
    return video, anchors


@torch.no_grad()
def _exact_kv_selection(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    video_idx: torch.Tensor,
    anchor_idx: torch.Tensor,
    query_perm: torch.Tensor,
    config: MCSparseConfig,
) -> tuple[torch.Tensor, float, float]:
    """Select individual video KV tokens by exact grouped attention mass.

    Returns global KV indices in ``[Hq, groups, K]`` layout.  Non-video keys
    are appended exactly when ``preserve_nonvideo`` is enabled.
    """
    qh = q[0].transpose(0, 1).float()
    kh = k[0].transpose(0, 1).float()
    q_heads, _, dim = qh.shape
    kv_map = _gqa_kv_map(q_heads, kh.shape[0], q.device)
    video_tokens = int(video_idx.numel())
    groups = (video_tokens + config.query_tile - 1) // config.query_tile
    sink_count = min(config.video_sink_tokens, video_tokens)
    non_sink_tokens = video_tokens - sink_count
    nominal_video_budget = max(1, math.ceil(config.density * video_tokens))
    aligned_video_budget = (
        (nominal_video_budget + config.kv_tile - 1) // config.kv_tile
    ) * config.kv_tile
    video_budget = min(
        non_sink_tokens,
        aligned_video_budget,
    )
    scale = dim**-0.5
    selected_groups = []
    captured_mass = 0.0
    full_video_mass = 0.0
    query_rows = 0

    for group_idx in range(groups):
        start = group_idx * config.query_tile
        end = min(start + config.query_tile, video_tokens)
        local_q = query_perm[:, start:end]
        absolute_q = video_idx[local_q]
        q_group = qh.gather(1, absolute_q[..., None].expand(-1, -1, dim))
        lse_group = lse.gather(1, absolute_q)

        best_values = torch.empty((q_heads, 0), device=q.device, dtype=torch.float32)
        best_indices = torch.empty((q_heads, 0), device=q.device, dtype=torch.long)
        sink_mass = torch.zeros(q_heads, device=q.device, dtype=torch.float32)
        for key_start in range(0, video_tokens, config.selector_kv_chunk):
            key_end = min(key_start + config.selector_kv_chunk, video_tokens)
            absolute_k = video_idx[key_start:key_end]
            k_chunk = kh[:, absolute_k][kv_map]
            scores = torch.bmm(q_group, k_chunk.transpose(1, 2)) * scale
            mass = torch.exp(scores - lse_group.unsqueeze(-1)).sum(dim=1)
            full_video_mass += float(mass.sum().item())
            local_indices = torch.arange(
                key_start, key_end, device=q.device, dtype=torch.long
            ).expand(q_heads, -1)
            if key_start < sink_count:
                sink_end = min(key_end, sink_count)
                sink_mass += mass[:, : sink_end - key_start].sum(dim=1)
            candidate_mask = local_indices[0] >= sink_count
            if not candidate_mask.any():
                continue
            mass = mass[:, candidate_mask]
            local_indices = local_indices[:, candidate_mask]
            candidate_values = torch.cat((best_values, mass), dim=1)
            candidate_indices = torch.cat((best_indices, local_indices), dim=1)
            keep = min(video_budget, candidate_values.shape[1])
            best_values, positions = torch.topk(
                candidate_values, keep, dim=1, sorted=False
            )
            best_indices = candidate_indices.gather(1, positions)

        chosen = video_idx[best_indices]
        if sink_count:
            chosen = torch.cat(
                (video_idx[:sink_count].expand(q_heads, -1), chosen), dim=1
            )
        if config.preserve_nonvideo and anchor_idx.numel():
            chosen = torch.cat((chosen, anchor_idx.expand(q_heads, -1)), dim=1)
        selected_groups.append(chosen)
        captured_mass += float((best_values.sum() + sink_mass.sum()).item())
        query_rows += q_heads * (end - start)

    selected = torch.stack(selected_groups, dim=1)
    denominator = max(query_rows, 1)
    video_mass = captured_mass / denominator
    # Every key is either video or non-video.  When all non-video keys are
    # mandatory, their exact attention mass is the complement of the full
    # video mass already accumulated by the selector.  This avoids another
    # QK pass over audio/text keys solely for diagnostics.
    mandatory_mass = (
        max(0.0, min(1.0, 1.0 - full_video_mass / denominator))
        if config.preserve_nonvideo and anchor_idx.numel()
        else 0.0
    )
    return selected, video_mass, mandatory_mass


@torch.no_grad()
def _sparse_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    video_idx: torch.Tensor,
    anchor_idx: torch.Tensor,
    query_perm: torch.Tensor,
    selected_indices: torch.Tensor,
    config: MCSparseConfig,
    dense_attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Execute gathered token attention with native GQA storage."""
    qh = q[0].transpose(0, 1)
    kh = k[0].transpose(0, 1)
    vh = v[0].transpose(0, 1)
    q_heads, _, dim = qh.shape
    kv_map = _gqa_kv_map(q_heads, kh.shape[0], q.device)
    output_h = torch.zeros_like(qh)
    video_tokens = int(video_idx.numel())
    groups = selected_indices.shape[1]
    scale = dim**-0.5

    for group_idx in range(groups):
        start = group_idx * config.query_tile
        end = min(start + config.query_tile, video_tokens)
        absolute_q = video_idx[query_perm[:, start:end]]
        q_group = qh.gather(
            1, absolute_q[..., None].expand(-1, -1, dim)
        )
        kv_index = selected_indices[:, group_idx]
        # Pairwise advanced indexing gathers only the selected rows while
        # mapping every Q head to its GQA KV head; it does not expand full K/V.
        keys = kh[kv_map[:, None], kv_index]
        values = vh[kv_map[:, None], kv_index]
        scores = torch.bmm(
            q_group.float(), keys.float().transpose(1, 2)
        ) * scale
        probabilities = torch.softmax(scores, dim=-1)
        group_output = torch.bmm(probabilities, values.float()).to(q.dtype)
        output_h.scatter_(
            1, absolute_q[..., None].expand(-1, -1, dim), group_output
        )

    output = output_h.transpose(0, 1).unsqueeze(0)
    if anchor_idx.numel():
        # Audio/text queries remain exact in the quality-first MagiHuman policy.
        output[:, anchor_idx] = dense_attention(q[:, anchor_idx], k, v)
    return output.contiguous()


@dataclass
class MCSparseCache:
    query_perm: torch.Tensor
    selected_indices: torch.Tensor
    residual_video: torch.Tensor | None
    video_idx: torch.Tensor
    anchor_idx: torch.Tensor


class MCSparseCore:
    """One layer/CFG-branch MC-Sparse cache and reference executor."""

    def __init__(self, config: MCSparseConfig):
        self.config = config
        self.cache: MCSparseCache | None = None
        self.last_mode = "uninitialized"
        self.last_density = 1.0
        self.last_realized_video_density = 1.0
        self.last_video_attention_mass = 1.0
        self.last_mandatory_attention_mass = 0.0
        self.last_total_attention_recall = 1.0
        self.anchor_calls = 0
        self.reuse_calls = 0
        self.dense_fallback_calls = 0

    def _fallback_reason(self, video_tokens: int) -> str | None:
        if video_tokens < self.config.min_video_tokens:
            return f"video_tokens={video_tokens} below min={self.config.min_video_tokens}"
        if self.config.max_reference_tokens and video_tokens > self.config.max_reference_tokens:
            return (
                f"video_tokens={video_tokens} above reference_limit="
                f"{self.config.max_reference_tokens}"
            )
        return None

    def can_run(self, modality: torch.Tensor) -> tuple[bool, str | None]:
        video_tokens = int((modality == int(Modality.VIDEO)).sum().item())
        reason = "sequence has no video tokens" if video_tokens == 0 else self._fallback_reason(video_tokens)
        return reason is None, reason

    def _maybe_offload(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None or not self.config.cache_offload or tensor.device.type == "cpu":
            return tensor
        cpu = tensor.to("cpu", non_blocking=False)
        return cpu.pin_memory() if torch.cuda.is_available() else cpu

    def calibrate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        modality: torch.Tensor,
        dense_attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        dense_attention_with_lse: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | None = None,
    ) -> torch.Tensor:
        _validate_qkv(q, k, v)
        if q.shape[1] != k.shape[1]:
            raise ValueError("MC-Sparse calibration requires self-attention")
        video_idx, anchor_idx = _video_and_anchor_indices(modality)
        allowed, _ = self.can_run(modality)
        if not allowed:
            self.last_mode = "dense_fallback"
            self.dense_fallback_calls += 1
            return dense_attention(q, k, v)

        if dense_attention_with_lse is None:
            dense = dense_attention(q, k, v)
            lse = _gqa_lse(q, k, key_chunk=self.config.selector_kv_chunk)
        else:
            dense, lse = dense_attention_with_lse(q, k, v)
            lse = _normalize_lse(lse, q)
        q_video = q[0, video_idx].transpose(0, 1).contiguous()
        query_perm = (
            pddp_group_perm(
                q_video,
                tile_size=self.config.query_tile,
                power_iters=self.config.pddp_power_iters,
            )
            if self.config.pddp
            else torch.arange(video_idx.numel(), device=q.device)
            .expand(q.shape[2], -1)
            .contiguous()
        )
        selected, video_mass, mandatory_mass = _exact_kv_selection(
            q, k, lse, video_idx, anchor_idx, query_perm, self.config
        )
        if self.config.residual:
            sparse = _sparse_attention_reference(
                q,
                k,
                v,
                video_idx,
                anchor_idx,
                query_perm,
                selected,
                self.config,
                dense_attention,
            )
            residual = (dense[:, video_idx] - sparse[:, video_idx]).contiguous()
        else:
            residual = None
        self.cache = MCSparseCache(
            query_perm=self._maybe_offload(query_perm),
            selected_indices=self._maybe_offload(selected),
            residual_video=self._maybe_offload(residual),
            video_idx=video_idx,
            anchor_idx=anchor_idx,
        )
        self.last_mode = "anchor"
        self.last_density = selected.shape[-1] / q.shape[1]
        selected_video_keys = selected.shape[-1] - (
            anchor_idx.numel() if self.config.preserve_nonvideo else 0
        )
        self.last_realized_video_density = selected_video_keys / video_idx.numel()
        self.last_video_attention_mass = video_mass
        self.last_mandatory_attention_mass = mandatory_mass
        self.last_total_attention_recall = video_mass + mandatory_mass
        self.anchor_calls += 1
        # Anchor output is exact dense attention, as in the paper.
        return dense

    def reuse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dense_attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        _validate_qkv(q, k, v)
        if self.cache is None:
            raise RuntimeError("MC-Sparse reuse requested before an anchor calibration")
        cache = self.cache
        query_perm = cache.query_perm.to(q.device, non_blocking=True)
        selected = cache.selected_indices.to(q.device, non_blocking=True)
        output = _sparse_attention_reference(
            q,
            k,
            v,
            cache.video_idx,
            cache.anchor_idx,
            query_perm,
            selected,
            self.config,
            dense_attention,
        )
        if cache.residual_video is not None:
            output[:, cache.video_idx] += cache.residual_video.to(
                q.device, non_blocking=True
            )
        self.last_mode = "reuse"
        self.reuse_calls += 1
        return output.contiguous()


@dataclass
class _StepScope:
    session: "MCSparseSession"
    index: int
    branch: str


_ACTIVE: ContextVar[_StepScope | None] = ContextVar("mcsparse_step", default=None)


class MCSparseSession:
    """Schedule anchors and isolate metadata by layer and CFG branch."""

    def __init__(self, total_steps: int, config: MCSparseConfig | None = None):
        if total_steps < 1:
            raise ValueError("MC-Sparse total_steps must be positive")
        self.total_steps = total_steps
        self.config = config or MCSparseConfig.from_env()
        self.cores: dict[tuple, MCSparseCore] = {}
        dense_steps = math.ceil(total_steps * self.config.dense_step_ratio)
        self.first_anchor = max(0, dense_steps - 1)

    @contextmanager
    def step(self, index: int, branch: str):
        if not 0 <= index < self.total_steps or not branch:
            raise ValueError("Expected an in-range MC-Sparse step and nonempty branch")
        token = _ACTIVE.set(_StepScope(self, index, branch))
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def _is_dense_layer(self, layer_idx: int) -> bool:
        return (
            layer_idx < self.config.dense_layers
            or layer_idx in self.config.dense_layer_ids
        )

    def _is_anchor(self, index: int, layer_idx: int) -> bool:
        if self.config.anchor_steps:
            return index in self.config.anchor_steps
        refresh = dict(self.config.layer_refresh_steps).get(
            layer_idx, self.config.refresh_steps
        )
        return index >= self.first_anchor and (
            (index - self.first_anchor) % refresh == 0
        )

    def run(
        self,
        q,
        k,
        v,
        modality,
        dense_attention,
        dense_attention_with_lse=None,
        *,
        layer_idx: int,
    ):
        if layer_idx < 0:
            raise ValueError("MC-Sparse layer_idx must be nonnegative")
        scope = _ACTIVE.get()
        assert scope is not None and scope.session is self
        if self._is_dense_layer(layer_idx) or scope.index < self.first_anchor:
            return dense_attention(q, k, v)
        if q.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MC-Sparse dynamic metadata is incompatible with CUDA Graph capture")

        video_count = int((modality == int(Modality.VIDEO)).sum().item())
        key = (
            scope.branch,
            layer_idx,
            tuple(q.shape),
            tuple(k.shape),
            video_count,
            q.device,
        )
        core = self.cores.get(key)
        if core is None:
            core = MCSparseCore(self.config)
            self.cores[key] = core
        allowed, _ = core.can_run(modality)
        if not allowed:
            core.last_mode = "dense_fallback"
            core.dense_fallback_calls += 1
            return dense_attention(q, k, v)
        if self._is_anchor(scope.index, layer_idx) or core.cache is None:
            return core.calibrate(
                q,
                k,
                v,
                modality,
                dense_attention,
                dense_attention_with_lse,
            )
        return core.reuse(q, k, v, dense_attention)


def scheduled_mcsparse_attention(
    q,
    k,
    v,
    modality,
    dense_attention,
    dense_attention_with_lse=None,
    *,
    layer_idx: int,
):
    scope = _ACTIVE.get()
    if scope is not None:
        return scope.session.run(
            q,
            k,
            v,
            modality,
            dense_attention,
            dense_attention_with_lse,
            layer_idx=layer_idx,
        )
    config = MCSparseConfig.from_env()
    if layer_idx < config.dense_layers or layer_idx in config.dense_layer_ids:
        return dense_attention(q, k, v)
    return MCSparseCore(config).calibrate(
        q, k, v, modality, dense_attention, dense_attention_with_lse
    )


__all__ = [
    "MCSparseCache",
    "MCSparseConfig",
    "MCSparseCore",
    "MCSparseSession",
    "_dense_gqa",
    "_exact_kv_selection",
    "_gqa_lse",
    "_normalize_lse",
    "_sparse_attention_reference",
    "scheduled_mcsparse_attention",
]

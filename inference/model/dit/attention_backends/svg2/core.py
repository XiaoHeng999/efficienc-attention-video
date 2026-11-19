"""SVG2 clustering and attention backend selection for MagiHuman inference.

The FlexAttention executor handles all heads in one sparse attention call.
The SDPA executor remains as a simple variable-block numerical reference.

Algorithm reference: https://arxiv.org/abs/2505.18875
Code reference: https://github.com/svg-project/Sparse-VideoGen
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from inference.common import Modality

from .profile import stage
from .clustering import kmeans as _kmeans


_DENSITY_OBSERVER: ContextVar[Callable | None] = ContextVar("svg2_density_observer", default=None)
_INPUT_OBSERVER: ContextVar[Callable | None] = ContextVar("svg2_input_observer", default=None)


@contextmanager
def observe_svg2_density(observer: Callable):
    """Collect selected cluster maps during a real inference run."""
    token = _DENSITY_OBSERVER.set(observer)
    try:
        yield
    finally:
        _DENSITY_OBSERVER.reset(token)


@contextmanager
def observe_svg2_inputs(observer: Callable):
    """Capture selected post-RoPE Q/K/V calls for sparse-quality diagnosis."""
    token = _INPUT_OBSERVER.set(observer)
    try:
        yield
    finally:
        _INPUT_OBSERVER.reset(token)


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class SVG2Config:
    q_clusters: int = 16
    k_clusters: int = 32
    top_p: float = 0.90
    kmeans_init_iters: int = 50
    kmeans_iters: int = 2
    min_kc_ratio: float = 0.10
    dense_step_ratio: float = 0.30
    dense_layers: int = 1
    min_video_tokens: int = 1024
    executor: str = "flex"
    block_size: int = 64
    repair_threshold: float = 0.0
    zero_step_kmeans_init: bool = False

    def __post_init__(self):
        if not 0 <= self.repair_threshold <= 1:
            raise ValueError("SVG2 repair_threshold must be in [0, 1]")
        if not 0 < self.top_p <= 1:
            raise ValueError("SVG2 top_p must be in (0, 1]")
        if not 0 <= self.dense_step_ratio <= 1:
            raise ValueError("SVG2 dense_step_ratio must be in [0, 1]")
        if self.dense_layers < 0:
            raise ValueError("SVG2 dense_layers must be nonnegative")
        if min(self.q_clusters, self.k_clusters, self.kmeans_init_iters, self.kmeans_iters) < 1:
            raise ValueError("SVG2 cluster counts and iterations must be positive")
        if not 0 <= self.min_kc_ratio <= 1:
            raise ValueError("SVG2 min_kc_ratio must be in [0, 1]")
        if self.min_video_tokens < 0:
            raise ValueError("SVG2 min_video_tokens must be nonnegative")
        if self.executor not in {"flex", "flashinfer", "sdpa"}:
            raise ValueError("SVG2 executor must be 'flex', 'flashinfer', or 'sdpa'")
        if self.block_size not in {64, 128}:
            raise ValueError("SVG2 FlexAttention block_size must be 64 or 128")

    @classmethod
    def from_env(cls) -> SVG2Config:
        return cls(
            q_clusters=_env_int("MAGI_SVG2_Q_CLUSTERS", 16),
            k_clusters=_env_int("MAGI_SVG2_K_CLUSTERS", 32),
            top_p=float(os.environ.get("MAGI_SVG2_TOP_P", "0.90")),
            kmeans_init_iters=_env_int("MAGI_SVG2_KMEANS_INIT_ITERS", 50),
            kmeans_iters=_env_int("MAGI_SVG2_KMEANS_ITERS", 2),
            min_kc_ratio=float(os.environ.get("MAGI_SVG2_MIN_KC_RATIO", "0.10")),
            dense_step_ratio=float(os.environ.get("MAGI_SVG2_DENSE_STEP_RATIO", "0.30")),
            dense_layers=int(os.environ.get("MAGI_SVG2_DENSE_LAYERS", "1")),
            min_video_tokens=int(os.environ.get("MAGI_SVG2_MIN_VIDEO_TOKENS", "1024")),
            executor=os.environ.get("MAGI_SVG2_EXECUTOR", "flex").lower(),
            block_size=int(os.environ.get("MAGI_SVG2_BLOCK_SIZE", "64")),
            repair_threshold=float(os.environ.get("MAGI_SVG2_REPAIR_THRESHOLD", "0")),
            zero_step_kmeans_init=os.environ.get("MAGI_SVG2_ZERO_STEP_KMEANS_INIT", "0") == "1",
        )


def _top_p_map(
    q_centers: torch.Tensor,
    k_centers: torch.Tensor,
    k_counts: torch.Tensor,
    top_p: float,
    min_kc_ratio: float = 0.0,
) -> torch.Tensor:
    """SVG2 cluster mass: softmax(q_center @ k_center + log(cluster size))."""
    q_heads, _, dim = q_centers.shape
    kv_heads = k_centers.shape[0]
    group = q_heads // kv_heads
    kv_head_for_q = torch.arange(q_heads, device=q_centers.device) // group
    selected_k_centers = k_centers[kv_head_for_q]
    selected_k_counts = k_counts[kv_head_for_q]
    if top_p == 1:
        return (selected_k_counts[:, None, :] > 0).expand(
            -1, q_centers.shape[1], -1
        ).clone()
    scores = torch.bmm(q_centers, selected_k_centers.transpose(1, 2)) / math.sqrt(dim)
    # Preserve SAP rounding: weighted softmax accumulates in FP32 and casts
    # probabilities back to the score dtype before sort/cumsum.
    scores_fp32 = scores.float()
    weights = selected_k_counts[:, None].float()
    weighted_exp = weights * torch.exp(scores_fp32 - scores_fp32.amax(-1, keepdim=True))
    probabilities = (weighted_exp / weighted_exp.sum(-1, keepdim=True).clamp_min(1e-12)).to(scores.dtype)
    sorted_prob, sorted_idx = probabilities.sort(dim=-1, descending=True)
    remove = sorted_prob.cumsum(-1) > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    min_clusters = int(min_kc_ratio * k_centers.shape[1])
    if min_clusters:
        remove[..., :min_clusters] = False
    keep_sorted = ~remove

    return torch.zeros_like(keep_sorted).scatter_(-1, sorted_idx, keep_sorted)


def _dense_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Dense reference for selected non-video queries, without replicating K/V."""
    _, _, q_heads, _ = q.shape
    kv_heads = k.shape[2]
    group = q_heads // kv_heads
    output = torch.empty_like(q)
    for kv_head in range(kv_heads):
        q_group = q[:, :, kv_head * group : (kv_head + 1) * group].transpose(1, 2)
        k_group = k[:, :, kv_head : kv_head + 1].transpose(1, 2)
        v_group = v[:, :, kv_head : kv_head + 1].transpose(1, 2)
        output[:, :, kv_head * group : (kv_head + 1) * group] = F.scaled_dot_product_attention(
            q_group, k_group, v_group, dropout_p=0.0
        ).transpose(1, 2)
    return output


def svg2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    modality: torch.Tensor,
    config: SVG2Config,
    *,
    initial_centers: tuple[torch.Tensor, torch.Tensor] | None = None,
    dense_attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    modality_partition: tuple[torch.Tensor, torch.Tensor] | None = None,
    flashinfer_executor=None,
    profile=None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Cluster video Q/K independently; keep audio/text connections dense.

    Q and K are post-RoPE [1, sequence, heads, dim]. Q heads in a GQA group
    share one KV clustering and permutation, but select KV clusters separately.
    """
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[0] != 1:
        raise ValueError("SVG2 expects Q [1,S,Hq,D] and matching K/V [1,S,Hkv,D]")
    _, length, q_heads, dim = q.shape
    kv_heads = k.shape[2]
    if k.shape[1] != length or k.shape[-1] != dim or q_heads % kv_heads:
        raise ValueError("SVG2 requires equal Q/K token lengths and divisible GQA heads")
    if modality.ndim != 1 or modality.numel() != length:
        raise ValueError("SVG2 modality map must follow the full, post-CP sequence")
    dense_attention = dense_attention or _dense_gqa
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SVG2 dynamic clustering and routing cannot run in CUDA Graph capture")
    if modality_partition is None:
        with stage(profile, "partition"):
            video_idx = torch.nonzero(modality == int(Modality.VIDEO), as_tuple=True)[0]
            anchor_idx = torch.nonzero(modality != int(Modality.VIDEO), as_tuple=True)[0]
    else:
        video_idx, anchor_idx = modality_partition
    if video_idx.numel() < config.min_video_tokens or video_idx.numel() == 0:
        return dense_attention(q, k, v), None
    input_observer = _INPUT_OBSERVER.get()
    if input_observer is not None:
        input_observer(q, k, v, modality, config, initial_centers, _ACTIVE.get())
    with stage(profile, "gather_video"):
        q_video = q[0, video_idx].transpose(0, 1).contiguous()
        k_video = k[0, video_idx].transpose(0, 1).contiguous()
    q_initial, k_initial = initial_centers if initial_centers is not None else (None, None)
    with stage(profile, "q_kmeans"):
        q_iters = config.kmeans_iters if q_initial is not None else config.kmeans_init_iters
        q_labels, q_centers, q_counts = _kmeans(q_video, config.q_clusters, q_iters, q_initial)
    with stage(profile, "k_kmeans"):
        k_iters = config.kmeans_iters if k_initial is not None else config.kmeans_init_iters
        k_labels, k_centers, k_counts = _kmeans(k_video, config.k_clusters, k_iters, k_initial)
    with stage(profile, "top_p_map"):
        cluster_map = _top_p_map(q_centers, k_centers, k_counts, config.top_p, config.min_kc_ratio)
    if config.repair_threshold > 0 and config.top_p < 1:
        from .repair import repair_query_routes
        with stage(profile, "query_route_repair"):
            q_labels, q_counts, cluster_map = repair_query_routes(
                q_video, q_labels, k_centers, k_counts, cluster_map, config.repair_threshold
            )
    observer = _DENSITY_OBSERVER.get()
    if observer is not None:
        observer(q_counts, k_counts, cluster_map, video_idx.numel(), anchor_idx.numel(), _ACTIVE.get())

    with stage(profile, "sort_labels"):
        q_order = torch.argsort(q_labels, dim=-1, stable=True)
        k_order = torch.argsort(k_labels, dim=-1, stable=True)
    if config.executor == "flex":
        from .flex import svg2_flex_attention

        output = svg2_flex_attention(
            q, k, v, video_idx, anchor_idx,
            q_order, q_labels, k_order, k_labels,
            cluster_map, config.block_size, profile=profile,
        )
        return output, (q_centers, k_centers)
    if config.executor == "flashinfer":
        from .flashinfer import svg2_flashinfer_attention

        output = svg2_flashinfer_attention(
            q, k, v, video_idx, anchor_idx,
            q_order, k_order, q_counts, k_counts, cluster_map,
            executor=flashinfer_executor,
            profile=profile,
        )
        return output, (q_centers, k_centers)

    with stage(profile, "sdpa_pack"):
        v_video = v[0, video_idx].transpose(0, 1).contiguous()
        q_sorted = q_video.gather(1, q_order[..., None].expand_as(q_video))
        k_sorted = k_video.gather(1, k_order[..., None].expand_as(k_video))
        v_sorted = v_video.gather(1, k_order[..., None].expand_as(v_video))
    with stage(profile, "sdpa_host_metadata"):
        q_counts_host = q_counts.cpu().tolist()
        k_counts_host = k_counts.cpu().tolist()
        selected_host = cluster_map.cpu().tolist()

    with stage(profile, "sdpa_attention"):
        output = torch.empty_like(q)
        if anchor_idx.numel():
            output[:, anchor_idx] = dense_attention(q[:, anchor_idx], k, v)
        group = q_heads // kv_heads
        for head in range(q_heads):
            kv_head = head // group
            q_start = 0
            k_starts = [0]
            for count in k_counts_host[kv_head]:
                k_starts.append(k_starts[-1] + count)
            for q_cluster, q_count in enumerate(q_counts_host[head]):
                q_end = q_start + q_count
                if q_count:
                    k_pieces = [
                        k_sorted[kv_head, k_starts[c] : k_starts[c + 1]]
                        for c, chosen in enumerate(selected_host[head][q_cluster])
                        if chosen and k_counts_host[kv_head][c]
                    ]
                    v_pieces = [
                        v_sorted[kv_head, k_starts[c] : k_starts[c + 1]]
                        for c, chosen in enumerate(selected_host[head][q_cluster])
                        if chosen and k_counts_host[kv_head][c]
                    ]
                    if anchor_idx.numel():
                        k_pieces.append(k[0, anchor_idx, kv_head])
                        v_pieces.append(v[0, anchor_idx, kv_head])
                    q_block = q_sorted[head, q_start:q_end][None, None]
                    k_block = torch.cat(k_pieces, dim=0)[None, None]
                    v_block = torch.cat(v_pieces, dim=0)[None, None]
                    block_out = F.scaled_dot_product_attention(
                        q_block, k_block, v_block, dropout_p=0.0
                    )[0, 0]
                    # Scatter from the semantic permutation to original token order.
                    output[0, video_idx[q_order[head, q_start:q_end]], head] = block_out
                q_start = q_end
    return output, (q_centers, k_centers)


@dataclass
class _StepScope:
    session: SVG2Session
    index: int
    branch: str
    call_index: int = 0
    layer_idx: int | None = None
    modality_partitions: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


_ACTIVE: ContextVar[_StepScope | None] = ContextVar("svg2_attention_step", default=None)


class SVG2Session:
    """Protect early steps/shallow layers; cache centroids per branch and real layer."""

    def __init__(self, total_steps: int, config: SVG2Config | None = None):
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.total_steps = total_steps
        self.config = config or SVG2Config.from_env()
        self.centers: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        # Created lazily: only FlashInfer needs persistent dynamic-plan state.
        self._flashinfer_executor = None

    def _get_flashinfer_executor(self):
        if self._flashinfer_executor is None:
            from .flashinfer import FlashInferSVG2Executor
            self._flashinfer_executor = FlashInferSVG2Executor()
        return self._flashinfer_executor

    @contextmanager
    def step(self, index: int, branch: str):
        if not 0 <= index < self.total_steps or not branch:
            raise ValueError("Expected an in-range denoising step and branch")
        token = _ACTIVE.set(_StepScope(self, index, branch))
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def run(self, q, k, v, modality, dense_attention, *, layer_idx: int):
        if layer_idx < 0:
            raise ValueError("SVG2 requires a nonnegative, real model layer_idx")
        scope = _ACTIVE.get()
        assert scope is not None and scope.session is self
        scope.call_index += 1
        scope.layer_idx = layer_idx
        full_attention = (
            scope.index < math.floor(self.total_steps * self.config.dense_step_ratio)
            or layer_idx < self.config.dense_layers
        )
        if full_attention and not self.config.zero_step_kmeans_init:
            return dense_attention(q, k, v)
        if q.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SVG2 dynamic clustering and routing cannot run in CUDA Graph capture")
        partition_key = (modality.data_ptr(), modality.numel(), modality.device)
        partition = scope.modality_partitions.get(partition_key)
        if partition is None:
            partition = (
                torch.nonzero(modality == int(Modality.VIDEO), as_tuple=True)[0],
                torch.nonzero(modality != int(Modality.VIDEO), as_tuple=True)[0],
            )
            scope.modality_partitions[partition_key] = partition
        key = (scope.branch, layer_idx, tuple(q.shape), tuple(k.shape), partition[0].numel(), q.device)
        if full_attention:
            # Hunyuan SAP's optional warmup clustering also updates protected
            # shallow layers; it never changes their dense output.
            video_idx = partition[0]
            if video_idx.numel() and video_idx.numel() >= self.config.min_video_tokens:
                cached = self.centers.get(key)
                iterations = self.config.kmeans_init_iters if cached is None else self.config.kmeans_iters
                _, qc, _ = _kmeans(q[0, video_idx].transpose(0, 1).contiguous(),
                                  self.config.q_clusters, iterations, None if cached is None else cached[0])
                _, kc, _ = _kmeans(k[0, video_idx].transpose(0, 1).contiguous(),
                                  self.config.k_clusters, iterations, None if cached is None else cached[1])
                self.centers[key] = (qc, kc)
            return dense_attention(q, k, v)
        output, centers = svg2_attention(
            q, k, v, modality, self.config,
            initial_centers=self.centers.get(key), dense_attention=dense_attention,
            modality_partition=partition,
            flashinfer_executor=(
                self._get_flashinfer_executor() if self.config.executor == "flashinfer" else None
            ),
        )
        if centers is not None:
            self.centers[key] = centers
        return output


def scheduled_svg2_attention(q, k, v, modality, dense_attention, *, layer_idx: int):
    if layer_idx < 0:
        raise ValueError("SVG2 requires a nonnegative, real model layer_idx")
    scope = _ACTIVE.get()
    if scope is not None:
        return scope.session.run(q, k, v, modality, dense_attention, layer_idx=layer_idx)
    config = SVG2Config.from_env()
    if layer_idx < config.dense_layers:
        return dense_attention(q, k, v)
    return svg2_attention(q, k, v, modality, config, dense_attention=dense_attention)[0]

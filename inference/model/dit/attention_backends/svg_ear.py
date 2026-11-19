"""Standalone SVG-EAR reference attention backend for MagiHuman.

SVG-EAR keeps high-error semantic blocks at token resolution and replaces the
remaining video KV clusters with a parameter-free centroid.  This module is
intentionally self-contained: it does not import or share state with the SVG2
backend.  The executor is a correctness-oriented PyTorch implementation; a
fused sparse kernel can later consume the same ``route`` produced here.

Input tensors use MagiHuman's post-RoPE layout ``[1, sequence, heads, dim]``.
Video tokens are routed sparsely, while every audio/text token remains an exact
anchor.  Native KV heads are retained for GQA instead of materialising repeated
K/V tensors.
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


_ROUTE_OBSERVER: ContextVar[Callable | None] = ContextVar(
    "svg_ear_route_observer", default=None
)
_INPUT_OBSERVER: ContextVar[Callable | None] = ContextVar(
    "svg_ear_input_observer", default=None
)


@contextmanager
def observe_svg_ear_routes(observer: Callable):
    """Observe route tensors without sharing state with another backend."""
    token = _ROUTE_OBSERVER.set(observer)
    try:
        yield
    finally:
        _ROUTE_OBSERVER.reset(token)


@contextmanager
def observe_svg_ear_inputs(observer: Callable):
    """Capture selected post-RoPE calls for fixed-QKV quality analysis."""
    token = _INPUT_OBSERVER.set(observer)
    try:
        yield
    finally:
        _INPUT_OBSERVER.reset(token)


@contextmanager
def _profile_stage(profile, name: str):
    if profile is None:
        yield
    else:
        with profile(name):
            yield


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class SVGEARConfig:
    """Configuration isolated under the ``MAGI_SVG_EAR_*`` namespace."""

    # The Sparse-VideoGen EAR presets use hundreds of Q clusters and about one
    # thousand K clusters.  Real MagiHuman captures confirmed that the much
    # smaller 16/32 exploratory setting is not accurate enough (28.2% relative
    # L2 at top_p=0.9), while 300/1000 reduces it to 0.28% at similar density.
    q_clusters: int = 300
    k_clusters: int = 1000
    top_p: float = 0.90
    gamma: float = 1.0
    min_kc_ratio: float = 0.05
    # Two iterations are the measured MagiHuman reference preset.  Increasing
    # this remains available for convergence studies through the environment.
    kmeans_init_iters: int = 2
    kmeans_iters: int = 2
    kmeans_chunk_size: int = 4096
    score_q_chunk_size: int = 8
    executor_q_chunk_size: int = 512
    executor: str = "auto"
    flashinfer_cache_states: int = 2
    dense_step_ratio: float = 0.30
    dense_layers: int = 1
    min_video_tokens: int = 1024
    warmup_kmeans: bool = False

    def __post_init__(self) -> None:
        if min(
            self.q_clusters,
            self.k_clusters,
            self.kmeans_init_iters,
            self.kmeans_iters,
            self.kmeans_chunk_size,
            self.score_q_chunk_size,
            self.executor_q_chunk_size,
            self.flashinfer_cache_states,
        ) < 1:
            raise ValueError("SVG-EAR cluster counts, iterations, and chunks must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("SVG-EAR top_p must be in (0, 1]")
        if self.gamma < 0:
            raise ValueError("SVG-EAR gamma must be nonnegative")
        if not 0 <= self.min_kc_ratio <= 1:
            raise ValueError("SVG-EAR min_kc_ratio must be in [0, 1]")
        if not 0 <= self.dense_step_ratio <= 1:
            raise ValueError("SVG-EAR dense_step_ratio must be in [0, 1]")
        if self.dense_layers < 0 or self.min_video_tokens < 0:
            raise ValueError("SVG-EAR dense_layers/min_video_tokens must be nonnegative")
        if self.executor not in {"auto", "reference", "flashinfer"}:
            raise ValueError("SVG-EAR executor must be auto/reference/flashinfer")

    @classmethod
    def from_env(cls) -> "SVGEARConfig":
        return cls(
            q_clusters=_positive_env_int("MAGI_SVG_EAR_Q_CLUSTERS", 300),
            k_clusters=_positive_env_int("MAGI_SVG_EAR_K_CLUSTERS", 1000),
            top_p=float(os.environ.get("MAGI_SVG_EAR_TOP_P", "0.90")),
            gamma=float(os.environ.get("MAGI_SVG_EAR_GAMMA", "1.0")),
            min_kc_ratio=float(os.environ.get("MAGI_SVG_EAR_MIN_KC_RATIO", "0.05")),
            kmeans_init_iters=_positive_env_int("MAGI_SVG_EAR_KMEANS_INIT_ITERS", 2),
            kmeans_iters=_positive_env_int("MAGI_SVG_EAR_KMEANS_ITERS", 2),
            kmeans_chunk_size=_positive_env_int("MAGI_SVG_EAR_KMEANS_CHUNK_SIZE", 4096),
            score_q_chunk_size=_positive_env_int("MAGI_SVG_EAR_SCORE_Q_CHUNK_SIZE", 8),
            executor_q_chunk_size=_positive_env_int(
                "MAGI_SVG_EAR_EXECUTOR_Q_CHUNK_SIZE", 512
            ),
            executor=os.environ.get("MAGI_SVG_EAR_EXECUTOR", "auto").lower(),
            flashinfer_cache_states=_positive_env_int(
                "MAGI_SVG_EAR_FLASHINFER_CACHE_STATES", 2
            ),
            dense_step_ratio=float(os.environ.get("MAGI_SVG_EAR_DENSE_STEP_RATIO", "0.30")),
            dense_layers=int(os.environ.get("MAGI_SVG_EAR_DENSE_LAYERS", "1")),
            min_video_tokens=int(os.environ.get("MAGI_SVG_EAR_MIN_VIDEO_TOKENS", "1024")),
            warmup_kmeans=os.environ.get("MAGI_SVG_EAR_WARMUP_KMEANS", "0") == "1",
        )


def _dense_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Dense SDPA reference without physically repeating native GQA K/V heads."""
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    if q_heads % kv_heads:
        raise ValueError("SVG-EAR requires Q heads divisible by KV heads")
    group = q_heads // kv_heads
    output = torch.empty_like(q)
    for kv_head in range(kv_heads):
        q_slice = slice(kv_head * group, (kv_head + 1) * group)
        output[:, :, q_slice] = F.scaled_dot_product_attention(
            q[:, :, q_slice].transpose(1, 2),
            k[:, :, kv_head : kv_head + 1].transpose(1, 2),
            v[:, :, kv_head : kv_head + 1].transpose(1, 2),
            dropout_p=0.0,
        ).transpose(1, 2)
    return output


def _assign_labels(tokens: torch.Tensor, centers: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Assign ``[H,N,D]`` tokens in bounded chunks using squared L2 distance."""
    heads, tokens_count, _ = tokens.shape
    labels = torch.empty((heads, tokens_count), dtype=torch.long, device=tokens.device)
    centers_f = centers.float()
    center_norm = centers_f.square().sum(-1)[:, None]
    for start in range(0, tokens_count, chunk_size):
        block = tokens[:, start : start + chunk_size].float()
        distance = (
            block.square().sum(-1)[..., None]
            + center_norm
            - 2.0 * torch.bmm(block, centers_f.transpose(1, 2))
        )
        labels[:, start : start + block.shape[1]] = distance.argmin(-1)
    return labels


def _kmeans(
    tokens: torch.Tensor,
    clusters: int,
    iterations: int,
    chunk_size: int,
    initial_centers: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small deterministic batched K-means used only by the SVG-EAR backend."""
    if tokens.ndim != 3 or tokens.shape[1] < clusters:
        raise ValueError("SVG-EAR K-means expects [heads,tokens,dim] with tokens >= clusters")
    heads, token_count, dim = tokens.shape
    if initial_centers is None:
        indices = torch.linspace(0, token_count - 1, clusters, device=tokens.device).long()
        centers = tokens[:, indices].float().clone()
    else:
        if initial_centers.shape != (heads, clusters, dim):
            raise ValueError("SVG-EAR cached centroid shape does not match the current call")
        centers = initial_centers.float().clone()

    for _ in range(iterations):
        labels = _assign_labels(tokens, centers, chunk_size)
        sums = torch.zeros((heads, clusters, dim), dtype=torch.float32, device=tokens.device)
        sums.scatter_add_(1, labels[..., None].expand(-1, -1, dim), tokens.float())
        counts = torch.zeros((heads, clusters), dtype=torch.long, device=tokens.device)
        counts.scatter_add_(1, labels, torch.ones_like(labels))
        means = sums / counts.clamp_min(1)[..., None]
        centers = torch.where((counts > 0)[..., None], means, centers)

    return labels, centers.to(tokens.dtype), counts


def _cluster_means(
    values: torch.Tensor, labels: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    heads, _, dim = values.shape
    clusters = counts.shape[1]
    sums = torch.zeros((heads, clusters, dim), dtype=torch.float32, device=values.device)
    sums.scatter_add_(1, labels[..., None].expand(-1, -1, dim), values.float())
    return (sums / counts.clamp_min(1)[..., None]).to(values.dtype)


def _error_efficiency_scores(
    q_centers: torch.Tensor,
    q_counts: torch.Tensor,
    k_tokens: torch.Tensor,
    v_tokens: torch.Tensor,
    k_labels: torch.Tensor,
    k_centers: torch.Tensor,
    v_centers: torch.Tensor,
    k_counts: torch.Tensor,
    gamma: float,
    q_chunk_size: int,
) -> torch.Tensor:
    """Compute SVG-EAR's value-error per unit block area.

    For a query centroid and KV cluster, the numerator estimates the squared
    output error of replacing every token ``(k_i, v_i)`` by the cluster
    centroid ``(k_c, v_c)``.  Larger scores are routed to exact token attention.
    """
    q_heads, q_clusters, dim = q_centers.shape
    kv_heads, _, _ = k_tokens.shape
    if q_heads % kv_heads:
        raise ValueError("SVG-EAR score construction requires valid GQA groups")
    group = q_heads // kv_heads
    scores = torch.full(
        (q_heads, q_clusters, k_centers.shape[1]),
        -float("inf"),
        dtype=torch.float32,
        device=q_centers.device,
    )
    del q_chunk_size  # The grouped formulation is already bounded by GQA*Q-clusters.
    scale = dim**-0.5
    k_clusters = k_centers.shape[1]
    for kv_head in range(kv_heads):
        head_start = kv_head * group
        head_end = head_start + group
        queries = q_centers[head_start:head_end].float().reshape(-1, dim)
        keys = k_tokens[kv_head].float()
        values = v_tokens[kv_head].float()
        labels = k_labels[kv_head]
        centers_k = k_centers[kv_head].float()
        centers_v = v_centers[kv_head].float()
        rows = queries.shape[0]
        expanded_labels = labels[None].expand(rows, -1)

        token_logits = queries @ keys.transpose(0, 1) * scale
        center_logits = queries @ centers_k.transpose(0, 1) * scale
        cluster_max = torch.full(
            (rows, k_clusters),
            -float("inf"),
            dtype=torch.float32,
            device=queries.device,
        )
        cluster_max.scatter_reduce_(
            1, expanded_labels, token_logits, reduce="amax", include_self=True
        )
        shift = torch.maximum(cluster_max, center_logits)
        token_shift = torch.gather(shift, 1, expanded_labels)
        token_weight = (token_logits - token_shift).exp()
        center_weight = (center_logits - shift).exp()

        value_norm = values.square().sum(-1)
        token_center_dot = (values * centers_v.index_select(0, labels)).sum(-1)
        center_norm = centers_v.square().sum(-1)
        center_weight_for_token = torch.gather(
            center_weight, 1, expanded_labels
        )
        token_error = (
            token_weight.square() * value_norm[None]
            - 2.0
            * token_weight
            * center_weight_for_token
            * token_center_dot[None]
            + center_weight_for_token.square()
            * center_norm.index_select(0, labels)[None]
        )
        error_sum = torch.zeros_like(center_weight)
        error_sum.scatter_add_(1, expanded_labels, token_error)
        # A singleton is represented by its exact K/V token, so its
        # replacement error is mathematically zero.  Enforce that identity
        # instead of letting cancellation in the expanded norm perturb ranks.
        error_sum[:, k_counts[kv_head] == 1] = 0
        log_error = error_sum.clamp_min_(1.0e-30).log() + 2.0 * shift

        area = (
            q_counts[head_start:head_end].float().reshape(-1, 1)
            * k_counts[kv_head].float()[None]
        )
        efficiency = log_error - gamma * area.clamp_min(1).log()
        valid = (area > 0) & torch.isfinite(cluster_max)
        scores[head_start:head_end] = torch.where(
            valid, efficiency, -float("inf")
        ).reshape(group, q_clusters, k_clusters)
    return scores


def _error_aware_route(
    scores: torch.Tensor,
    q_centers: torch.Tensor,
    k_centers: torch.Tensor,
    q_counts: torch.Tensor,
    k_counts: torch.Tensor,
    top_p: float,
    min_kc_ratio: float,
) -> torch.Tensor:
    """Allocate SVG2-style area budget using SVG-EAR error efficiency ranking."""
    q_heads, q_clusters, dim = q_centers.shape
    kv_heads = k_centers.shape[0]
    group = q_heads // kv_heads
    kv_for_q = torch.arange(q_heads, device=q_centers.device) // group
    selected_k_centers = k_centers[kv_for_q]
    selected_k_counts = k_counts[kv_for_q]
    nonempty = selected_k_counts[:, None] > 0
    if top_p == 1:
        return nonempty.expand(-1, q_clusters, -1).clone()

    logits = torch.bmm(q_centers.float(), selected_k_centers.float().transpose(1, 2)) / math.sqrt(dim)
    weights = selected_k_counts[:, None].float()
    weighted = weights * torch.exp(logits - logits.amax(-1, keepdim=True))
    probabilities = weighted / weighted.sum(-1, keepdim=True).clamp_min(1.0e-12)
    sorted_prob, mass_order = probabilities.sort(-1, descending=True)
    keep_for_budget = sorted_prob.cumsum(-1) <= top_p
    keep_for_budget[..., 1:] = keep_for_budget[..., :-1].clone()
    keep_for_budget[..., 0] = True

    k_clusters = k_centers.shape[1]
    preserve = int(min_kc_ratio * k_clusters)
    budget_preserve = min(k_clusters, max(1, int(min_kc_ratio * k_clusters * 2)))
    if min_kc_ratio > 0:
        keep_for_budget[..., :budget_preserve] = True

    area = q_counts[..., None].float() * selected_k_counts[:, None].float()
    sorted_area = torch.gather(area, -1, mass_order)
    budget = (keep_for_budget * sorted_area).sum(-1, keepdim=True)

    guaranteed = torch.zeros_like(scores, dtype=torch.bool)
    if preserve:
        guaranteed.scatter_(-1, mass_order[..., :preserve], True)
    guaranteed &= nonempty
    remaining = (budget - (guaranteed * area).sum(-1, keepdim=True)).clamp_min(0)

    candidates = scores.masked_fill(guaranteed | ~nonempty, -float("inf"))
    _, error_order = candidates.sort(-1, descending=True)
    candidate_area = torch.gather(area, -1, error_order)
    cumulative = candidate_area.cumsum(-1)
    # Include the first candidate that crosses the area budget, matching EAR.
    keep_sorted = (cumulative - candidate_area) < remaining
    dynamic = torch.zeros_like(guaranteed).scatter_(-1, error_order, keep_sorted)
    return (guaranteed | dynamic) & nonempty & (q_counts[..., None] > 0)


def _compensated_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    video_idx: torch.Tensor,
    anchor_idx: torch.Tensor,
    q_labels: torch.Tensor,
    k_labels: torch.Tensor,
    k_centers: torch.Tensor,
    v_centers: torch.Tensor,
    k_counts: torch.Tensor,
    route: torch.Tensor,
    dense_attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q_chunk_size: int,
) -> torch.Tensor:
    """Chunked GPU reference: exact tokens plus centroid-compensated blocks."""
    output = torch.empty_like(q)
    if anchor_idx.numel():
        output[:, anchor_idx] = dense_attention(q[:, anchor_idx], k, v)

    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    group = q_heads // kv_heads
    scale = q.shape[-1] ** -0.5
    q_video = q[0, video_idx].transpose(0, 1)
    k_video = k[0, video_idx].transpose(0, 1)
    v_video = v[0, video_idx].transpose(0, 1)

    for q_head in range(q_heads):
        kv_head = q_head // group
        keys = k_video[kv_head].float()
        values = v_video[kv_head].float()
        center_keys = k_centers[kv_head].float()
        center_values = v_centers[kv_head].float()
        center_prior = k_counts[kv_head].float().clamp_min(1).log()
        anchor_keys = k[0, anchor_idx, kv_head].float()
        anchor_values = v[0, anchor_idx, kv_head].float()
        for start in range(0, video_idx.numel(), q_chunk_size):
            end = min(start + q_chunk_size, video_idx.numel())
            query = q_video[q_head, start:end].float()
            routes = route[q_head].index_select(0, q_labels[q_head, start:end])

            token_logits = query @ keys.transpose(0, 1) * scale
            token_keep = routes.index_select(1, k_labels[kv_head])
            token_logits.masked_fill_(~token_keep, -float("inf"))
            center_logits = query @ center_keys.transpose(0, 1) * scale
            center_logits += center_prior[None]
            center_logits.masked_fill_(routes, -float("inf"))

            logits = [token_logits, center_logits]
            all_values = [values, center_values]
            if anchor_idx.numel():
                logits.append(query @ anchor_keys.transpose(0, 1) * scale)
                all_values.append(anchor_values)
            probability = torch.cat(logits, dim=-1).softmax(-1)
            block_output = probability @ torch.cat(all_values)
            output[0, video_idx[start:end], q_head] = block_output.to(output.dtype)
    return output


def svg_ear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    modality: torch.Tensor,
    config: SVGEARConfig,
    *,
    initial_centers: tuple[torch.Tensor, torch.Tensor] | None = None,
    dense_attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    modality_partition: tuple[torch.Tensor, torch.Tensor] | None = None,
    profile=None,
    flashinfer_executor=None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Run standalone error-aware routing with centroid compensation."""
    if q.ndim != 4 or q.shape[0] != 1 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("SVG-EAR expects Q [1,S,Hq,D] and matching K/V [1,S,Hkv,D]")
    _, length, q_heads, dim = q.shape
    kv_heads = k.shape[2]
    if k.shape[1] != length or k.shape[-1] != dim or q_heads % kv_heads:
        raise ValueError("SVG-EAR requires equal Q/K lengths and divisible GQA heads")
    if modality.ndim != 1 or modality.numel() != length:
        raise ValueError("SVG-EAR modality map must cover the full post-CP sequence")
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SVG-EAR dynamic routing cannot run inside CUDA Graph capture")
    dense_attention = dense_attention or _dense_gqa
    if modality_partition is None:
        with _profile_stage(profile, "partition"):
            video_idx = torch.nonzero(modality == int(Modality.VIDEO), as_tuple=True)[0]
            anchor_idx = torch.nonzero(modality != int(Modality.VIDEO), as_tuple=True)[0]
    else:
        video_idx, anchor_idx = modality_partition
    if (
        video_idx.numel() < config.min_video_tokens
        or video_idx.numel() < max(config.q_clusters, config.k_clusters)
    ):
        return dense_attention(q, k, v), None

    input_observer = _INPUT_OBSERVER.get()
    if input_observer is not None:
        input_observer(q, k, v, modality, config, initial_centers, _ACTIVE.get())
    with _profile_stage(profile, "gather_video"):
        q_video = q[0, video_idx].transpose(0, 1).contiguous()
        k_video = k[0, video_idx].transpose(0, 1).contiguous()
        v_video = v[0, video_idx].transpose(0, 1).contiguous()
    q_initial, k_initial = initial_centers or (None, None)
    q_iterations = config.kmeans_iters if q_initial is not None else config.kmeans_init_iters
    k_iterations = config.kmeans_iters if k_initial is not None else config.kmeans_init_iters
    with _profile_stage(profile, "q_kmeans"):
        q_labels, q_centers, q_counts = _kmeans(
            q_video, config.q_clusters, q_iterations, config.kmeans_chunk_size, q_initial
        )
    with _profile_stage(profile, "k_kmeans"):
        k_labels, k_centers, k_counts = _kmeans(
            k_video, config.k_clusters, k_iterations, config.kmeans_chunk_size, k_initial
        )
    with _profile_stage(profile, "v_centroids"):
        v_centers = _cluster_means(v_video, k_labels, k_counts)
    with _profile_stage(profile, "value_error"):
        scores = _error_efficiency_scores(
            q_centers,
            q_counts,
            k_video,
            v_video,
            k_labels,
            k_centers,
            v_centers,
            k_counts,
            config.gamma,
            config.score_q_chunk_size,
        )
    with _profile_stage(profile, "route"):
        route = _error_aware_route(
            scores,
            q_centers,
            k_centers,
            q_counts,
            k_counts,
            config.top_p,
            config.min_kc_ratio,
        )
    route_observer = _ROUTE_OBSERVER.get()
    if route_observer is not None:
        route_observer(
            q_counts,
            k_counts,
            route,
            video_idx.numel(),
            anchor_idx.numel(),
            _ACTIVE.get(),
        )
    with _profile_stage(profile, "compensated_attention"):
        use_flashinfer = config.executor == "flashinfer" or (
            config.executor == "auto"
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        )
        if use_flashinfer:
            if flashinfer_executor is None:
                from inference.model.dit.attention_backends.svg_ear_flashinfer import (
                    SVGEARFlashInferExecutor,
                )

                flashinfer_executor = SVGEARFlashInferExecutor(
                    config.flashinfer_cache_states
                )
            output = flashinfer_executor.run(
                q,
                k,
                v,
                video_idx,
                anchor_idx,
                q_labels,
                k_labels,
                q_counts,
                k_counts,
                k_centers,
                v_centers,
                route,
                profile=profile,
            )
        else:
            output = _compensated_attention(
                q,
                k,
                v,
                video_idx,
                anchor_idx,
                q_labels,
                k_labels,
                k_centers,
                v_centers,
                k_counts,
                route,
                dense_attention,
                config.executor_q_chunk_size,
            )
    return output, (q_centers, k_centers)


@dataclass
class _StepScope:
    session: "SVGEARSession"
    index: int
    branch: str
    call_index: int = 0
    layer_idx: int | None = None
    modality_partitions: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


_ACTIVE: ContextVar[_StepScope | None] = ContextVar("svg_ear_attention_step", default=None)


class SVGEARSession:
    """Generation-local schedule and centroid cache for SVG-EAR."""

    def __init__(self, total_steps: int, config: SVGEARConfig | None = None):
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.total_steps = total_steps
        self.config = config or SVGEARConfig.from_env()
        self.centers: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.flashinfer_executor = None

    @contextmanager
    def step(self, index: int, branch: str):
        if not 0 <= index < self.total_steps or not branch:
            raise ValueError("Expected an in-range denoising step and nonempty branch")
        token = _ACTIVE.set(_StepScope(self, index, branch))
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def run(self, q, k, v, modality, dense_attention, *, layer_idx: int):
        if layer_idx < 0:
            raise ValueError("SVG-EAR requires a nonnegative model layer_idx")
        scope = _ACTIVE.get()
        if scope is None or scope.session is not self:
            raise RuntimeError("SVG-EAR session must run inside its step context")
        scope.call_index += 1
        scope.layer_idx = layer_idx
        protected = (
            scope.index < math.floor(self.total_steps * self.config.dense_step_ratio)
            or layer_idx < self.config.dense_layers
        )
        if protected and not self.config.warmup_kmeans:
            return dense_attention(q, k, v)

        partition_key = (modality.data_ptr(), modality.numel(), modality.device)
        partition = scope.modality_partitions.get(partition_key)
        if partition is None:
            partition = (
                torch.nonzero(modality == int(Modality.VIDEO), as_tuple=True)[0],
                torch.nonzero(modality != int(Modality.VIDEO), as_tuple=True)[0],
            )
            scope.modality_partitions[partition_key] = partition
        cache_key = (
            scope.branch,
            layer_idx,
            tuple(q.shape),
            tuple(k.shape),
            partition[0].numel(),
            q.device,
        )
        output, centers = svg_ear_attention(
            q,
            k,
            v,
            modality,
            self.config,
            initial_centers=self.centers.get(cache_key),
            dense_attention=dense_attention,
            modality_partition=partition,
            flashinfer_executor=self._flashinfer_executor(),
        )
        if centers is not None:
            self.centers[cache_key] = centers
        return dense_attention(q, k, v) if protected else output

    def _flashinfer_executor(self):
        if self.config.executor == "reference":
            return None
        if self.flashinfer_executor is None:
            from inference.model.dit.attention_backends.svg_ear_flashinfer import (
                SVGEARFlashInferExecutor,
            )

            self.flashinfer_executor = SVGEARFlashInferExecutor(
                self.config.flashinfer_cache_states
            )
        return self.flashinfer_executor


def scheduled_svg_ear_attention(q, k, v, modality, dense_attention, *, layer_idx: int):
    """Dispatch through the active generation session or a one-shot config."""
    if layer_idx < 0:
        raise ValueError("SVG-EAR requires a nonnegative model layer_idx")
    scope = _ACTIVE.get()
    if scope is not None:
        return scope.session.run(q, k, v, modality, dense_attention, layer_idx=layer_idx)
    config = SVGEARConfig.from_env()
    if layer_idx < config.dense_layers:
        return dense_attention(q, k, v)
    return svg_ear_attention(q, k, v, modality, config, dense_attention=dense_attention)[0]


__all__ = [
    "SVGEARConfig",
    "SVGEARSession",
    "observe_svg_ear_inputs",
    "observe_svg_ear_routes",
    "scheduled_svg_ear_attention",
    "svg_ear_attention",
]

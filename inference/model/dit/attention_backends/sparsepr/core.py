"""Independent SparsePR attention backend for MagiHuman.

This module implements response-coupled Q/KV partitioning and current-call
probe-fitted residual reconstruction.  It intentionally does not import any
other experimental attention backend.
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


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _env_nonnegative_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


def _auto_token_threshold(
    device: torch.device,
    *,
    override: int = 0,
) -> int:
    """Return a conservative SparsePR/dense crossover for the CUDA device.

    SM120 is calibrated from this repository's 2K--100K benchmark on an RTX
    PRO 5000 Blackwell.  Other architecture bands deliberately err toward
    dense attention until they are calibrated locally.  CPU/reference calls
    keep their explicit ``min_video_tokens`` behavior.
    """
    if override:
        return override
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    major, _minor = torch.cuda.get_device_capability(device)
    if major >= 12:
        return 42_000
    if major >= 9:
        return 45_000
    if major >= 8:
        return 55_000
    return 65_536


def _estimated_sparsepr_working_bytes(video_tokens: int) -> int:
    """Estimate transient bytes beyond resident model/QKV allocations.

    The affine envelope comes from the SM120 FlashInfer benchmark and is
    intentionally rounded upward: roughly 0.5 GiB fixed plus 84 KiB per video
    token.  It is a guardrail rather than an allocator reservation.
    """
    return int(0.5 * 1024**3 + video_tokens * 84 * 1024)


def _auto_fallback_reason(
    device: torch.device,
    video_tokens: int,
    *,
    explicit_min_tokens: int,
    auto_min_tokens: int,
    memory_guard: bool,
) -> tuple[str | None, int]:
    threshold = max(
        explicit_min_tokens,
        _auto_token_threshold(device, override=auto_min_tokens),
    )
    if video_tokens < threshold:
        return f"video_tokens={video_tokens} below crossover={threshold}", threshold
    if memory_guard and device.type == "cuda" and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        reserve = max(2 * 1024**3, int(total_bytes * 0.05))
        required = _estimated_sparsepr_working_bytes(video_tokens) + reserve
        if free_bytes < required:
            return (
                f"free_cuda_bytes={free_bytes} below sparsepr_guard={required}",
                threshold,
            )
    return None, threshold


@dataclass(frozen=True)
class SparsePRConfig:
    q_clusters: int = 300
    k_clusters: int = 1000
    target_density: float = 0.22
    q_rank: int = 64
    k_rank: int = 48
    metric_samples: int = 2048
    factor_power_iters: int = 2
    factor_cholesky_qr_ridge: float = 1.0e-5
    factor_rayleigh_ridge: float = 1.0e-4
    kmeans_init_iters: int = 25
    kmeans_step_iters: int = 2
    kmeans_tol: float = 1.0e-4
    probes: int = 64
    repair_rank: int = 16
    ridge: float = 0.1
    repair_alpha: float = 1.0
    repair_norm_cap: float = 0.0
    repair_basis_source: str = "raw_residual"
    repair: bool = True
    dense_step_ratio: float = 0.20
    dense_layers: int = 1
    min_video_tokens: int = 1024
    auto_fallback: bool = True
    auto_min_video_tokens: int = 0
    auto_memory_guard: bool = True
    allow_zero_video_budget: bool = False
    executor: str = "flashinfer"
    selector: str = "attention_mass"

    def __post_init__(self):
        if not 0 < self.target_density <= 1:
            raise ValueError("SparsePR target_density must be in (0, 1]")
        if not 0 <= self.dense_step_ratio <= 1:
            raise ValueError("SparsePR dense_step_ratio must be in [0, 1]")
        if (
            self.dense_layers < 0
            or self.min_video_tokens < 0
            or self.auto_min_video_tokens < 0
        ):
            raise ValueError("SparsePR dense layer/token guards must be nonnegative")
        if min(
            self.q_clusters,
            self.k_clusters,
            self.q_rank,
            self.k_rank,
            self.metric_samples,
            self.factor_power_iters,
            self.kmeans_init_iters,
            self.kmeans_step_iters,
            self.probes,
            self.repair_rank,
        ) < 1:
            raise ValueError("SparsePR cluster/rank/iteration/probe values must be positive")
        if min(
            self.ridge,
            self.factor_cholesky_qr_ridge,
            self.factor_rayleigh_ridge,
        ) <= 0:
            raise ValueError("SparsePR ridge values must be positive")
        if self.repair_alpha < 0 or self.repair_norm_cap < 0:
            raise ValueError("SparsePR repair alpha/norm cap must be nonnegative")
        if self.repair_basis_source not in {"raw_residual", "fitted_residual"}:
            raise ValueError(
                "SparsePR repair basis source must be raw_residual or fitted_residual"
            )
        if self.kmeans_tol < 0:
            raise ValueError("SparsePR K-means tolerance must be nonnegative")
        if self.executor not in {"flashinfer", "sdpa"}:
            raise ValueError("SparsePR executor must be 'flashinfer' or 'sdpa'")
        if self.selector not in {"value_error", "attention_mass"}:
            raise ValueError("SparsePR selector must be 'value_error' or 'attention_mass'")

    @classmethod
    def from_env(cls) -> "SparsePRConfig":
        return cls(
            q_clusters=_env_int("MAGI_SPARSEPR_Q_CLUSTERS", 300),
            k_clusters=_env_int("MAGI_SPARSEPR_K_CLUSTERS", 1000),
            target_density=float(os.environ.get("MAGI_SPARSEPR_DENSITY", "0.22")),
            q_rank=_env_int("MAGI_SPARSEPR_Q_RANK", 64),
            k_rank=_env_int("MAGI_SPARSEPR_K_RANK", 48),
            metric_samples=_env_int("MAGI_SPARSEPR_METRIC_SAMPLES", 2048),
            factor_power_iters=_env_int("MAGI_SPARSEPR_FACTOR_POWER_ITERS", 2),
            factor_cholesky_qr_ridge=float(
                os.environ.get("MAGI_SPARSEPR_FACTOR_CHOLESKY_QR_RIDGE", "1e-5")
            ),
            factor_rayleigh_ridge=float(
                os.environ.get("MAGI_SPARSEPR_FACTOR_RAYLEIGH_RIDGE", "1e-4")
            ),
            kmeans_init_iters=_env_int("MAGI_SPARSEPR_KMEANS_INIT_ITERS", 25),
            kmeans_step_iters=_env_int("MAGI_SPARSEPR_KMEANS_STEP_ITERS", 2),
            kmeans_tol=float(os.environ.get("MAGI_SPARSEPR_KMEANS_TOL", "1e-4")),
            probes=_env_int("MAGI_SPARSEPR_PROBES", 64),
            repair_rank=_env_int("MAGI_SPARSEPR_REPAIR_RANK", 16),
            ridge=float(os.environ.get("MAGI_SPARSEPR_RIDGE", "0.1")),
            repair_alpha=float(os.environ.get("MAGI_SPARSEPR_REPAIR_ALPHA", "1.0")),
            repair_norm_cap=float(
                os.environ.get("MAGI_SPARSEPR_REPAIR_NORM_CAP", "0.0")
            ),
            repair_basis_source=os.environ.get(
                "MAGI_SPARSEPR_REPAIR_BASIS_SOURCE", "raw_residual"
            ).lower(),
            repair=os.environ.get("MAGI_SPARSEPR_REPAIR", "1").lower()
            not in {"0", "false", "no", "off"},
            dense_step_ratio=float(os.environ.get("MAGI_SPARSEPR_DENSE_STEP_RATIO", "0.20")),
            dense_layers=int(os.environ.get("MAGI_SPARSEPR_DENSE_LAYERS", "1")),
            min_video_tokens=int(os.environ.get("MAGI_SPARSEPR_MIN_VIDEO_TOKENS", "1024")),
            auto_fallback=_env_bool("MAGI_SPARSEPR_AUTO_FALLBACK", True),
            auto_min_video_tokens=_env_nonnegative_int(
                "MAGI_SPARSEPR_AUTO_MIN_VIDEO_TOKENS", 0
            ),
            auto_memory_guard=_env_bool("MAGI_SPARSEPR_AUTO_MEMORY_GUARD", True),
            allow_zero_video_budget=_env_bool(
                "MAGI_SPARSEPR_ALLOW_ZERO_VIDEO_BUDGET", False
            ),
            executor=os.environ.get("MAGI_SPARSEPR_EXECUTOR", "flashinfer").lower(),
            selector=os.environ.get("MAGI_SPARSEPR_SELECTOR", "attention_mass").lower(),
        )


def _dense_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """CPU/reference dense GQA for tensors in [B,N,H,D] layout."""
    qh, kh = q.shape[2], k.shape[2]
    if qh % kh:
        raise ValueError("SparsePR requires divisible GQA heads")
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        enable_gqa=qh != kh,
        dropout_p=0.0,
    ).transpose(1, 2)


def _metric_factor(metric: torch.Tensor, rank: int) -> torch.Tensor:
    """Leading response-metric factor with eigenvalue weighting."""
    metric = metric.float()
    values, vectors = torch.linalg.eigh(metric)
    take = min(rank, metric.shape[-1])
    values = values[..., -take:].clamp_min(0)
    vectors = vectors[..., -take:]
    scale = values.mean(dim=-1, keepdim=True).clamp_min(1e-12)
    return vectors * (values / scale).sqrt().unsqueeze(-2)


def _warm_metric_factor(
    metric: torch.Tensor,
    previous: torch.Tensor | None,
    rank: int,
    power_iters: int,
) -> torch.Tensor:
    """Compatibility wrapper for the Cholesky-QR/device-recovery update."""
    if previous is None or previous.shape != (
        metric.shape[0],
        metric.shape[1],
        min(rank, metric.shape[1]),
    ):
        return _metric_factor(metric, rank)
    from .factor import warm_cholesky_metric_factor

    factor, _, _ = warm_cholesky_metric_factor(
        metric,
        previous,
        previous,
        rank,
        power_iters=power_iters,
    )
    return factor


def _project_query_features(
    q: torch.Tensor,
    factor: torch.Tensor,
) -> torch.Tensor:
    if q.is_cuda and q.shape[-1] <= 256 and factor.shape[-1] >= 16:
        from .kernels.role_projection import role_projection_rms_triton

        return role_projection_rms_triton(q, factor)
    projected = torch.bmm(q.float(), factor.float())
    return projected / projected.square().mean(
        -1, keepdim=True
    ).sqrt().clamp_min(1e-6)


def _assign(samples: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    x2 = samples.float().square().sum(-1, keepdim=True)
    c2 = centers.float().square().sum(-1).unsqueeze(1)
    distance = (
        x2
        + c2
        - 2 * torch.bmm(samples.float(), centers.float().transpose(1, 2))
    ).clamp_min_(0)
    return distance.argmin(-1)


def _centroids(
    samples: torch.Tensor,
    labels: torch.Tensor,
    clusters: int,
    fallback: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads, _, dim = samples.shape
    counts = torch.zeros((heads, clusters), dtype=torch.int32, device=samples.device)
    counts.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.int32))
    sums = torch.zeros((heads, clusters, dim), dtype=torch.float32, device=samples.device)
    sums.scatter_add_(1, labels[..., None].expand(-1, -1, dim), samples.float())
    updated = sums / counts.clamp_min(1)[..., None]
    if fallback is None:
        fallback = samples[:, :1].float().expand(-1, clusters, -1)
    updated = torch.where(counts[..., None] > 0, updated, fallback.float())
    return updated, counts


def _kmeans(
    samples: torch.Tensor,
    clusters: int,
    iterations: int,
    initial: torch.Tensor | None,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flash-KMeans-compatible Lloyd iteration and convergence semantics."""
    clusters = min(clusters, samples.shape[1])
    original_dim = samples.shape[2]
    work_samples = samples.contiguous()
    use_triton = samples.is_cuda and original_dim >= 16
    if initial is None or initial.shape != (samples.shape[0], clusters, samples.shape[2]):
        indices = torch.randint(
            0,
            samples.shape[1],
            (samples.shape[0], clusters),
            device=samples.device,
        )
        centers = torch.gather(
            work_samples,
            1,
            indices[..., None].expand(-1, -1, work_samples.shape[2]),
        ).to(work_samples.dtype)
    else:
        centers = initial.to(work_samples.dtype)
    if use_triton:
        from .kernels.kmeans import (
            euclid_assign_triton,
            triton_centroid_update_sorted_euclid,
        )

        x_sq = work_samples.float().square().sum(-1)
        for _ in range(iterations):
            labels = euclid_assign_triton(work_samples, centers, x_sq)
            updated, counts = triton_centroid_update_sorted_euclid(
                work_samples, labels, centers
            )
            center_shift = torch.linalg.vector_norm(
                updated.float() - centers.float(), dim=-1
            ).amax()
            if center_shift < tolerance:
                break
            centers = updated
        return labels.long(), centers[..., :original_dim].float(), counts
    for _ in range(iterations):
        labels = _assign(work_samples, centers)
        updated, counts = _centroids(work_samples, labels, clusters, centers)
        center_shift = torch.linalg.vector_norm(
            updated.float() - centers.float(), dim=-1
        ).amax()
        if center_shift < tolerance:
            break
        centers = updated
    return labels, centers[..., :original_dim], counts


def _original_centroids(
    samples: torch.Tensor,
    labels: torch.Tensor,
    clusters: int,
    fallback: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _centroids(samples, labels, clusters, fallback)


def _sample_gqa_queries(q: torch.Tensor, kv_heads: int, limit: int) -> torch.Tensor:
    q_heads, tokens, dim = q.shape
    group = q_heads // kv_heads
    grouped = q.view(kv_heads, group * tokens, dim)
    count = min(group * limit, grouped.shape[1])
    indices = torch.linspace(0, grouped.shape[1] - 1, count, device=q.device).long()
    return grouped.index_select(1, indices)


def _response_partitions(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: SparsePRConfig,
    state: "SparsePRCore",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return response-coupled Q/K labels, sizes, centroids and Q features."""
    q_heads, tokens, dim = q.shape
    kv_heads = k.shape[0]
    group = q_heads // kv_heads

    from .factor import warm_cholesky_metric_factor

    def update_factor(
        metric: torch.Tensor,
        basis: torch.Tensor | None,
        current: torch.Tensor | None,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if basis is None or current is None:
            factor = _metric_factor(metric, rank)
            return factor, factor
        factor, updated_basis, _ = warm_cholesky_metric_factor(
            metric,
            basis,
            current,
            rank,
            power_iters=config.factor_power_iters,
            cholesky_qr_ridge=config.factor_cholesky_qr_ridge,
            rayleigh_ridge=config.factor_rayleigh_ridge,
        )
        return factor, updated_basis

    sampled = _sample_gqa_queries(q, kv_heads, config.metric_samples)
    k_metric = torch.bmm(sampled.float().transpose(1, 2), sampled.float()) / sampled.shape[1]
    k_factor, k_factor_basis = update_factor(
        k_metric,
        state.k_factor_basis,
        state.k_factor,
        config.k_rank,
    )
    k_features = torch.bmm(k.float(), k_factor)
    k_initial = None
    if state.k_centroids is not None:
        k_initial = torch.bmm(state.k_centroids.float(), k_factor)
    iterations = config.kmeans_step_iters if state.initialized else config.kmeans_init_iters
    k_labels, _, k_sizes = _kmeans(
        k_features,
        config.k_clusters,
        iterations,
        k_initial,
        config.kmeans_tol,
    )
    k_clusters = min(config.k_clusters, tokens)
    k_centroids, k_sizes = _original_centroids(k, k_labels, k_clusters, state.k_centroids)
    v_centroids, _ = _original_centroids(v, k_labels, k_clusters, state.v_centroids)

    nonempty = (k_sizes > 0).float()
    mean = (k_centroids * nonempty[..., None]).sum(
        1, keepdim=True
    ) / nonempty.sum(1, keepdim=True).clamp_min(1)[..., None]
    centered = (k_centroids - mean) * nonempty[..., None]
    q_metric = torch.bmm(centered.transpose(1, 2), centered) / (
        nonempty.sum(1).clamp_min(1)[:, None, None] * dim
    )
    q_factor_kv, q_factor_basis = update_factor(
        q_metric,
        state.q_factor_basis,
        state.q_factor,
        config.q_rank,
    )
    q_factor = q_factor_kv.repeat_interleave(group, dim=0)
    q_features = _project_query_features(q, q_factor)
    q_initial = None
    if state.q_centroids is not None:
        q_initial = torch.bmm(state.q_centroids.float(), q_factor)
        q_initial = q_initial / q_initial.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    q_labels, _, q_sizes = _kmeans(
        q_features,
        config.q_clusters,
        iterations,
        q_initial,
        config.kmeans_tol,
    )
    q_clusters = min(config.q_clusters, tokens)
    q_centroids, q_sizes = _original_centroids(q, q_labels, q_clusters, state.q_centroids)

    state.q_centroids = q_centroids.detach()
    state.k_centroids = k_centroids.detach()
    state.v_centroids = v_centroids.detach()
    state.q_factor = q_factor_kv.detach()
    state.k_factor = k_factor.detach()
    state.q_factor_basis = q_factor_basis.detach()
    state.k_factor_basis = k_factor_basis.detach()
    return (
        q_labels,
        k_labels,
        q_sizes,
        k_sizes,
        q_centroids,
        k_centroids,
        v_centroids,
        q_features,
    )


def _attention_mass_scores(
    q_centroids: torch.Tensor,
    k_tokens: torch.Tensor,
    k_labels: torch.Tensor,
    k_sizes: torch.Tensor,
) -> torch.Tensor:
    """Exact log attention mass of every key cluster at each Q centroid.

    GQA query heads sharing a KV head are evaluated together. The chunk bound
    keeps temporary Q-by-token score matrices bounded for long sequences.
    """
    q_heads, q_clusters, dim = q_centroids.shape
    kv_heads, tokens, _ = k_tokens.shape
    group = q_heads // kv_heads
    clusters = k_sizes.shape[1]
    if q_centroids.is_cuda and dim == 128:
        from .kernels.selector import attention_mass_scores_triton

        order = torch.argsort(k_labels, dim=1, stable=True)
        k_sorted = torch.gather(
            k_tokens,
            1,
            order[..., None].expand(-1, -1, dim),
        )
        return attention_mass_scores_triton(
            q_centroids, k_sorted, k_sizes, group
        )
    result = torch.full(
        (q_heads, q_clusters, clusters),
        -float("inf"),
        device=q_centroids.device,
        dtype=torch.float32,
    )
    # Keep the largest temporary logits tensor near 16 MiB in FP32.
    max_elements = 4 * 1024 * 1024
    rows_per_chunk = max(
        1, min(q_clusters, max_elements // max(group * tokens, 1))
    )
    for kv_head in range(kv_heads):
        keys = k_tokens[kv_head].float()
        labels = k_labels[kv_head]
        label_index = labels.view(1, 1, tokens)
        head_start = kv_head * group
        queries = q_centroids[head_start : head_start + group].float()
        for start in range(0, q_clusters, rows_per_chunk):
            rows = queries[:, start : start + rows_per_chunk]
            logits = torch.matmul(rows, keys.transpose(0, 1)) / math.sqrt(dim)
            expanded_labels = label_index.expand(group, rows.shape[1], -1)
            maxima = torch.full(
                (group, rows.shape[1], clusters),
                -float("inf"),
                device=logits.device,
                dtype=torch.float32,
            )
            maxima.scatter_reduce_(
                2, expanded_labels, logits, reduce="amax", include_self=True
            )
            centered = logits - torch.gather(maxima, 2, expanded_labels)
            sums = torch.zeros_like(maxima)
            sums.scatter_add_(2, expanded_labels, centered.exp())
            scores = maxima + sums.clamp_min(1e-30).log()
            scores.masked_fill_(
                k_sizes[kv_head].view(1, 1, -1) == 0, -float("inf")
            )
            result[
                head_start : head_start + group,
                start : start + rows.shape[1],
            ] = scores
    return result


def _value_error_scores(
    q_centroids: torch.Tensor,
    k_tokens: torch.Tensor,
    v_tokens: torch.Tensor,
    k_labels: torch.Tensor,
    k_sizes: torch.Tensor,
    k_centroids: torch.Tensor,
    v_centroids: torch.Tensor,
) -> torch.Tensor:
    """SVG-EAR value-error score from the SparsePR paper, in bounded chunks."""
    q_heads, q_clusters, dim = q_centroids.shape
    kv_heads = k_tokens.shape[0]
    group = q_heads // kv_heads
    if q_centroids.is_cuda and dim == 128:
        from .kernels.selector import grouped_value_error_scores_triton

        order = torch.argsort(k_labels, dim=1, stable=True)
        gather_index = order[..., None].expand(-1, -1, dim)
        k_sorted = torch.gather(k_tokens, 1, gather_index)
        v_sorted = torch.gather(v_tokens, 1, gather_index)
        return grouped_value_error_scores_triton(
            q_centroids,
            k_sorted,
            v_sorted,
            k_centroids,
            v_centroids,
            k_sizes,
            group,
        )
    result = torch.empty(
        (q_heads, q_clusters, k_centroids.shape[1]),
        device=q_centroids.device,
        dtype=torch.float32,
    )
    chunk = 16
    for head in range(q_heads):
        kv_head = head // group
        keys = k_tokens[kv_head].float()
        values = v_tokens[kv_head].float()
        labels = k_labels[kv_head]
        counts = k_sizes[kv_head].float().clamp_min(1)
        kc = k_centroids[kv_head].float()
        vc = v_centroids[kv_head].float()
        value_norm = values.square().sum(-1)
        for start in range(0, q_clusters, chunk):
            query = q_centroids[head, start : start + chunk].float()
            token_logits = query @ keys.transpose(0, 1) / math.sqrt(dim)
            center_logits = query @ kc.transpose(0, 1) / math.sqrt(dim)
            shift = torch.maximum(token_logits.amax(1), center_logits.amax(1)).unsqueeze(1)
            token_weight = (token_logits - shift).exp()
            center_weight = (center_logits - shift).exp()
            rows = query.shape[0]
            mean_b = torch.zeros((rows, kc.shape[0], dim), device=query.device)
            mean_b.scatter_add_(
                1,
                labels.view(1, -1, 1).expand(rows, -1, dim),
                token_weight[..., None] * values[None],
            )
            mean_b /= counts[None, :, None]
            mean_b2 = torch.zeros((rows, kc.shape[0]), device=query.device)
            mean_b2.scatter_add_(
                1,
                labels.view(1, -1).expand(rows, -1),
                token_weight.square() * value_norm[None],
            )
            mean_b2 /= counts[None]
            a = center_weight[..., None] * vc[None]
            score = a.square().sum(-1) - 2 * (a * mean_b).sum(-1) + mean_b2
            log_score = score.clamp_min_(1e-30).log() + 2 * shift
            log_score.masked_fill_(k_sizes[kv_head][None] == 0, -float("inf"))
            result[head, start : start + rows] = log_score
    return result


def _video_key_budget(
    *,
    video_tokens: int,
    anchor_tokens: int,
    probes: int,
    total_tokens: int,
    target_density: float,
) -> int:
    """Official N8 fixed-density budget for video keys per video query."""
    probe_fraction = min(probes, video_tokens) / max(video_tokens, 1)
    base_total_density = max(0.0, target_density - probe_fraction)
    target_keys = int(math.ceil(base_total_density * total_tokens))
    return min(video_tokens, max(0, target_keys - anchor_tokens))


def _fixed_density_map(
    scores: torch.Tensor,
    k_sizes: torch.Tensor,
    q_sizes: torch.Tensor,
    *,
    video_tokens: int,
    anchor_tokens: int,
    probes: int,
    total_tokens: int,
    target_density: float,
) -> torch.Tensor:
    q_heads = scores.shape[0]
    kv_heads = k_sizes.shape[0]
    group = q_heads // kv_heads
    selected_sizes = k_sizes.repeat_interleave(group, dim=0).long()
    # Match the official N8 fixed-density semantics: density is the average
    # key fraction for each *video query*. Dense repair probes are removed
    # additively first, then always-attended anchor keys are removed from the
    # resulting per-row key budget. Non-video query rows are handled densely
    # by the model adapter and are not charged against video routing here.
    key_budget = _video_key_budget(
        video_tokens=video_tokens,
        anchor_tokens=anchor_tokens,
        probes=probes,
        total_tokens=total_tokens,
        target_density=target_density,
    )
    # Without always-attended anchors, every Q row still needs at least one key.
    if anchor_tokens == 0:
        key_budget = max(1, key_budget)
    if scores.is_cuda:
        from .kernels.budget import exact_budget_mask_triton

        budgets = torch.full(
            scores.shape[:2],
            key_budget,
            dtype=torch.int32,
            device=scores.device,
        )
        result = exact_budget_mask_triton(scores, selected_sizes, budgets)
    else:
        order = scores.argsort(dim=-1, descending=True)
        sorted_sizes = torch.gather(
            selected_sizes[:, None, :].expand(-1, scores.shape[1], -1),
            2,
            order,
        )
        cumulative = sorted_sizes.cumsum(-1)
        keep_sorted = (cumulative <= key_budget) & (sorted_sizes > 0)
        selected_cost = (sorted_sizes * keep_sorted.long()).sum(-1)
        selected_count = keep_sorted.long().sum(-1)
        valid_count = (sorted_sizes > 0).long().sum(-1)
        include_crossing = (
            (key_budget > 0)
            & (selected_cost < key_budget)
            & (selected_count < valid_count)
        )
        crossing = selected_count.clamp_max(scores.shape[-1] - 1)
        crossing_mask = torch.zeros_like(keep_sorted)
        crossing_mask.scatter_(
            2, crossing[..., None], include_crossing[..., None]
        )
        keep_sorted |= crossing_mask
        result = torch.zeros_like(keep_sorted)
        result.scatter_(2, order, keep_sorted)
    # Empty Q clusters are harmless, but keeping their maps empty saves planning work.
    result &= q_sizes[..., None] > 0
    return result


def _sdpa_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    video_idx: torch.Tensor,
    anchor_idx: torch.Tensor,
    q_labels: torch.Tensor,
    k_labels: torch.Tensor,
    cluster_map: torch.Tensor,
) -> torch.Tensor:
    """Dense-mask numerical reference for the standalone backend."""
    qh, kh = q.shape[2], k.shape[2]
    group = qh // kh
    length = q.shape[1]
    allowed = torch.zeros((qh, length, length), dtype=torch.bool, device=q.device)
    if anchor_idx.numel():
        allowed[:, anchor_idx, :] = True
        allowed[:, :, anchor_idx] = True
    for head in range(qh):
        kv_head = head // group
        video_map = cluster_map[head][q_labels[head][:, None], k_labels[kv_head][None, :]]
        allowed[head][video_idx[:, None], video_idx[None, :]] = video_map
    kh_expanded = k[0].transpose(0, 1).repeat_interleave(group, dim=0)
    vh_expanded = v[0].transpose(0, 1).repeat_interleave(group, dim=0)
    output = F.scaled_dot_product_attention(
        q[0].transpose(0, 1).unsqueeze(0),
        kh_expanded.unsqueeze(0),
        vh_expanded.unsqueeze(0),
        attn_mask=allowed.unsqueeze(0),
        dropout_p=0.0,
    )
    return output.transpose(1, 2).contiguous()


def _select_probes(
    q_features: torch.Tensor,
    q_labels: torch.Tensor,
    q_sizes: torch.Tensor,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One centroid-nearest representative from each largest nonempty Q group."""
    heads, tokens, dim = q_features.shape
    count = min(rows, tokens)
    selected_all, weights_all = [], []
    token_ids = torch.arange(tokens, device=q_features.device)
    for head in range(heads):
        labels = q_labels[head]
        clusters = q_sizes.shape[1]
        centers, _ = _centroids(
            q_features[head : head + 1], labels[None], clusters, None
        )
        distance = (q_features[head] - centers[0, labels]).square().sum(-1)
        best_distance = torch.full((clusters,), float("inf"), device=q_features.device)
        best_distance.scatter_reduce_(0, labels, distance, reduce="amin", include_self=True)
        candidates = torch.where(
            distance == best_distance[labels], token_ids, torch.full_like(token_ids, tokens)
        )
        first = torch.full((clusters,), tokens, device=q_features.device, dtype=torch.long)
        first.scatter_reduce_(0, labels, candidates, reduce="amin", include_self=True)
        order = torch.argsort(q_sizes[head], descending=True, stable=True)
        chosen = first[order]
        chosen = chosen[chosen < tokens][:count]
        if chosen.numel() < count:
            used = torch.zeros(tokens, dtype=torch.bool, device=q_features.device)
            used[chosen] = True
            chosen = torch.cat((chosen, token_ids[~used][: count - chosen.numel()]))
        chosen_labels = labels[chosen]
        sampled = torch.zeros(clusters, device=q_features.device)
        sampled.scatter_add_(0, chosen_labels, torch.ones_like(chosen_labels, dtype=torch.float32))
        group_weights = q_sizes[head].float() / sampled.clamp_min(1)
        weights = group_weights[chosen_labels]
        weights /= weights.mean().clamp_min(1e-12)
        selected_all.append(chosen)
        weights_all.append(weights)
    return torch.stack(selected_all), torch.stack(weights_all)


def _repair(
    base: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    video_idx: torch.Tensor,
    q_features: torch.Tensor,
    q_labels: torch.Tensor,
    q_sizes: torch.Tensor,
    probes: int,
    repair_rank: int,
    ridge: float,
    repair_alpha: float,
    repair_norm_cap: float,
    repair_basis_source: str,
    previous_state,
    dense_attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, tuple, torch.Tensor]:
    from .residual import (
        fit_reduced_rank_residual,
        predict_reduced_rank_residual,
        recover_repair_state,
    )

    qh, dim = q.shape[2], q.shape[3]
    rows, weights = _select_probes(q_features, q_labels, q_sizes, probes)
    absolute = video_idx[rows]
    q_by_head = q[0].transpose(0, 1)
    q_probe_h = torch.gather(q_by_head, 1, absolute[..., None].expand(-1, -1, dim))
    q_probe = q_probe_h.transpose(0, 1).unsqueeze(0).contiguous()
    dense_probe = dense_attention(q_probe, k, v)[0].transpose(0, 1).float()
    base_by_head = base[0].transpose(0, 1)
    sparse_probe = torch.gather(base_by_head, 1, absolute[..., None].expand(-1, -1, dim)).float()
    residual = dense_probe - sparse_probe

    state = fit_reduced_rank_residual(
        sparse_probe,
        residual,
        rank=min(repair_rank, sparse_probe.shape[1], dim),
        ridge=ridge,
        weights=weights,
        basis_source=repair_basis_source,
    )
    state, recovered = recover_repair_state(state, previous_state)
    x_mean, x_scale, r_mean, left, basis = state

    corrected = base_by_head.contiguous()
    rank = left.shape[-1]
    if corrected.is_cuda and dim <= 256 and rank >= 16 and not (rank & (rank - 1)):
        from .kernels.repair import repair_apply_triton

        repair_apply_triton(
            corrected,
            video_idx,
            x_mean,
            x_scale,
            r_mean,
            left,
            basis,
            repair_alpha,
            repair_norm_cap,
        )
    else:
        video_base = corrected[:, video_idx].float()
        predicted = predict_reduced_rank_residual(video_base, state)
        if repair_norm_cap > 0:
            predicted_norm = torch.linalg.vector_norm(predicted, dim=2, keepdim=True)
            base_norm = torch.linalg.vector_norm(video_base, dim=2, keepdim=True)
            predicted *= (
                repair_norm_cap * base_norm / predicted_norm.clamp_min(1e-20)
            ).clamp_max(1.0)
        corrected[:, video_idx] = (
            video_base + repair_alpha * predicted
        ).to(corrected.dtype)
    corrected.scatter_(
        1,
        absolute[..., None].expand(-1, -1, dim),
        dense_probe.to(corrected.dtype),
    )
    output = corrected.transpose(0, 1).unsqueeze(0).to(base.dtype).contiguous()
    return output, state, recovered


class SparsePRCore:
    """Per-layer, per-CFG-branch state for the standalone SparsePR method."""

    def __init__(self, config: SparsePRConfig):
        self.config = config
        self.q_centroids: torch.Tensor | None = None
        self.k_centroids: torch.Tensor | None = None
        self.v_centroids: torch.Tensor | None = None
        self.q_factor: torch.Tensor | None = None
        self.k_factor: torch.Tensor | None = None
        self.q_factor_basis: torch.Tensor | None = None
        self.k_factor_basis: torch.Tensor | None = None
        self.repair_state = None
        self.repair_recovery_mask: torch.Tensor | None = None
        self.calls = 0
        self.last_density = 1.0
        self.last_mode = "uninitialized"
        self.last_fallback_reason: str | None = None
        self.last_auto_threshold = self.config.min_video_tokens

    @property
    def initialized(self) -> bool:
        return self.q_centroids is not None and self.k_centroids is not None

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        modality: torch.Tensor,
        dense_attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        flashinfer_executor=None,
    ) -> torch.Tensor:
        if (
            q.ndim != 4
            or q.shape[0] != 1
            or k.ndim != 4
            or k.shape[0] != 1
            or v.shape != k.shape
        ):
            raise ValueError("SparsePR expects Q/K/V in [1,N,H,D] layout")
        if (
            q.shape[1] != k.shape[1]
            or q.shape[-1] != k.shape[-1]
            or q.shape[2] % k.shape[2]
        ):
            raise ValueError("SparsePR requires equal sequence/head dimensions and divisible GQA")
        if modality.ndim != 1 or modality.numel() != q.shape[1]:
            raise ValueError("SparsePR modality map must describe the global sequence")
        video_idx = torch.nonzero(modality == int(Modality.VIDEO), as_tuple=True)[0]
        anchor_idx = torch.nonzero(modality != int(Modality.VIDEO), as_tuple=True)[0]
        video_tokens = video_idx.numel()
        fallback_reason = None
        if video_tokens == 0:
            fallback_reason = "sequence has no video tokens"
        elif self.config.auto_fallback:
            fallback_reason, self.last_auto_threshold = _auto_fallback_reason(
                q.device,
                video_tokens,
                explicit_min_tokens=self.config.min_video_tokens,
                auto_min_tokens=self.config.auto_min_video_tokens,
                memory_guard=self.config.auto_memory_guard,
            )
        else:
            self.last_auto_threshold = self.config.min_video_tokens
            if video_tokens < self.config.min_video_tokens:
                fallback_reason = (
                    f"video_tokens={video_tokens} below explicit_min="
                    f"{self.config.min_video_tokens}"
                )
        if fallback_reason is None and not self.config.allow_zero_video_budget:
            key_budget = _video_key_budget(
                video_tokens=video_tokens,
                anchor_tokens=anchor_idx.numel(),
                probes=self.config.probes if self.config.repair else 0,
                total_tokens=q.shape[1],
                target_density=self.config.target_density,
            )
            if key_budget == 0:
                fallback_reason = (
                    "configured density leaves zero video-video key budget "
                    f"(video={video_tokens}, anchors={anchor_idx.numel()})"
                )
        if fallback_reason is not None:
            self.last_mode = "dense_fallback"
            self.last_fallback_reason = fallback_reason
            self.last_density = 1.0
            return dense_attention(q, k, v)
        self.last_mode = "sparsepr"
        self.last_fallback_reason = None

        q_video = q[0, video_idx].transpose(0, 1).contiguous()
        k_video = k[0, video_idx].transpose(0, 1).contiguous()
        v_video = v[0, video_idx].transpose(0, 1).contiguous()
        (
            q_labels,
            k_labels,
            q_sizes,
            k_sizes,
            q_centroids,
            k_centroids,
            v_centroids,
            q_features,
        ) = _response_partitions(q_video, k_video, v_video, self.config, self)
        group = q.shape[2] // k.shape[2]
        if self.config.selector == "value_error":
            scores = _value_error_scores(
                q_centroids, k_video, v_video, k_labels, k_sizes, k_centroids, v_centroids
            )
        else:
            scores = _attention_mass_scores(q_centroids, k_video, k_labels, k_sizes)
        cluster_map = _fixed_density_map(
            scores,
            k_sizes,
            q_sizes,
            video_tokens=video_idx.numel(),
            anchor_tokens=anchor_idx.numel(),
            probes=self.config.probes if self.config.repair else 0,
            total_tokens=q.shape[1],
            target_density=self.config.target_density,
        )
        selected = (
            q_sizes.long()[..., None]
            * k_sizes.repeat_interleave(group, 0).long()[:, None]
        )
        routed_pairs = (selected * cluster_map.long()).sum(dim=(1, 2)).float()
        always_pairs = (
            anchor_idx.numel() * q.shape[1]
            + video_idx.numel() * anchor_idx.numel()
        )
        probe_pairs = (
            min(self.config.probes, video_idx.numel()) * q.shape[1]
            if self.config.repair
            else 0
        )
        self.last_density = float(
            ((routed_pairs + always_pairs + probe_pairs) / (q.shape[1] ** 2)).mean().item()
        )

        if self.config.executor == "flashinfer" and q.is_cuda:
            if flashinfer_executor is None:
                from .flashinfer import SparsePRFlashInferExecutor

                flashinfer_executor = SparsePRFlashInferExecutor()
            base = flashinfer_executor.run(
                q, k, v, video_idx, anchor_idx,
                q_labels, k_labels, q_sizes, k_sizes, cluster_map,
            )
        else:
            base = _sdpa_sparse(
                q, k, v, video_idx, anchor_idx, q_labels, k_labels, cluster_map
            )
        if self.config.repair:
            output, self.repair_state, self.repair_recovery_mask = _repair(
                base, q, k, v, video_idx, q_features, q_labels, q_sizes,
                self.config.probes,
                self.config.repair_rank,
                self.config.ridge,
                self.config.repair_alpha,
                self.config.repair_norm_cap,
                self.config.repair_basis_source,
                self.repair_state,
                dense_attention,
            )
        else:
            output = base
        self.calls += 1
        return output


@dataclass
class _StepScope:
    session: "SparsePRSession"
    index: int
    branch: str


_ACTIVE: ContextVar[_StepScope | None] = ContextVar("sparsepr_attention_step", default=None)


class SparsePRSession:
    """Own SparsePR state without sharing configuration or caches with other backends."""

    def __init__(self, total_steps: int, config: SparsePRConfig | None = None):
        if total_steps < 1:
            raise ValueError("SparsePR total_steps must be positive")
        self.total_steps = total_steps
        self.config = config or SparsePRConfig.from_env()
        self.cores: dict[tuple, SparsePRCore] = {}
        self._executor = None

    def _flashinfer_executor(self):
        if self._executor is None:
            from .flashinfer import SparsePRFlashInferExecutor

            self._executor = SparsePRFlashInferExecutor()
        return self._executor

    @contextmanager
    def step(self, index: int, branch: str):
        if not 0 <= index < self.total_steps or not branch:
            raise ValueError("Expected an in-range SparsePR step and nonempty branch")
        token = _ACTIVE.set(_StepScope(self, index, branch))
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def run(self, q, k, v, modality, dense_attention, *, layer_idx: int):
        if layer_idx < 0:
            raise ValueError("SparsePR layer_idx must be nonnegative")
        scope = _ACTIVE.get()
        assert scope is not None and scope.session is self
        if (
            scope.index < math.floor(self.total_steps * self.config.dense_step_ratio)
            or layer_idx < self.config.dense_layers
        ):
            return dense_attention(q, k, v)
        if q.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SparsePR dynamic routing cannot run inside CUDA Graph capture")
        video_count = int((modality == int(Modality.VIDEO)).sum().item())
        key = (scope.branch, layer_idx, tuple(q.shape), tuple(k.shape), video_count, q.device)
        core = self.cores.get(key)
        if core is None:
            core = SparsePRCore(self.config)
            self.cores[key] = core
        executor = (
            self._flashinfer_executor()
            if self.config.executor == "flashinfer" and q.is_cuda
            else None
        )
        return core.run(q, k, v, modality, dense_attention, flashinfer_executor=executor)


def scheduled_sparsepr_attention(q, k, v, modality, dense_attention, *, layer_idx: int):
    if layer_idx < 0:
        raise ValueError("SparsePR layer_idx must be nonnegative")
    scope = _ACTIVE.get()
    if scope is not None:
        return scope.session.run(q, k, v, modality, dense_attention, layer_idx=layer_idx)
    config = SparsePRConfig.from_env()
    if layer_idx < config.dense_layers:
        return dense_attention(q, k, v)
    return SparsePRCore(config).run(q, k, v, modality, dense_attention)


__all__ = [
    "SparsePRConfig",
    "SparsePRCore",
    "SparsePRSession",
    "scheduled_sparsepr_attention",
    "_dense_gqa",
]

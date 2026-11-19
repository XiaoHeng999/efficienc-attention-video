# Derived from SparsePR revision e46646f.
# SPDX-License-Identifier: Apache-2.0
"""Device-only warm response-metric factor updates."""

from __future__ import annotations

import torch


def _cholesky_qr(
    values: torch.Tensor,
    relative_ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cholesky-QR with a finite placeholder for failed device batches."""
    normalized = values / torch.linalg.vector_norm(
        values, dim=1, keepdim=True
    ).clamp_min(1.0e-12)
    gram = torch.bmm(normalized.transpose(1, 2), normalized)
    gram = 0.5 * (gram + gram.transpose(1, 2))
    scale = (
        gram.diagonal(dim1=1, dim2=2)
        .mean(dim=1, keepdim=True)
        .clamp_min(1.0e-12)
    )
    gram.diagonal(dim1=1, dim2=2).add_(scale * float(relative_ridge))
    factor, info = torch.linalg.cholesky_ex(gram, check_errors=False)
    failed = info != 0
    identity = torch.eye(
        factor.shape[-1], device=factor.device, dtype=factor.dtype
    ).expand_as(factor)
    factor = torch.where(failed[:, None, None], identity, factor)
    basis = torch.linalg.solve_triangular(
        factor, normalized.transpose(1, 2), upper=False
    ).transpose(1, 2)
    return basis, failed


def _cholesky_with_device_fallback(
    matrix: torch.Tensor,
    scale: torch.Tensor,
    relative_ridge: float,
    fallback_relative_ridge: float = 1.0e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch both Cholesky candidates and select entirely on device."""
    fallback_relative_ridge = max(fallback_relative_ridge, relative_ridge)
    identity = torch.eye(
        matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
    ).expand(matrix.shape[0], -1, -1)
    primary, primary_info = torch.linalg.cholesky_ex(
        matrix + (scale * relative_ridge)[:, None, None] * identity,
        check_errors=False,
    )
    fallback, fallback_info = torch.linalg.cholesky_ex(
        matrix + (scale * fallback_relative_ridge)[:, None, None] * identity,
        check_errors=False,
    )
    primary_ok = (primary_info == 0)[:, None, None]
    fallback_ok = (fallback_info == 0)[:, None, None]
    selected = torch.where(
        primary_ok,
        primary,
        torch.where(fallback_ok, fallback, identity),
    )
    return selected, fallback_info != 0


def warm_cholesky_metric_factor(
    metric: torch.Tensor,
    previous_basis: torch.Tensor,
    previous_factor: torch.Tensor,
    rank: int,
    *,
    power_iters: int = 2,
    cholesky_qr_ridge: float = 1.0e-5,
    rayleigh_ridge: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official-style Cholesky-QR2 plus Cholesky Rayleigh--Ritz update.

    Failed heads reuse their immediately previous finite factor and basis with
    ``torch.where``; no ``.item()`` or host synchronization is needed.
    """
    expected = (metric.shape[0], metric.shape[1], min(rank, metric.shape[1]))
    if previous_basis.shape != expected or previous_factor.shape != expected:
        raise ValueError("SparsePR warm factor state shape mismatch")
    metric_f = 0.5 * (metric.float() + metric.float().transpose(1, 2))
    basis = previous_basis.float()
    basis = basis / torch.linalg.vector_norm(
        basis, dim=1, keepdim=True
    ).clamp_min(1.0e-12)
    recovered = torch.zeros(
        metric.shape[0], dtype=torch.bool, device=metric.device
    )
    for iteration in range(power_iters):
        basis, failed = _cholesky_qr(
            torch.bmm(metric_f, basis), cholesky_qr_ridge
        )
        recovered |= failed
        if iteration == power_iters - 1:
            basis, failed = _cholesky_qr(
                basis, max(cholesky_qr_ridge, 1.0e-5)
            )
            recovered |= failed

    rayleigh = torch.bmm(
        basis.transpose(1, 2), torch.bmm(metric_f, basis)
    )
    rayleigh = 0.5 * (rayleigh + rayleigh.transpose(1, 2))
    normalizer = (
        rayleigh.diagonal(dim1=1, dim2=2)
        .mean(dim=1)
        .clamp_min(1.0e-12)
    )
    rayleigh_factor, failed = _cholesky_with_device_fallback(
        rayleigh,
        normalizer,
        rayleigh_ridge,
    )
    recovered |= failed
    factor = torch.bmm(basis, rayleigh_factor) / torch.sqrt(normalizer)[
        :, None, None
    ]
    recovered |= ~torch.isfinite(factor).all(dim=(1, 2))
    recovered |= ~torch.isfinite(basis).all(dim=(1, 2))
    choose_previous = recovered[:, None, None]
    factor = torch.where(choose_previous, previous_factor.float(), factor)
    basis = torch.where(choose_previous, previous_basis.float(), basis)
    return factor, basis, recovered


__all__ = ["warm_cholesky_metric_factor"]

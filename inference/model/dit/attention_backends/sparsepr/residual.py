# Adapted from SparsePR revision e46646f.
# SPDX-License-Identifier: Apache-2.0
"""M64/r16 reduced-rank residual repair used by the SparsePR paper path."""

from __future__ import annotations

import torch

RepairState = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def fit_reduced_rank_residual(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    *,
    rank: int,
    ridge: float,
    weights: torch.Tensor,
    basis_source: str = "raw_residual",
) -> RepairState:
    """Fit the official weighted dual-ridge reduced-rank estimator in FP32."""
    if x_fit.ndim != 3 or y_fit.ndim != 3 or x_fit.shape[:2] != y_fit.shape[:2]:
        raise ValueError("SparsePR RRR expects X/Y with matching [H,M,*] dimensions")
    if not 0 < rank <= min(y_fit.shape[1], y_fit.shape[2]):
        raise ValueError("SparsePR RRR rank exceeds probe/output dimensions")
    if basis_source not in {"raw_residual", "fitted_residual"}:
        raise ValueError("Unknown SparsePR RRR basis source")
    device_type = x_fit.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        x = x_fit.float()
        y = y_fit.float()
        w = weights.float().unsqueeze(2)
        w_sum = w.sum(1, keepdim=True).clamp_min(1e-20)
        x_mean = (w * x).sum(1, keepdim=True) / w_sum
        x_var = (w * (x - x_mean).square()).sum(1, keepdim=True) / w_sum
        x_scale = x_var.sqrt().clamp_min(1e-5)
        y_mean = (w * y).sum(1, keepdim=True) / w_sum
        x_normalized = (x - x_mean) / x_scale
        y_centered = y - y_mean
        sqrt_w = w.sqrt()
        x_weighted = x_normalized * sqrt_w
        y_weighted = y_centered * sqrt_w

        # M=64 < D=128: solve in probe-row space and never materialize D x D.
        gram = torch.bmm(x_weighted, x_weighted.transpose(1, 2))
        regularized = gram.clone()
        regularized.diagonal(dim1=1, dim2=2).add_(float(ridge))
        factor, info = torch.linalg.cholesky_ex(regularized, check_errors=False)
        dual = torch.cholesky_solve(y_weighted, factor)
        weighted_fitted = y_weighted - float(ridge) * dual
        basis_input = y_weighted if basis_source == "raw_residual" else weighted_fitted

        # Exact skinny right basis from the M x M row Gram.
        row_gram = torch.bmm(basis_input, basis_input.transpose(1, 2))
        eigenvalues, left_vectors = torch.linalg.eigh(row_gram)
        left_vectors = left_vectors[:, :, -rank:]
        scales = eigenvalues[:, -rank:].clamp_min(1e-20).rsqrt().unsqueeze(1)
        basis = (
            torch.bmm(basis_input.transpose(1, 2), left_vectors) * scales
        ).contiguous()
        left = torch.bmm(
            x_weighted.transpose(1, 2),
            torch.bmm(dual, basis),
        ).contiguous()

        # A failed Cholesky is represented as non-finite state and recovered
        # per head by ``recover_repair_state`` without a host synchronization.
        failed = info != 0
        left = torch.where(failed[:, None, None], torch.full_like(left, float("nan")), left)
        return x_mean, x_scale, y_mean, left, basis


def recover_repair_state(
    state: RepairState,
    previous: RepairState | None,
) -> tuple[RepairState, torch.Tensor]:
    """Reuse the previous finite estimator, or zero only a failed cold head."""
    heads = state[0].shape[0]
    failed = torch.zeros(heads, dtype=torch.bool, device=state[0].device)
    for value in state:
        failed |= ~torch.isfinite(value).reshape(heads, -1).all(1)
    if previous is None:
        fallback: RepairState = (
            torch.zeros_like(state[0]),
            torch.ones_like(state[1]),
            torch.zeros_like(state[2]),
            torch.zeros_like(state[3]),
            torch.zeros_like(state[4]),
        )
    else:
        fallback = previous
    selected = tuple(
        torch.where(
            failed.reshape(heads, *((1,) * (current.ndim - 1))),
            old,
            current,
        )
        for current, old in zip(state, fallback, strict=True)
    )
    return selected, failed  # type: ignore[return-value]


def predict_reduced_rank_residual(x: torch.Tensor, state: RepairState) -> torch.Tensor:
    x_mean, x_scale, y_mean, left, basis = state
    coordinates = torch.bmm((x.float() - x_mean) / x_scale, left)
    return torch.bmm(coordinates, basis.transpose(1, 2)) + y_mean


__all__ = [
    "RepairState",
    "fit_reduced_rank_residual",
    "predict_reduced_rank_residual",
    "recover_repair_state",
]

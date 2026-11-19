"""Tile-aligned query grouping used by the MC-Sparse reference backend.

Adapted from ``mcsparse/kernels/flash_mcsa/pca_group.py`` in the official
MC-Sparse repository (Apache-2.0).  Only the inference-side permutation
construction required by MagiHuman is retained here.
"""

from __future__ import annotations

import torch
from torch import Tensor


def identity_perm(heads: int, tokens: int, device: torch.device) -> Tensor:
    return torch.arange(tokens, device=device).unsqueeze(0).expand(heads, tokens).contiguous()


def apply_perm(x: Tensor, perm: Tensor) -> Tensor:
    """Apply a per-head permutation to ``[H, N, D]`` data."""
    return x.gather(1, perm.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def invert_perm(x: Tensor, perm: Tensor) -> Tensor:
    """Undo :func:`apply_perm`."""
    inverse = perm.argsort(dim=1)
    return x.gather(1, inverse.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _split_block_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for lo, hi in ranges:
        if hi - lo <= 1:
            result.append((lo, hi))
        else:
            mid = lo + (hi - lo) // 2
            result.extend(((lo, mid), (mid, hi)))
    return result


@torch.no_grad()
def pddp_group_perm(
    x: Tensor,
    tile_size: int = 128,
    power_iters: int = 4,
    seed: int = 0,
) -> Tensor:
    """Return a Fast-PDDP permutation for ``x`` in ``[H, N, D]`` layout.

    Recursive splits always land on ``tile_size`` boundaries.  Consequently
    every leaf except the final one contains exactly one execution tile.
    Only token indices are propagated between tree depths; the full grouped
    query tensor is never kept as persistent metadata.
    """
    if x.ndim != 3:
        raise ValueError("PDDP expects [heads, tokens, dim]")
    if tile_size < 1 or power_iters < 1:
        raise ValueError("PDDP tile size and power iterations must be positive")

    heads, tokens, dim = x.shape
    if tokens <= tile_size:
        return identity_perm(heads, tokens, x.device)

    blocks = (tokens + tile_size - 1) // tile_size
    perm = identity_perm(heads, tokens, x.device)
    # Padding maps to this zero row and therefore contributes nothing to the
    # centered Gram matrix or the projected ordering.
    x_ext = torch.cat((x, x.new_zeros(heads, 1, dim)), dim=1)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    ranges = [(0, blocks)]

    while any(hi - lo > 1 for lo, hi in ranges):
        starts = torch.tensor([lo * tile_size for lo, _ in ranges], device=x.device)
        ends = torch.tensor(
            [min(hi * tile_size, tokens) for _, hi in ranges], device=x.device
        )
        sizes = ends - starts
        nodes = len(ranges)
        max_length = int(sizes.max().item())

        offsets = torch.arange(max_length, device=x.device)
        positions = starts[:, None] + offsets[None]
        valid = offsets[None] < sizes[:, None]
        safe_positions = torch.where(valid, positions, torch.zeros_like(positions))
        flat_positions = safe_positions.reshape(1, -1).expand(heads, -1)
        ids = perm.gather(1, flat_positions)
        flat_valid = valid.reshape(1, -1).expand(heads, -1)
        ids = torch.where(flat_valid, ids, torch.full_like(ids, tokens))
        ids = ids.reshape(heads * nodes, max_length)

        gathered = x_ext.gather(
            1,
            ids.reshape(heads, nodes * max_length)
            .unsqueeze(-1)
            .expand(-1, -1, dim),
        ).reshape(heads * nodes, max_length, dim)
        valid_rows = flat_valid.reshape(heads * nodes, max_length)
        counts = valid_rows.sum(1, keepdim=True).clamp_min(1).float()
        sums = gathered.sum(1, dtype=torch.float32)
        gram = torch.bmm(gathered.transpose(1, 2), gathered).float()
        gram -= (sums.unsqueeze(2) * sums.unsqueeze(1)) / counts.unsqueeze(2)

        direction = torch.randn(
            heads * nodes, dim, device=x.device, generator=generator
        )
        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        for _ in range(power_iters):
            direction = torch.bmm(gram, direction.unsqueeze(-1)).squeeze(-1)
            direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        projection = torch.bmm(
            gathered, direction.to(gathered.dtype).unsqueeze(-1)
        ).squeeze(-1).float()
        projection = torch.where(
            valid_rows, projection, torch.full_like(projection, float("inf"))
        )
        order = projection.argsort(dim=1)
        sorted_ids = ids.gather(1, order).reshape(heads, nodes * max_length)
        keep = flat_valid[0]
        perm = perm.scatter(1, flat_positions[:, keep], sorted_ids[:, keep])
        ranges = _split_block_ranges(ranges)

    return perm


__all__ = ["apply_perm", "identity_perm", "invert_perm", "pddp_group_perm"]

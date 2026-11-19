"""SAP-compatible Lloyd iteration; CUDA kernels live in ``kernels/``."""
import torch


def kmeans(samples, num_clusters, iterations, initial=None, *, backend="triton", tol=1e-4):
    """Match SAP's dtype, global RNG, stopping rule and returned assignment.

    CPU/PyTorch is a diagnostic reference. CUDA defaults to the vendored SAP
    Triton assignment and sorted FP32 centroid reduction. There is no extra
    assignment/update after the requested iterations. As in SAP, convergence
    is checked before replacing the previous centers.
    """
    if iterations < 1 or num_clusters < 1 or samples.shape[1] < 1:
        raise ValueError("k-means needs positive iterations, clusters and token count")
    if backend not in {"triton", "torch"}:
        raise ValueError("Unknown k-means backend")
    samples = samples.contiguous()
    heads, length, dim = samples.shape
    clusters = min(num_clusters, length)
    if initial is None:
        seeds = torch.randint(length, (heads, clusters), device=samples.device)
        centers = samples.gather(1, seeds[..., None].expand(-1, -1, dim))
    else:
        if initial.shape != (heads, clusters, dim):
            raise ValueError("Cached k-means centers have the wrong shape")
        centers = initial.to(device=samples.device, dtype=samples.dtype).contiguous()
    x_sq = (samples ** 2).sum(-1)
    use_triton = samples.is_cuda and backend == "triton"
    if use_triton:
        from .kernels.sap_kmeans import euclid_assign_triton, triton_centroid_update_sorted_euclid
    for _ in range(iterations):
        if use_triton:
            labels = euclid_assign_triton(samples, centers, x_sq)
            updated, counts = triton_centroid_update_sorted_euclid(samples, labels, centers)
        else:
            # FP32 accumulation, then round centers back to the input dtype.
            distance = (x_sq.float()[..., None]
                        + (centers.float() ** 2).sum(-1)[:, None]
                        - 2 * torch.bmm(samples.float(), centers.float().transpose(1, 2)))
            labels = distance.clamp_min(0).argmin(-1)
            counts = torch.zeros((heads, clusters), dtype=torch.int32, device=samples.device)
            counts.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.int32))
            sums = torch.zeros((heads, clusters, dim), dtype=torch.float32, device=samples.device)
            sums.scatter_add_(1, labels[..., None].expand(-1, -1, dim), samples.float())
            updated = torch.where(counts[..., None] > 0,
                                  sums / counts.clamp_min(1)[..., None], centers.float()).to(samples.dtype)
        if (updated - centers).norm(dim=-1).max() < tol:
            break
        centers = updated
    return labels, centers.detach(), counts

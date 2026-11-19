"""CuTe online k-means and label grouping for V-Smooth on SM120."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack
import torch


DIM = 128
WARPS = 4
THREADS = WARPS * 32
_compiled = {}


def _ptr(tensor, coord):
    return tensor.iterator + cute.crd2idx(coord, tensor.layout)


@cute.kernel
def _init_centers(values: cute.Tensor, initial: cute.Tensor, centers: cute.Tensor,
                  sequence: cutlass.Constexpr, heads: cutlass.Constexpr,
                  clusters: cutlass.Constexpr, warm: cutlass.Constexpr):
    channel, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    head = block // clusters
    cluster = block - head * clusters
    if channel < DIM:
        if const_expr(warm):
            centers[head, cluster, channel] = initial[head, cluster, channel]
        else:
            token = cluster * (sequence - 1) // max(clusters - 1, 1)
            centers[head, cluster, channel] = values[head, token, channel].to(Float32)


@cute.kernel
def _zero_stats(sums: cute.Tensor, counts: cute.Tensor,
                heads: cutlass.Constexpr, clusters: cutlass.Constexpr):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * 256 + thread
    total = heads * clusters * DIM
    if linear < total:
        channel = linear % DIM
        cluster = (linear // DIM) % clusters
        head = linear // (clusters * DIM)
        sums[head, cluster, channel] = Float32(0.0)
    if linear < heads * clusters:
        counts[linear // clusters, linear % clusters] = Int32(0)


@cute.kernel
def _zero_counts(counts: cute.Tensor, heads: cutlass.Constexpr,
                 clusters: cutlass.Constexpr):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * 256 + thread
    if linear < heads * clusters:
        counts[linear // clusters, linear % clusters] = Int32(0)


@cute.kernel
def _assign(values: cute.Tensor, centers: cute.Tensor, labels: cute.Tensor,
            sums: cute.Tensor, counts: cute.Tensor,
            sequence: cutlass.Constexpr, heads: cutlass.Constexpr,
            clusters: cutlass.Constexpr, accumulate: cutlass.Constexpr):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = thread % 32
    warp = thread // 32
    linear = block * WARPS + warp
    if linear < heads * sequence:
        head = linear // sequence
        token = linear - head * sequence
        best_score = -Float32.inf
        best_cluster = Int32(0)
        values_reg = cute.make_rmem_tensor((4,), Float32)
        for item in cutlass.range_constexpr(4):
            channel = lane + item * 32
            values_reg[item] = values[head, token, channel].to(Float32)
        for cluster in cutlass.range_constexpr(clusters):
            score = Float32(0.0)
            for item in cutlass.range_constexpr(4):
                channel = lane + item * 32
                center = centers[head, cluster, channel]
                score += values_reg[item] * center - Float32(0.5) * center * center
            score = cute.arch.warp_reduction_sum(score)
            if score > best_score:
                best_score = score
                best_cluster = Int32(cluster)
        if lane == 0:
            labels[head, token] = best_cluster
            cute.arch.atomic_add(_ptr(counts, (head, best_cluster)), Int32(1))
        if const_expr(accumulate):
            for item in cutlass.range_constexpr(4):
                channel = lane + item * 32
                cute.arch.atomic_add(
                    _ptr(sums, (head, best_cluster, channel)), values_reg[item]
                )


@cute.kernel
def _finish_centers(sums: cute.Tensor, counts: cute.Tensor, centers: cute.Tensor,
                    heads: cutlass.Constexpr, clusters: cutlass.Constexpr):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * 256 + thread
    total = heads * clusters * DIM
    if linear < total:
        channel = linear % DIM
        cluster = (linear // DIM) % clusters
        head = linear // (clusters * DIM)
        count = counts[head, cluster]
        if count > 0:
            centers[head, cluster, channel] = (
                sums[head, cluster, channel] / count.to(Float32)
            )


@cute.kernel
def _prefix(counts: cute.Tensor, positions: cute.Tensor,
            clusters: cutlass.Constexpr):
    thread, _, _ = cute.arch.thread_idx()
    head, _, _ = cute.arch.block_idx()
    if thread == 0:
        offset = Int32(0)
        for cluster in cutlass.range_constexpr(clusters):
            positions[head, cluster] = offset
            offset += counts[head, cluster]


@cute.kernel
def _scatter(labels: cute.Tensor, positions: cute.Tensor, permutation: cute.Tensor,
             sequence: cutlass.Constexpr, heads: cutlass.Constexpr):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * 256 + thread
    if linear < heads * sequence:
        head = linear // sequence
        token = linear - head * sequence
        cluster = labels[head, token]
        output = cute.arch.atomic_add(_ptr(positions, (head, cluster)), Int32(1))
        permutation[head, output] = Int32(token)


@cute.jit
def _group(values: cute.Tensor, initial: cute.Tensor, centers: cute.Tensor,
           labels: cute.Tensor, sums: cute.Tensor, counts: cute.Tensor,
           positions: cute.Tensor, permutation: cute.Tensor,
           sequence: cutlass.Constexpr, heads: cutlass.Constexpr,
           clusters: cutlass.Constexpr, warm: cutlass.Constexpr,
           stream: cuda.CUstream):
    _init_centers(values, initial, centers, sequence, heads, clusters, warm).launch(
        grid=(heads * clusters, 1, 1), block=(THREADS, 1, 1), stream=stream
    )
    iterations = 1 if const_expr(warm) else 2
    for _ in cutlass.range_constexpr(iterations):
        _zero_stats(sums, counts, heads, clusters).launch(
            grid=((heads * clusters * DIM + 255) // 256, 1, 1),
            block=(256, 1, 1), stream=stream,
        )
        _assign(
            values, centers, labels, sums, counts,
            sequence, heads, clusters, True,
        ).launch(
            grid=((heads * sequence + WARPS - 1) // WARPS, 1, 1),
            block=(THREADS, 1, 1), stream=stream,
        )
        _finish_centers(sums, counts, centers, heads, clusters).launch(
            grid=((heads * clusters * DIM + 255) // 256, 1, 1),
            block=(256, 1, 1), stream=stream,
        )
    _zero_counts(counts, heads, clusters).launch(
        grid=((heads * clusters + 255) // 256, 1, 1),
        block=(256, 1, 1), stream=stream,
    )
    _assign(
        values, centers, labels, sums, counts,
        sequence, heads, clusters, False,
    ).launch(
        grid=((heads * sequence + WARPS - 1) // WARPS, 1, 1),
        block=(THREADS, 1, 1), stream=stream,
    )
    _prefix(counts, positions, clusters).launch(
        grid=(heads, 1, 1), block=(32, 1, 1), stream=stream
    )
    _scatter(labels, positions, permutation, sequence, heads).launch(
        grid=((heads * sequence + 255) // 256, 1, 1),
        block=(256, 1, 1), stream=stream,
    )


def group_values_cute(values: torch.Tensor, *, initial_centers=None):
    """Return a cluster-contiguous permutation and updated FP32 centroids."""
    heads, sequence, dim = values.shape
    clusters = min(32, max(1, sequence // 128))
    warm = initial_centers is not None
    values = values.contiguous()
    centers = torch.empty((heads, clusters, dim), device=values.device, dtype=torch.float32)
    if warm:
        initial = initial_centers.contiguous()
    else:
        initial = centers
    labels = torch.empty((heads, sequence), device=values.device, dtype=torch.int32)
    sums = torch.empty_like(centers)
    counts = torch.empty((heads, clusters), device=values.device, dtype=torch.int32)
    positions = torch.empty_like(counts)
    permutation = torch.empty_like(labels)
    stream = cuda.CUstream(torch.cuda.current_stream(values.device).cuda_stream)
    tensors = (values, initial, centers, labels, sums, counts, positions, permutation)
    args = tuple(from_dlpack(t) for t in tensors) + (
        sequence, heads, clusters, warm, stream,
    )
    key = (values.device.index, sequence, heads, clusters, warm)
    if key not in _compiled:
        _compiled[key] = cute.compile(_group, *args)
    _compiled[key](*args[:8], stream)
    return permutation, centers

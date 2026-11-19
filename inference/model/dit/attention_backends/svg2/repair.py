"""Experimental per-query coverage repair, grouped for existing executors."""
import math
import torch


def repair_query_routes(q_video, q_labels, k_centers, k_counts, cluster_map, threshold):
    from .core import _top_p_map

    heads, length, _ = q_video.shape
    kv_for_q = torch.arange(heads, device=q_video.device) // (heads // k_centers.shape[0])
    logits = torch.bmm(q_video.float(), k_centers[kv_for_q].float().transpose(1, 2)) / math.sqrt(q_video.shape[-1])
    logits += k_counts[kv_for_q, None, :].clamp_min(1).float().log()
    logits.masked_fill_(k_counts[kv_for_q, None, :] == 0, -torch.inf)
    selected = cluster_map[torch.arange(heads, device=q_video.device)[:, None], q_labels.long()]
    flagged = (logits.softmax(-1) * selected).sum(-1) < threshold
    selected |= flagged[..., None] & _top_p_map(q_video.float(), k_centers.float(), k_counts, .99, .1)
    groups = [torch.unique(selected[h], dim=0, return_inverse=True, return_counts=True) for h in range(heads)]
    max_groups = max(group[0].shape[0] for group in groups)
    maps = torch.zeros(heads, max_groups, k_centers.shape[1], device=q_video.device, dtype=torch.bool)
    counts = torch.zeros(heads, max_groups, device=q_video.device, dtype=torch.int32)
    labels = torch.stack([group[1] for group in groups])
    for head, (patterns, _, sizes) in enumerate(groups):
        maps[head, :len(patterns)] = patterns
        counts[head, :len(patterns)] = sizes.int()
    return labels, counts, maps

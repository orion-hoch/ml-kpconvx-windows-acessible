#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# Modifications Copyright (C) 2026 Orion Hoch: PyKeOps replaced by torch-cluster
# (Windows support, no runtime CUDA JIT). Same contract as the keops version.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: gpu_neigbors.py
#       > Neighbors search functions on gpu
#

from typing import Tuple

import torch
from torch import Tensor

try:
    import torch_cluster
except ImportError as e:
    raise ImportError(
        "KPConvX (keops-free fork) requires torch-cluster. Install the wheel "
        "matching your torch/CUDA build, e.g.:\n"
        "  pip install torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html"
    ) from e

from pointcept.models.kpconvx.utils.batch_conversion import batch_to_pack, pack_to_batch, pack_to_list, list_to_pack


# ----------------------------------------------------------------------------------------------------------------------
#
#           Implementation of k-nn search with torch-cluster
#       \****************************************************/
#


def _flatten_batched(points: Tensor) -> Tuple[Tensor, Tensor, int]:
    """(B, N, C) -> packed (B*N, C) + batch vector."""
    B, N, _ = points.shape
    batch = torch.arange(B, device=points.device).repeat_interleave(N)
    return points.reshape(B * N, points.shape[-1]), batch, N


def _pairs_to_table(q_idx: Tensor, s_idx: Tensor, d2: Tensor, num_queries: int, k: int,
                    fill_index: int) -> Tuple[Tensor, Tensor]:
    """Pairs -> dense row-sorted (Q, k) tables; ties break by support index;
    empty slots get dist=inf / index=fill_index. Sort order is enforced here,
    never assumed from torch-cluster."""
    device = d2.device

    if s_idx.numel():
        perm = torch.argsort(q_idx * (int(s_idx.max()) + 2) + s_idx)
        q_idx, s_idx, d2 = q_idx[perm], s_idx[perm], d2[perm]

    counts = torch.bincount(q_idx, minlength=num_queries)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(q_idx.shape[0], device=device) - starts[q_idx]

    table_d2 = torch.full((num_queries, k), float('inf'), dtype=d2.dtype, device=device)
    table_idx = torch.full((num_queries, k), fill_index, dtype=torch.long, device=device)
    table_d2[q_idx, slot] = d2
    table_idx[q_idx, slot] = s_idx

    table_d2, order = torch.sort(table_d2, dim=-1, stable=True)
    table_idx = torch.gather(table_idx, -1, order)
    return table_d2, table_idx


@torch.no_grad()
def tc_radius_count(q_points: Tensor, s_points: Tensor, radius: float) -> Tensor:
    """
    Count neighbors strictly inside radius (d < radius, like keops did).
    Args:
        q_points (Tensor): (N, C) or (B, N, C)
        s_points (Tensor): (M, C) or (B, M, C)
        radius (float)
    Returns:
        radius_counts (Tensor): (N,) or (B, N)
    """
    batched = q_points.dim() == 3
    if batched:
        B = q_points.shape[0]
        q_flat, q_batch, Nq = _flatten_batched(q_points)
        s_flat, s_batch, _ = _flatten_batched(s_points)
    else:
        q_flat, s_flat, q_batch, s_batch = q_points, s_points, None, None

    # a saturated max_num_neighbors cap would silently truncate calibration counts
    cap = 64
    while True:
        edge = torch_cluster.radius(x=s_flat, y=q_flat, r=radius,
                                    batch_x=s_batch, batch_y=q_batch,
                                    max_num_neighbors=cap)
        raw = torch.bincount(edge[0], minlength=q_flat.shape[0])
        if raw.numel() == 0 or raw.max().item() < cap:
            break
        cap *= 2

    # torch-cluster includes d == radius; keops counted strictly
    d2 = ((q_flat[edge[0]] - s_flat[edge[1]]) ** 2).sum(-1)
    strict = edge[0][d2 < radius * radius]
    counts = torch.bincount(strict, minlength=q_flat.shape[0])
    if batched:
        counts = counts.reshape(B, Nq)
    return counts


@torch.no_grad()
def tc_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """
    kNN: SQUARED distances, rows sorted ascending, indices local per batch element.
    Args:
        q_points (Tensor): (N, C) or (B, N, C)
        s_points (Tensor): (M, C) or (B, M, C)
        k (int)
    Returns:
        knn_d2 (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """
    batched = q_points.dim() == 3
    if batched:
        B = q_points.shape[0]
        q_flat, q_batch, Nq = _flatten_batched(q_points)
        s_flat, s_batch, Ms = _flatten_batched(s_points)
    else:
        q_flat, s_flat, q_batch, s_batch = q_points, s_points, None, None

    edge = torch_cluster.knn(x=s_flat, y=q_flat, k=k, batch_x=s_batch, batch_y=q_batch)
    q_idx, s_idx = edge[0], edge[1]
    d2 = ((q_flat[q_idx] - s_flat[s_idx]) ** 2).sum(-1)

    table_d2, table_idx = _pairs_to_table(q_idx, s_idx, d2, q_flat.shape[0], k,
                                          fill_index=0)

    if batched:
        # global -> per-element indices; inf (empty) slots must not go negative
        offsets = (torch.arange(B, device=table_idx.device) * Ms).repeat_interleave(Nq)
        table_idx = torch.where(torch.isinf(table_d2), table_idx,
                                table_idx - offsets.view(-1, 1))
        table_d2 = table_d2.reshape(B, Nq, k)
        table_idx = table_idx.reshape(B, Nq, k)
    return table_d2, table_idx


@torch.no_grad()
def knn(q_points: Tensor,
        s_points: Tensor,
        k: int,
        dilation: int = 1,
        distance_limit: float = None,
        return_distance: bool = False,
        remove_nearest: bool = False,
        transposed: bool = False,
        padding_mode: str = "nearest",
        inf: float = 1e10):
    """
    Compute the kNNs of the points in `q_points` from the points in `s_points`.
    Args:
        s_points (Tensor): coordinates of the support points, (*, C, N) or (*, N, C).
        q_points (Tensor): coordinates of the query points, (*, C, M) or (*, M, C).
        k (int): number of nearest neighbors to compute.
        dilation (int): dilation for dilated knn.
        distance_limit (float=None): if further than this radius, the neighbors are replaced according to `padding_mode`.
        return_distance (bool=False): whether return distances.
        remove_nearest (bool=True) whether remove the nearest neighbor (itself).
        transposed (bool=False): if True, the points shape is (*, C, N).
        padding_mode (str='nearest'): padding mode for neighbors further than distance radius. ('nearest', 'empty').
        inf (float=1e10): infinity value for padding.
    Returns:
        knn_distances (Tensor): The distances of the kNNs, (*, M, k).
        knn_indices (LongTensor): The indices of the kNNs, (*, M, k).
    """
    if transposed:
        q_points = q_points.transpose(-1, -2)  # (*, C, N) -> (*, N, C)
        s_points = s_points.transpose(-1, -2)  # (*, C, M) -> (*, M, C)

    num_s_points = s_points.shape[-2]

    dilated_k = (k - 1) * dilation + 1
    if remove_nearest:
        dilated_k += 1
    final_k = min(dilated_k, num_s_points)

    knn_distances, knn_indices = tc_knn(q_points, s_points, final_k)  # (*, N, k)

    if remove_nearest:
        knn_distances = knn_distances[..., 1:]
        knn_indices = knn_indices[..., 1:]

    if dilation > 1:
        knn_distances = knn_distances[..., ::dilation]
        knn_indices = knn_indices[..., ::dilation]

    knn_distances = knn_distances.contiguous()
    knn_indices = knn_indices.contiguous()

    if distance_limit is not None:
        assert padding_mode in ["nearest", "empty"]
        knn_masks = torch.ge(knn_distances, distance_limit)
        if padding_mode == "nearest":
            knn_distances[knn_masks] = knn_distances[..., 0]
            knn_indices[knn_masks] = knn_indices[..., 0]
        else:
            knn_distances[knn_masks] = inf
            knn_indices[knn_masks] = num_s_points

    if return_distance:
        return knn_distances, knn_indices

    return knn_indices

@torch.no_grad()
def radius_search_pack_mode(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit, shadow=False, inf=1e8, return_dist=False):
    """Radius search in pack mode (fast version).
    Args:
        q_points (Tensor): query points (M, 3).
        s_points (Tensor): support points (N, 3).
        q_lengths (LongTensor): the numbers of query points in the batch (B,).
        s_lengths (LongTensor): the numbers of support points in the batch (B,).
        radius (float): radius radius.
        neighbor_limit (int): neighbor radius.
        inf (float=1e10): infinity value.
    Returns:
        neighbor_indices (LongTensor): the indices of the neighbors. Equal to N if not exist.
    """
    device = q_points.device
    # lengths may arrive as lists or off-device; repeat_interleave is strict
    q_lengths = torch.as_tensor(q_lengths, device=device).long()
    s_lengths = torch.as_tensor(s_lengths, device=device).long()
    q_batch = torch.arange(q_lengths.shape[0], device=device).repeat_interleave(q_lengths)
    s_batch = torch.arange(s_lengths.shape[0], device=device).repeat_interleave(s_lengths)

    edge = torch_cluster.knn(x=s_points, y=q_points, k=neighbor_limit,
                             batch_x=s_batch, batch_y=q_batch)
    q_idx, s_idx = edge[0], edge[1]
    d2 = ((q_points[q_idx] - s_points[s_idx]) ** 2).sum(-1)

    knn_distances, knn_indices = _pairs_to_table(q_idx, s_idx, d2, q_points.shape[0],
                                                 neighbor_limit,
                                                 fill_index=s_points.shape[0])

    # Limit for shadow neighbors
    if shadow:
        # shadow everything outside radius
        shadow_limit = radius ** 2
    else:
        # keep knns, only shadow invalid indices like when s_pts.shape < K
        shadow_limit = inf / 10

    # Fill shadow neighbors values
    knn_masks = torch.gt(knn_distances, shadow_limit)
    knn_indices = knn_indices.masked_fill(knn_masks, s_points.shape[0])  # (M, K)

    if return_dist:
        return knn_indices, torch.sqrt(knn_distances)

    return knn_indices

@torch.no_grad()
def radius_search_list_mode(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit, shadow=False):
    """
    Radius search in pack mode (fast version). This function is actually a knn search
    but with option to shadow furthest neighbors (d > radius).
    Args:
        q_points (Tensor): query points (M, 3).
        s_points (Tensor): support points (N, 3).
        q_lengths (LongTensor): the numbers of query points in the batch (B,).
        s_lengths (LongTensor): the numbers of support points in the batch (B,).
        radius (float): search radius, only used for shadowing furthest neighbors.
        neighbor_limit (int): max number of neighbors, actual knn limit used for computing neighbors.
        inf (float=1e10): infinity value.
    Returns:
        neighbor_indices (LongTensor): the indices of the neighbors. Equal to N if not exist.
    """

    # pack to batch
    batch_q_list = pack_to_list(q_points, q_lengths)  # (B)(?, 3)
    batch_s_list = pack_to_list(s_points, s_lengths)  # (B)(?, 3)

    # knn on each element of the list (B)[(?, K), (?, K)]
    knn_dists_inds = [tc_knn(b_q_pts, b_s_pts, neighbor_limit)
                          for b_q_pts, b_s_pts in zip(batch_q_list, batch_s_list)]

    # Accumualte indices
    b_start_ind = torch.cumsum(s_lengths, dim=0) - s_lengths
    knn_inds_list = [b_knn_inds + b_start_ind[i] for i, (_, b_knn_inds) in enumerate(knn_dists_inds)]

    # Convert list to pack (B)[(?, K) -> (M, K)
    knn_indices, _ = list_to_pack(knn_inds_list)

    # Apply shadow inds (optional because knn to far away from convolution kernel will be ignored anyway)
    if shadow:
        knn_dists_list = [b_knn_dists for b_knn_dists, _ in knn_dists_inds]
        knn_dists, _ = list_to_pack(knn_dists_list)
        knn_masks = torch.gt(knn_dists, radius**2)
        knn_indices.masked_fill_(knn_masks, s_points.shape[0])

    return knn_indices

@torch.no_grad()
def tiled_knn(q_points: Tensor, s_points: Tensor, k: int, tile_size: float, margin: float) -> Tuple[Tensor, Tensor]:
    """
    Divide the query and support in tiles and .
    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k           (int): number of neighbors
        tile_size (float): size of the square tiles
        margin    (float): margin for tiling the support (must be > max_knn_dist)
    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """

    # Get limits
    min_q, _ = torch.min(q_points, dim=-2)
    min_s, _ = torch.min(s_points, dim=-2)
    min_p = torch.minimum(min_q, min_s) - margin
    max_q, _ = torch.max(q_points, dim=-2)
    max_s, _ = torch.max(s_points, dim=-2)
    max_p = torch.maximum(max_q, max_s) + margin

    # Create tiles
    tile_N = torch.ceil((max_p - min_p) / tile_size).type(torch.long)

    # Init neighbors and dists
    knn_indices = torch.zeros((q_points.shape[0],), dtype=torch.long, device=q_points.device)
    knn_distances = torch.zeros((q_points.shape[0],), dtype=q_points.dtype, device=q_points.device) + 1e8
    s_inds = torch.arange(s_points.shape[0], dtype=torch.long, device=q_points.device)

    # Loop on tiles
    for xi in range(tile_N[0].item()):
        for yi in range(tile_N[1].item()):
            for zi in range(tile_N[2].item()):

                # Get tile limits
                tile_min = min_p + tile_size * torch.tensor([xi, yi, zi],
                                                            dtype=q_points.dtype,
                                                            device=q_points.device)
                tile_max = tile_min + tile_size + 0.1* margin
                q_mask = torch.logical_and(torch.all(q_points > tile_min, dim=-1),
                                           torch.all(q_points <= tile_max, dim=-1))
                s_mask = torch.logical_and(torch.all(s_points > tile_min - margin, dim=-1),
                                           torch.all(s_points <= tile_max + margin, dim=-1))

                # Get points in the tile
                q_pts = q_points[q_mask]
                s_pts = s_points[s_mask]
                if q_pts.shape[0] < 1:
                    continue
                if s_pts.shape[0] < 1:
                    raise ValueError('got queries but no support points')

                # Get knn
                knn_d, knn_i = tc_knn(q_pts, s_pts, k)

                # (*, N_i),  (*, N_i) values in M_i
                knn_distances[q_mask] = knn_d.view((-1))
                knn_indices[q_mask] = s_inds[s_mask][knn_i.view((-1))]

    return knn_distances, knn_indices

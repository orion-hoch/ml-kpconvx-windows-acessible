#
# Parity tests for the keops-free neighbor search (utils/gpu_neigbors.py).
# Copyright (C) 2026 Orion Hoch. MIT License.
#
# Verifies the torch-cluster backend against a brute-force reference on CPU:
# exact same contract the original keops backend provided. Squared distances,
# rows sorted ascending, indices local per batch element, shadow index = N.
#
# 
#

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu_neigbors import tc_knn, tc_radius_count, radius_search_pack_mode

torch.manual_seed(0)


def brute_knn(q, s, k):
    """Reference: full distance matrix + tie-break by support index."""
    d2 = torch.cdist(q.double(), s.double()) ** 2
    k_eff = min(k, s.shape[0])
    order = torch.argsort(d2 + torch.arange(s.shape[0]).double() * 1e-12, dim=-1, stable=True)
    idx = order[:, :k_eff]
    dist = torch.gather(d2, -1, idx)
    return dist.float(), idx


def check_knn_2d():
    q, s = torch.randn(200, 3), torch.randn(300, 3)
    for k in (1, 5, 40):
        d2, idx = tc_knn(q, s, k)
        rd2, ridx = brute_knn(q, s, k)
        assert torch.equal(idx, ridx), f"knn indices mismatch (k={k})"
        assert torch.allclose(d2, rd2, atol=1e-5), f"knn dists mismatch (k={k})"
        assert torch.all(d2[:, 1:] >= d2[:, :-1]), "rows not sorted"
    print("ok: tc_knn 2D matches brute force, rows sorted")


def check_knn_3d_local_indices():
    B, N, M, k = 3, 50, 80, 7
    q, s = torch.randn(B, N, 3), torch.randn(B, M, 3)
    d2, idx = tc_knn(q, s, k)
    assert d2.shape == (B, N, k) and idx.shape == (B, N, k)
    for b in range(B):
        rd2, ridx = brute_knn(q[b], s[b], k)
        assert torch.equal(idx[b], ridx), f"batched knn local indices wrong (b={b})"
        assert torch.allclose(d2[b], rd2, atol=1e-5)
    assert idx.min() >= 0 and idx.max() < M, "indices not local to batch element"
    print("ok: tc_knn 3D batched, indices local per element")


def check_knn_short_support():
    q, s = torch.randn(10, 3), torch.randn(4, 3)
    k = 9
    d2, idx = tc_knn(q, s, k)
    assert torch.isinf(d2[:, 4:]).all(), "empty slots must be inf"
    rd2, ridx = brute_knn(q, s, k)
    assert torch.equal(idx[:, :4], ridx), "filled slots wrong with short support"
    print("ok: tc_knn pads with inf when support < k")


def check_knn_3d_short_support():
    q, s = torch.randn(2, 6, 3), torch.randn(2, 4, 3)
    d2, idx = tc_knn(q, s, 5)
    assert idx.min() >= 0, "empty slots must not go negative in batched mode"
    assert torch.isinf(d2[..., 4:]).all(), "slots beyond support count must be inf"
    for b in range(2):
        rd2, ridx = brute_knn(q[b], s[b], 5)
        assert torch.equal(idx[b][..., :4], ridx), "filled slots wrong"
    print("ok: tc_knn 3D short support, empty slots inf + non-negative")


def check_pack_mode_list_lengths():
    q, s = torch.randn(12, 3), torch.randn(20, 3)
    a = radius_search_pack_mode(q, s, [5, 7], [11, 9], 0.8, 4)
    b = radius_search_pack_mode(q, s, torch.tensor([5, 7]), torch.tensor([11, 9]), 0.8, 4)
    assert torch.equal(a, b), "list lengths must behave like tensor lengths"
    print("ok: radius_search_pack_mode accepts list lengths")


def check_radius_count():
    q, s, r = torch.randn(150, 3), torch.randn(250, 3), 0.5
    counts = tc_radius_count(q, s, r)
    ref = (torch.cdist(q.double(), s.double()) < r).sum(-1)
    assert torch.equal(counts.long(), ref.long()), "radius counts mismatch (strict d < r)"
    # dense cluster forces counts past the initial 64 cap
    dense = torch.randn(300, 3) * 0.01
    counts = tc_radius_count(dense, dense, 1.0)
    assert counts.min() == 300, "cap-doubling failed on dense cluster"
    print("ok: tc_radius_count strict + cap-doubling")


def check_pack_mode():
    torch.manual_seed(1)
    q_lengths = torch.tensor([40, 3, 60])
    s_lengths = torch.tensor([70, 5, 90])
    q = torch.randn(int(q_lengths.sum()), 3)
    s = torch.randn(int(s_lengths.sum()), 3)
    radius, limit = 0.8, 12
    N = s.shape[0]

    for shadow in (False, True):
        inds = radius_search_pack_mode(q, s, q_lengths, s_lengths, radius, limit, shadow=shadow)
        assert inds.shape == (q.shape[0], limit)
        q_off = torch.cumsum(q_lengths, 0) - q_lengths
        s_off = torch.cumsum(s_lengths, 0) - s_lengths
        for b in range(len(q_lengths)):
            qb = q[q_off[b]:q_off[b] + q_lengths[b]]
            sb = s[s_off[b]:s_off[b] + s_lengths[b]]
            rd2, ridx = brute_knn(qb, sb, limit)
            ref = ridx + s_off[b]
            if shadow:
                ref = torch.where(rd2 > radius ** 2, torch.full_like(ref, N), ref)
            got = inds[q_off[b]:q_off[b] + q_lengths[b]]
            assert torch.equal(got[:, :ref.shape[1]], ref), f"pack mode mismatch (b={b}, shadow={shadow})"
            assert (got[:, ref.shape[1]:] == N).all(), "missing-neighbor slots must be shadow"
    inds, dists = radius_search_pack_mode(q, s, q_lengths, s_lengths, radius, limit, return_dist=True)
    assert dists.shape == inds.shape and (dists[inds < N] >= 0).all()
    print("ok: radius_search_pack_mode matches per-element brute force, global + shadow indices")


def check_determinism_and_ties():
    base = torch.randn(20, 3)
    s = torch.cat([base, base], 0)  # duplicated points -> every distance tied
    q = torch.randn(30, 3)
    d2a, idxa = tc_knn(q, s, 6)
    d2b, idxb = tc_knn(q, s, 6)
    assert torch.equal(idxa, idxb) and torch.equal(d2a, d2b), "not deterministic"
    ties = d2a[:, 1:] == d2a[:, :-1]
    assert (idxa[:, 1:][ties] > idxa[:, :-1][ties]).all(), "ties not broken by ascending index"
    print("ok: deterministic, ties break by support index")


if __name__ == "__main__":
    check_knn_2d()
    check_knn_3d_local_indices()
    check_knn_short_support()
    check_knn_3d_short_support()
    check_radius_count()
    check_pack_mode()
    check_pack_mode_list_lengths()
    check_determinism_and_ties()
    print("\nall neighbor-search parity tests passed")

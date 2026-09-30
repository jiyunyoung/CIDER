#!/usr/bin/env python3
"""
Construct LDPC parity-check matrix using pure PEG (Progressive Edge Growth).

PEG algorithm (Hu, Eleftheriou, Arnold 2005):
  For each variable node v, for each edge e of v:
    - Find the check node c that maximizes the local girth
    - Among ties, pick the one with lowest current degree
    - Add edge (v, c)

No QC lifting, no protograph — pure greedy edge-by-edge construction.

Used for the PEG-LDPC code of the paper (Table 2c / tab:app_peg_ldpc, and the
PEG-LDPC row of tab:response_transfer). The seed drives BOTH the tie-breaking
among equally good checks (so the topology itself is a seeded draw) and the
nonzero GF(q) edge coefficients, from one RNG stream. With --seed 42 and
(q, L, M) = (64, 12, 8) this reproduces ~/data/demix/tiny_LDPC_PEG/H_matrix.pt
exactly: a (2,3)-regular code with Tanner girth 8 whose check graph is the
Möbius ladder on 8 vertices (not the 3-cube of tiny_LDPC).

Usage:
    python construct_H_peg.py --q 64 --L 12 --M 8 --d_v 2 --d_c 3 --output H_peg.pt
    # full PEG dataset (H + train/val/test): data/gen_data/ldpc_tiny_peg.sh
"""

import argparse
import os
import sys
import numpy as np
import torch
import galois
from collections import deque

script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

from ldpc_codes_torch import make_systematic_from_H


def _bfs_from_variable(v_start, var_to_chk, chk_to_var, M, max_depth=50):
    """
    BFS from variable node v_start. Returns distance to each check node.
    Distance -1 means unreachable within max_depth.
    """
    dist_to_check = [-1] * M
    visited_v = {v_start}
    visited_c = set()

    # Queue: (node_type, node_idx, depth)
    queue = deque()
    for c in var_to_chk[v_start]:
        queue.append(('c', c, 1))
        visited_c.add(c)
        dist_to_check[c] = 1

    while queue:
        ntype, nidx, depth = queue.popleft()
        if depth >= max_depth:
            continue

        if ntype == 'c':
            for v in chk_to_var[nidx]:
                if v not in visited_v:
                    visited_v.add(v)
                    queue.append(('v', v, depth + 1))
        else:  # variable
            for c in var_to_chk[nidx]:
                if c not in visited_c:
                    visited_c.add(c)
                    dist_to_check[c] = depth + 1
                    queue.append(('c', c, depth + 1))

    return dist_to_check


def pure_peg(q, L, M, d_v, d_c, seed=42):
    """
    Pure PEG construction for q-ary LDPC codes.

    Algorithm:
      For v = 0, 1, ..., L-1:
        For edge j = 0, 1, ..., d_v-1:
          1. BFS from v to find distance to all check nodes
          2. Among check nodes with degree < d_c and not already connected to v:
             - Pick the one with maximum BFS distance (maximizes local girth)
             - Break ties by picking the one with minimum current degree
          3. Add edge (v, c)

    Args:
        q: Field size GF(q)
        L: Codeword length
        M: Number of parity checks
        d_v: Variable node degree
        d_c: Check node degree
        seed: Random seed

    Returns:
        H (galois array), var_to_chk (list of lists), chk_to_var (list of lists)
    """
    assert L * d_v == M * d_c, f"Degree constraint: L*d_v={L*d_v} != M*d_c={M*d_c}"

    np.random.seed(seed)
    GF = galois.GF(q)

    var_to_chk = [[] for _ in range(L)]
    chk_to_var = [[] for _ in range(M)]

    for v in range(L):
        for edge_idx in range(d_v):
            if edge_idx == 0 and len(var_to_chk[v]) == 0:
                # First edge of first few variables: no graph yet
                # Pick check with minimum degree
                candidates = [c for c in range(M) if len(chk_to_var[c]) < d_c]
                if not candidates:
                    raise RuntimeError(f"No available check node for v={v}")

                # Among candidates, pick minimum degree, break ties randomly
                min_deg = min(len(chk_to_var[c]) for c in candidates)
                best = [c for c in candidates if len(chk_to_var[c]) == min_deg]
                chosen = best[np.random.randint(len(best))]
            else:
                # BFS from v to find distances to all checks
                dist = _bfs_from_variable(v, var_to_chk, chk_to_var, M)

                # Filter: not already connected to v, degree < d_c
                candidates = []
                for c in range(M):
                    if c in var_to_chk[v]:
                        continue
                    if len(chk_to_var[c]) >= d_c:
                        continue
                    candidates.append(c)

                if not candidates:
                    raise RuntimeError(f"No available check node for v={v}, edge={edge_idx}")

                # Among candidates: maximize distance (maximize girth)
                # dist=-1 means unreachable = effectively infinite distance (best)
                max_dist = max(dist[c] for c in candidates)
                unreachable = [c for c in candidates if dist[c] == -1]

                if unreachable:
                    # Unreachable checks are best — no cycle created
                    # Break ties by minimum degree
                    min_deg = min(len(chk_to_var[c]) for c in unreachable)
                    best = [c for c in unreachable if len(chk_to_var[c]) == min_deg]
                else:
                    # All reachable — pick maximum distance
                    best_dist = max(dist[c] for c in candidates)
                    far = [c for c in candidates if dist[c] == best_dist]
                    # Break ties by minimum degree
                    min_deg = min(len(chk_to_var[c]) for c in far)
                    best = [c for c in far if len(chk_to_var[c]) == min_deg]

                chosen = best[np.random.randint(len(best))]

            var_to_chk[v].append(chosen)
            chk_to_var[chosen].append(v)

    # Build H with random non-zero GF(q) coefficients
    H = GF.Zeros((M, L))
    for c in range(M):
        for v in chk_to_var[c]:
            H[c, v] = GF(np.random.randint(1, q))

    return H, var_to_chk, chk_to_var


def compute_girth(var_to_chk, chk_to_var, L):
    """Compute girth by checking for short cycles."""
    # Check 4-cycles: two variables sharing two checks
    for v1 in range(L):
        for v2 in range(v1 + 1, L):
            shared = len(set(var_to_chk[v1]) & set(var_to_chk[v2]))
            if shared >= 2:
                return 4

    # Check 6-cycles
    M = len(chk_to_var)
    for c1 in range(M):
        for c2 in range(c1 + 1, M):
            shared_vars = set(chk_to_var[c1]) & set(chk_to_var[c2])
            if len(shared_vars) < 1:
                continue
            for v in shared_vars:
                other_c1_vars = [vv for vv in chk_to_var[c1] if vv != v]
                other_c2_vars = [vv for vv in chk_to_var[c2] if vv != v]
                for v1 in other_c1_vars:
                    for v2 in other_c2_vars:
                        if v1 != v2:
                            if set(var_to_chk[v1]) & set(var_to_chk[v2]) - {c1, c2}:
                                return 6

    # Check 8-cycles (BFS-based)
    for v_start in range(L):
        queue = deque()
        for c in var_to_chk[v_start]:
            queue.append(('c', c, 1, -1))
        visited = {}
        while queue:
            ntype, nidx, depth, parent_id = queue.popleft()
            if depth > 8:
                continue
            key = (ntype, nidx)
            if key in visited:
                continue
            visited[key] = depth
            if ntype == 'c':
                for v in chk_to_var[nidx]:
                    if v == v_start and depth >= 3:
                        return depth + 1
                    vkey = ('v', v)
                    if vkey not in visited:
                        queue.append(('v', v, depth + 1, nidx))
            else:
                for c in var_to_chk[nidx]:
                    ckey = ('c', c)
                    if ckey not in visited:
                        queue.append(('c', c, depth + 1, nidx))

    return '>8'


def construct_H_peg(q, L, M, d_v, d_c, seed=42):
    """Construct PEG LDPC H matrix with all encoding components."""
    print(f"Constructing pure PEG LDPC code:")
    print(f"  q={q}, L={L}, M={M}, d_v={d_v}, d_c={d_c}, k={L-M}")

    H, var_to_chk, chk_to_var = pure_peg(q, L, M, d_v, d_c, seed)

    girth = compute_girth(var_to_chk, chk_to_var, L)
    print(f"  H shape: {H.shape}")
    print(f"  Girth: {girth}")

    # Connectivity check
    visited_v = set()
    visited_c = set()
    stack = [('v', 0)]
    visited_v.add(0)
    while stack:
        ntype, nidx = stack.pop()
        if ntype == 'v':
            for c in var_to_chk[nidx]:
                if c not in visited_c:
                    visited_c.add(c)
                    stack.append(('c', c))
        else:
            for v in chk_to_var[nidx]:
                if v not in visited_v:
                    visited_v.add(v)
                    stack.append(('v', v))

    connected = len(visited_v) == L and len(visited_c) == M
    print(f"  Connected: {connected} ({len(visited_v)}/{L} VNs, {len(visited_c)}/{M} CNs)")

    if not connected:
        raise RuntimeError("PEG produced disconnected graph!")

    # Convert to systematic form
    H1, H2, Pi, GF = make_systematic_from_H(H)
    print(f"  Systematic: H1={H1.shape}, H2={H2.shape}")

    # Compute H2 inverse
    I = GF.Identity(M)
    aug = GF.Zeros((M, 2 * M))
    aug[:, :M] = H2
    aug[:, M:] = I
    aug_rref = aug.row_reduce()
    H2_inv = aug_rref[:, M:]

    check = H2 @ H2_inv
    assert np.array_equal(np.array(check), np.array(I)), "H2_inv computation failed"
    print(f"  H2_inv verified")

    # Convert adjacency to dict
    v2c_dict = {v: var_to_chk[v] for v in range(L)}
    c2v_dict = {c: chk_to_var[c] for c in range(M)}

    result = {
        'q': q,
        'L': L,
        'M': M,
        'd_v': d_v,
        'd_c': d_c,
        'k': L - M,
        'Z': 1,
        'seed': seed,
        'code_type': 'pure_PEG',

        'H_matrix': torch.tensor(np.array(H, dtype=np.int64), dtype=torch.long),
        'H1': torch.tensor(np.array(H1, dtype=np.int64), dtype=torch.long),
        'H2': torch.tensor(np.array(H2, dtype=np.int64), dtype=torch.long),
        'H2_inv': torch.tensor(np.array(H2_inv, dtype=np.int64), dtype=torch.long),
        'Pi': torch.tensor(np.array(Pi, dtype=np.int64), dtype=torch.long),

        'var_to_chk': v2c_dict,
        'chk_to_var': c2v_dict,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Construct pure PEG LDPC H matrix")
    parser.add_argument('--q', type=int, required=True)
    parser.add_argument('--L', type=int, required=True)
    parser.add_argument('--M', type=int, required=True)
    parser.add_argument('--d_v', type=int, required=True)
    parser.add_argument('--d_c', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()

    result = construct_H_peg(args.q, args.L, args.M, args.d_v, args.d_c, args.seed)

    if args.show:
        H = result['H_matrix'].numpy()
        M, L = H.shape
        print(f"\nH matrix ({M}x{L}):")
        for i in range(M):
            row = ['  X' if H[i, j] > 0 else '  .' for j in range(L)]
            print(f"  c{i:2} |{''.join(row)}")

        print(f"\nVN neighbors:")
        for v in range(L):
            print(f"  v{v}: checks {result['var_to_chk'][v]}")

    torch.save(result, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

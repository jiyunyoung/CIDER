#!/usr/bin/env python3
"""
Construct tree code parity-check matrix over GF(q).

Tree code structure (adapted from Amalladinne et al. 2022):
- L blocks, split into W information blocks and P parity blocks
- Blocks alternate: info, info, parity, info, info, parity, ...
- Each parity block is a random GF(q) linear combination of preceding info blocks
- Factor graph: bipartite between variable nodes (all blocks) and factor nodes (parity checks)

This uses GF(q) arithmetic (same as LDPC) instead of Z/2^v (ring),
so it is directly compatible with CIDER's GF(q) permutation-based neural MP.

The key difference from LDPC is the graph topology:
- LDPC: regular bipartite graph (d_v=2, d_c=3), all positions are equal
- Tree code: heterogeneous — info blocks vs parity blocks, causal dependency structure

Usage:
    python construct_tree_code.py --q 64 --L 12 --pattern iip --seed 42 --output tree_code.pt
"""

import argparse
import os
import sys
import numpy as np
import torch
import galois

script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

from ldpc_codes_torch import make_systematic_from_H, _gcd


def tiny_tree_code_H(q, seed=42):
    """
    Construct tree code H matrix for L=12, k=4, M=8 (rate 1/3).

    Triadic parity connectivity (d_c=3 for all checks) with hierarchical
    cascading structure where later parity blocks depend on earlier ones.

    Layout: [I, I, P, I, P, P, I, P, P, P, P, P]
    Info positions:   s1=0, s2=1, s4=3, s7=6
    Parity positions: s3=2, s5=4, s6=5, s8=7, s9=8, s10=9, s11=10, s12=11

    Parity equations (each check has degree 3):
      a3:  (s1, s2, s3)    — binds first two info blocks
      a5:  (s2, s4, s5)    — binds info across groups
      a6:  (s1, s4, s6)    — binds info across groups
      a8:  (s4, s7, s8)    — binds later info blocks
      a9:  (s2, s7, s9)    — binds info across groups
      a10: (s1, s7, s10)   — binds info across groups
      a11: (s3, s8, s11)   — cascading: binds parity to parity
      a12: (s5, s9, s12)   — cascading: binds parity to parity
    """
    GF = galois.GF(q)
    np.random.seed(seed)

    L = 12
    M = 8

    # Block types: I=info (first 4), P=parity (last 8)
    block_types = ['i', 'i', 'i', 'i', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p']
    info_positions = [0, 1, 2, 3]
    parity_positions = [4, 5, 6, 7, 8, 9, 10, 11]

    # Binary-tree structured triadic checks (d_c=3, girth=8, causal)
    #
    # Layer 1: bind info pairs
    #   a5:  (I0, I1) → P4       a6: (I2, I3) → P5
    #   a7:  (I0, I2) → P6       a8: (I1, I3) → P7
    # Layer 2: bind layer-1 parity pairs (cascade)
    #   a9:  (P4, P5) → P8      a10: (P6, P7) → P9
    # Layer 3: bind layer-2 (cascade)
    #   a11: (P8, P9) → P10
    # Layer 4: global binding back to root (cascade)
    #   a12: (I0, P10) → P11
    checks = [
        (0, 1, 4),    # a5:  I0, I1  → P4
        (2, 3, 5),    # a6:  I2, I3  → P5
        (0, 2, 6),    # a7:  I0, I2  → P6
        (1, 3, 7),    # a8:  I1, I3  → P7
        (4, 5, 8),    # a9:  P4, P5  → P8   (layer-2 cascade)
        (6, 7, 9),    # a10: P6, P7  → P9   (layer-2 cascade)
        (8, 9, 10),   # a11: P8, P9  → P10  (layer-3 cascade)
        (0, 10, 11),  # a12: I0, P10 → P11  (global binding)
    ]

    # Build H matrix
    H = GF.Zeros((M, L))
    var_to_chk = {v: [] for v in range(L)}
    chk_to_var = {c: [] for c in range(M)}

    for check_idx, (dep1, dep2, p_pos) in enumerate(checks):
        # Random non-zero GF(q) coefficients for dependencies
        for dep in [dep1, dep2]:
            coef = GF(np.random.randint(1, q))
            H[check_idx, dep] = coef
            var_to_chk[dep].append(check_idx)
            chk_to_var[check_idx].append(dep)

        # Parity block itself: coefficient 1 (in GF(2^m), -1 = 1)
        H[check_idx, p_pos] = GF(1)
        var_to_chk[p_pos].append(check_idx)
        chk_to_var[check_idx].append(p_pos)

    return H, var_to_chk, chk_to_var, block_types, info_positions, parity_positions


def build_tree_code_H(q, L, seed=42):
    """
    Build tree code parity-check matrix over GF(q).

    Uses predefined graph structures for known sizes, similar to
    how construct_H.py uses predefined LDPC structures.

    Args:
        q: Field size GF(q)
        L: Codeword length
        seed: Random seed

    Returns:
        H_matrix, var_to_chk, chk_to_var, block_types, info_positions, parity_positions
    """
    if L == 12:
        return tiny_tree_code_H(q, seed)
    else:
        raise ValueError(f"No predefined tree code for L={L}. Implement or use L=12.")


def construct_tree_code(q, L, seed=42):
    """
    Construct tree code with all encoding components (same output format as construct_H).

    Args:
        q: Alphabet size GF(q)
        L: Codeword length
        seed: Random seed

    Returns:
        dict with same keys as construct_H output (H_matrix, H1, H2, H2_inv, Pi, etc.)
    """
    H_matrix, var_to_chk, chk_to_var, block_types, info_positions, parity_positions = \
        build_tree_code_H(q, L, seed)

    M = len(parity_positions)
    k = len(info_positions)

    print(f"\nConstructing tree code:")
    print(f"  q={q}, L={L}, M={M}, k={k}")
    print(f"  H shape: {H_matrix.shape}")

    # Compute d_v (variable degree) and d_c (check degree) stats
    d_v_list = [len(var_to_chk[v]) for v in range(L)]
    d_c_list = [len(chk_to_var[c]) for c in range(M)]
    print(f"  d_v range: {min(d_v_list)}-{max(d_v_list)} (avg {np.mean(d_v_list):.1f})")
    print(f"  d_c range: {min(d_c_list)}-{max(d_c_list)} (avg {np.mean(d_c_list):.1f})")

    # Convert to systematic form for encoding (same as LDPC)
    GF = galois.GF(q)

    # Make systematic: find permutation Pi such that H[:, Pi] = [H1 | H2]
    # where H2 is invertible
    H1, H2, Pi, GF = make_systematic_from_H(H_matrix)
    print(f"  Systematic: H1={H1.shape}, H2={H2.shape}")

    # Compute H2 inverse
    I = GF.Identity(M)
    aug = GF.Zeros((M, 2 * M))
    aug[:, :M] = H2
    aug[:, M:] = I
    aug_rref = aug.row_reduce()
    H2_inv = aug_rref[:, M:]

    # Verify
    check = H2 @ H2_inv
    assert np.array_equal(np.array(check), np.array(I)), "H2_inv computation failed"
    print(f"  H2_inv verified")

    Z = _gcd(L, M)

    result = {
        # Code parameters
        'q': q,
        'L': L,
        'M': M,
        'd_v': 0,  # irregular, use 0 as placeholder
        'd_c': 0,
        'k': k,
        'Z': Z,
        'seed': seed,
        'code_type': 'tree_code',

        # H matrix
        'H_matrix': torch.tensor(np.array(H_matrix, dtype=np.int64), dtype=torch.long),

        # Encoding matrices
        'H1': torch.tensor(np.array(H1, dtype=np.int64), dtype=torch.long),
        'H2': torch.tensor(np.array(H2, dtype=np.int64), dtype=torch.long),
        'H2_inv': torch.tensor(np.array(H2_inv, dtype=np.int64), dtype=torch.long),
        'Pi': torch.tensor(np.array(Pi, dtype=np.int64), dtype=torch.long),

        # Adjacency
        'var_to_chk': var_to_chk,
        'chk_to_var': chk_to_var,

        # Tree code specific
        'block_types': block_types,
        'info_positions': info_positions,
        'parity_positions': parity_positions,
    }

    return result


def show_tree_code(result):
    """Display tree code structure."""
    H = result['H_matrix'].numpy()
    L, M = result['L'], result['M']
    block_types = result['block_types']

    print(f"\nTree code H matrix ({M}x{L}):")

    # Header with block type markers
    print("        ", end="")
    for j in range(L):
        t = block_types[j].upper()
        print(f" {t}{j:1}", end="")
    print()
    print("        " + "-" * (L * 3))

    for i in range(M):
        p_pos = result['parity_positions'][i]
        print(f"  p{p_pos:2} |", end="")
        for j in range(L):
            if H[i, j] > 0:
                print("  X", end="")
            else:
                print("  .", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Construct tree code over GF(q)")
    parser.add_argument('--q', type=int, default=64, help='Alphabet size GF(q)')
    parser.add_argument('--L', type=int, default=12, help='Codeword length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--show', action='store_true', help='Display structure')

    args = parser.parse_args()

    result = construct_tree_code(args.q, args.L, args.seed)

    if args.show:
        show_tree_code(result)

    torch.save(result, args.output)
    print(f"\nSaved to {args.output}")
    print(f"  Parameters: q={args.q}, L={args.L}, k={result['k']}, M={result['M']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Construct LDPC parity-check matrix H using protograph-based QC construction.

This script creates H matrix with uniform syndrome mixing and saves all
encoding matrices needed for data generation.

Usage:
    python construct_H.py --q 64 --L 12 --M 8 --d_v 2 --d_c 3 --output H_matrix.pt
    python construct_H.py --q 64 --L 12 --M 8 --d_v 2 --d_c 3 --output H_matrix.pt --show
"""

import argparse
import os
import sys
import numpy as np
import torch
import galois

script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

from ldpc_codes_torch import peg_ldpc_qary, make_systematic_from_H, _gcd


def tiny_ldpc_H(q, seed=42):
    """
    Construct H matrix for tiny (8, 12) LDPC code with high girth.

    This gives a (8, 12) LDPC code with:
    - L = 12 variables (codeword length)
    - M = 8 checks (parity checks)
    - d_v = 2 (column weight)
    - d_c = 3 (row weight)
    - k = 4 (info symbols)
    - High girth structure
    """
    GF = galois.GF(q)
    np.random.seed(seed)

    M = 8   # checks
    L = 12  # variables

    # Variable to check connections (each variable connects 2 checks)
    var_to_chk = {
        0: [0, 1],
        1: [0, 2],
        2: [0, 4],
        3: [1, 3],
        4: [1, 5],
        5: [2, 3],
        6: [2, 6],
        7: [3, 7],
        8: [4, 5],
        9: [4, 6],
        10: [5, 7],
        11: [6, 7],
    }

    # Build H matrix
    H = GF.Zeros((M, L))
    for v, checks in var_to_chk.items():
        for c in checks:
            # Use random non-zero GF element for edge weight
            H[c, v] = GF(np.random.randint(1, q))

    # Build chk_to_var
    chk_to_var = {c: [] for c in range(M)}
    for v, checks in var_to_chk.items():
        for c in checks:
            chk_to_var[c].append(v)

    return H, var_to_chk, chk_to_var


def large_ldpc_H(q, seed=42):
    """
    Construct H matrix for larger (24, 36) LDPC code.

    This gives a (24, 36) LDPC code with:
    - L = 36 variables (codeword length)
    - M = 24 checks (parity checks)
    - d_v = 2 (column weight)
    - d_c = 3 (row weight)
    - k = 12 (info symbols)
    - High girth structure

    Based on a specific edge assignment with good cycle properties.
    """
    GF = galois.GF(q)
    np.random.seed(seed)

    M = 24  # checks
    L = 36  # variables

    # Variable to check connections (each variable connects 2 checks)
    var_to_chk = {
        0: [0, 1],
        1: [0, 3],
        2: [0, 12],
        3: [1, 18],
        4: [1, 21],
        5: [2, 15],
        6: [2, 20],
        7: [2, 23],
        8: [3, 10],
        9: [3, 14],
        10: [4, 8],
        11: [4, 9],
        12: [4, 23],
        13: [5, 9],
        14: [5, 12],
        15: [5, 22],
        16: [6, 7],
        17: [6, 12],
        18: [6, 21],
        19: [7, 10],
        20: [7, 13],
        21: [8, 14],
        22: [8, 20],
        23: [9, 21],
        24: [10, 15],
        25: [11, 14],
        26: [11, 18],
        27: [11, 22],
        28: [13, 16],
        29: [13, 20],
        30: [15, 17],
        31: [16, 19],
        32: [16, 23],
        33: [17, 18],
        34: [17, 19],
        35: [19, 22],
    }

    # Build H matrix
    H = GF.Zeros((M, L))
    for v, checks in var_to_chk.items():
        for c in checks:
            # Use random non-zero GF element for edge weight
            H[c, v] = GF(np.random.randint(1, q))

    # Build chk_to_var
    chk_to_var = {c: [] for c in range(M)}
    for v, checks in var_to_chk.items():
        for c in checks:
            chk_to_var[c].append(v)

    return H, var_to_chk, chk_to_var


def small_ldpc_H(q, seed=42):
    """
    Construct H matrix for (12, 18) LDPC code with high girth.

    This gives a (12, 18) LDPC code with:
    - L = 18 variables (codeword length)
    - M = 12 checks (parity checks)
    - d_v = 2 (column weight)
    - d_c = 3 (row weight)
    - k = 6 (info symbols)
    - High girth structure
    """
    GF = galois.GF(q)
    np.random.seed(seed)

    M = 12  # checks
    L = 18  # variables

    # Variable to check connections (each variable connects 2 checks)
    var_to_chk = {
        0: [0, 1],
        1: [0, 2],
        2: [0, 5],
        3: [1, 4],
        4: [1, 10],
        5: [2, 6],
        6: [2, 8],
        7: [3, 8],
        8: [3, 4],
        9: [3, 9],
        10: [4, 6],
        11: [5, 7],
        12: [5, 9],
        13: [6, 11],
        14: [7, 11],
        15: [7, 8],
        16: [9, 10],
        17: [10, 11],
    }

    # Build H matrix
    H = GF.Zeros((M, L))
    for v, checks in var_to_chk.items():
        for c in checks:
            # Use random non-zero GF element for edge weight
            H[c, v] = GF(np.random.randint(1, q))

    # Build chk_to_var
    chk_to_var = {c: [] for c in range(M)}
    for v, checks in var_to_chk.items():
        for c in checks:
            chk_to_var[c].append(v)

    return H, var_to_chk, chk_to_var


def moderate_ldpc_H(q, seed=42):
    """
    Construct H matrix for (16, 24) LDPC code using Möbius ladder M16 structure.

    This gives a (16, 24) LDPC code with:
    - L = 24 variables (codeword length)
    - M = 16 checks (parity checks)
    - d_v = 2 (column weight)
    - d_c = 3 (row weight)
    - k = 8 (info symbols)
    - Girth = 8 (no 4-cycles or 6-cycles)

    Structure:
    - Cycle edges (v0-v15): vi connects to (c_i, c_{i+1 mod 16})
    - Opposite chords (v16-v23): v_{16+i} connects to (c_i, c_{i+8})
    """
    GF = galois.GF(q)
    np.random.seed(seed)

    M = 16  # checks
    L = 24  # variables

    # Row-wise adjacency from Möbius ladder M16
    # Each check (row) has exactly 3 variables (columns)
    chk_to_var = {
        0: [15, 0, 16],
        1: [0, 1, 17],
        2: [1, 2, 18],
        3: [2, 3, 19],
        4: [3, 4, 20],
        5: [4, 5, 21],
        6: [5, 6, 22],
        7: [6, 7, 23],
        8: [7, 8, 16],
        9: [8, 9, 17],
        10: [9, 10, 18],
        11: [10, 11, 19],
        12: [11, 12, 20],
        13: [12, 13, 21],
        14: [13, 14, 22],
        15: [14, 15, 23],
    }

    # Build var_to_chk from chk_to_var
    var_to_chk = {v: [] for v in range(L)}
    for c, vars in chk_to_var.items():
        for v in vars:
            var_to_chk[v].append(c)

    # Build H matrix
    H = GF.Zeros((M, L))
    for c, vars in chk_to_var.items():
        for v in vars:
            # Use random non-zero GF element for edge weight
            H[c, v] = GF(np.random.randint(1, q))

    return H, var_to_chk, chk_to_var


def large_mobius_ldpc_H(q, seed=42):
    """
    Construct H matrix for (32, 48) LDPC code using Möbius ladder M32 structure.

    This gives a (32, 48) LDPC code with:
    - L = 48 variables (codeword length)
    - M = 32 checks (parity checks)
    - d_v = 2 (column weight)
    - d_c = 3 (row weight)
    - k = 16 (info symbols)
    - No 4-cycles (high girth)

    Structure:
    - Cycle edges (v0-v31): vi connects to (c_i, c_{i+1 mod 32})
    - Opposite chords (v32-v47): v_{32+i} connects to (c_i, c_{i+16})
    """
    GF = galois.GF(q)
    np.random.seed(seed)

    M = 32  # checks
    L = 48  # variables

    # Row-wise adjacency from Möbius ladder M32
    # Each check (row) has exactly 3 variables (columns)
    chk_to_var = {
        0: [31, 0, 32],
        1: [0, 1, 33],
        2: [1, 2, 34],
        3: [2, 3, 35],
        4: [3, 4, 36],
        5: [4, 5, 37],
        6: [5, 6, 38],
        7: [6, 7, 39],
        8: [7, 8, 40],
        9: [8, 9, 41],
        10: [9, 10, 42],
        11: [10, 11, 43],
        12: [11, 12, 44],
        13: [12, 13, 45],
        14: [13, 14, 46],
        15: [14, 15, 47],
        16: [15, 16, 32],
        17: [16, 17, 33],
        18: [17, 18, 34],
        19: [18, 19, 35],
        20: [19, 20, 36],
        21: [20, 21, 37],
        22: [21, 22, 38],
        23: [22, 23, 39],
        24: [23, 24, 40],
        25: [24, 25, 41],
        26: [25, 26, 42],
        27: [26, 27, 43],
        28: [27, 28, 44],
        29: [28, 29, 45],
        30: [29, 30, 46],
        31: [30, 31, 47],
    }

    # Build var_to_chk from chk_to_var
    var_to_chk = {v: [] for v in range(L)}
    for c, vars in chk_to_var.items():
        for v in vars:
            var_to_chk[v].append(c)

    # Build H matrix
    H = GF.Zeros((M, L))
    for c, vars in chk_to_var.items():
        for v in vars:
            # Use random non-zero GF element for edge weight
            H[c, v] = GF(np.random.randint(1, q))

    return H, var_to_chk, chk_to_var


def construct_H(q, L, M, d_v, d_c, seed=42):
    """
    Construct LDPC parity-check matrix H with all encoding components.

    Args:
        q: Alphabet size (GF(q))
        L: Codeword length
        M: Number of parity checks
        d_v: Variable node degree
        d_c: Check node degree
        seed: Random seed

    Returns:
        dict containing H matrix and all encoding components
    """
    # Verify degree constraint
    assert L * d_v == M * d_c, f"Degree constraint failed: L*d_v={L*d_v} != M*d_c={M*d_c}"

    # Compute derived parameters
    Z = _gcd(L, M)
    n_b = L // Z
    m_b = M // Z
    k = L - M  # Information symbols

    print(f"Constructing LDPC code:")
    print(f"  q={q}, L={L}, M={M}, d_v={d_v}, d_c={d_c}")
    print(f"  Z={Z}, protograph={m_b}x{n_b}, k={k}")

    # Use specialized constructions for known good codes
    if L == 48 and M == 32 and d_v == 2 and d_c == 3:
        print(f"  Using large LDPC construction (Möbius ladder M32, high girth)")
        H_matrix, var_to_chk, chk_to_var = large_mobius_ldpc_H(q, seed)
    elif L == 36 and M == 24 and d_v == 2 and d_c == 3:
        print(f"  Using legacy large LDPC construction (high girth)")
        H_matrix, var_to_chk, chk_to_var = large_ldpc_H(q, seed)
    elif L == 24 and M == 16 and d_v == 2 and d_c == 3:
        print(f"  Using moderate LDPC construction (Möbius ladder M16, girth=8)")
        H_matrix, var_to_chk, chk_to_var = moderate_ldpc_H(q, seed)
    elif L == 18 and M == 12 and d_v == 2 and d_c == 3:
        print(f"  Using small LDPC construction (high girth)")
        H_matrix, var_to_chk, chk_to_var = small_ldpc_H(q, seed)
    elif L == 12 and M == 8 and d_v == 2 and d_c == 3:
        print(f"  Using tiny LDPC construction (high girth)")
        H_matrix, var_to_chk, chk_to_var = tiny_ldpc_H(q, seed)
    else:
        # Fallback to protograph + QC-PEG
        # For small fields (especially GF(2)), PEG may produce rank-deficient H.
        # Retry with different seeds until full rank is achieved.
        max_attempts = 50
        for attempt in range(max_attempts):
            try_seed = seed + attempt
            H_matrix, var_to_chk, chk_to_var = peg_ldpc_qary(q, L, M, d_v, d_c, try_seed)
            # Quick rank check for GF(2)
            if q == 2:
                import galois as _gal
                _GF2 = _gal.GF(2)
                _rref = _GF2(np.array(H_matrix, dtype=int)).row_reduce()
                _rank = sum(1 for row in _rref if any(row))
                if _rank < M:
                    if attempt < max_attempts - 1:
                        print(f"  Seed {try_seed}: rank {_rank}/{M}, retrying...")
                        continue
                    else:
                        print(f"  WARNING: Could not find full-rank H after {max_attempts} attempts (rank={_rank}/{M})")
                else:
                    print(f"  Seed {try_seed}: full rank {_rank}/{M}")
                    break
            else:
                break
    print(f"  H shape: {H_matrix.shape}")

    # Convert to systematic form for encoding
    H1, H2, Pi, GF = make_systematic_from_H(H_matrix)
    print(f"  Systematic: H1={H1.shape}, H2={H2.shape}")

    # Compute H2 inverse
    I = GF.Identity(M)
    aug = GF.Zeros((M, 2*M))
    aug[:, :M] = H2
    aug[:, M:] = I
    aug_rref = aug.row_reduce()
    H2_inv = aug_rref[:, M:]

    # Verify H2 @ H2_inv = I
    check = H2 @ H2_inv
    assert np.array_equal(np.array(check), np.array(I)), "H2_inv computation failed"
    print(f"  H2_inv verified")

    # Package everything
    result = {
        # Code parameters
        'q': q,
        'L': L,
        'M': M,
        'd_v': d_v,
        'd_c': d_c,
        'k': k,  # Information symbols
        'Z': Z,  # Lifting factor
        'seed': seed,

        # H matrix (as torch tensor)
        'H_matrix': torch.tensor(np.array(H_matrix, dtype=np.int64), dtype=torch.long),

        # Encoding matrices (as torch tensors)
        'H1': torch.tensor(np.array(H1, dtype=np.int64), dtype=torch.long),
        'H2': torch.tensor(np.array(H2, dtype=np.int64), dtype=torch.long),
        'H2_inv': torch.tensor(np.array(H2_inv, dtype=np.int64), dtype=torch.long),
        'Pi': torch.tensor(np.array(Pi, dtype=np.int64), dtype=torch.long),

        # Adjacency (for analysis)
        'var_to_chk': var_to_chk,
        'chk_to_var': chk_to_var,
    }

    return result


def show_H(result):
    """Display H matrix structure."""
    H = result['H_matrix'].numpy()
    L, M = result['L'], result['M']
    Z = result['Z']

    print(f"\nH matrix ({M}x{L}):")

    # Header with block separators
    n_b = L // Z
    print("        ", end="")
    for j in range(L):
        if j > 0 and j % Z == 0:
            print("|", end="")
        print(f"{j:3}", end="")
    print()

    print("        " + "-" * (L * 3 + n_b - 1))

    m_b = M // Z
    for i in range(M):
        if i > 0 and i % Z == 0:
            print("        " + "-" * (L * 3 + n_b - 1))
        print(f"   c{i:2} |", end="")
        for j in range(L):
            if j > 0 and j % Z == 0:
                print("|", end="")
            print("  X" if H[i, j] > 0 else "  .", end="")
        print()

    # Show which blocks each check connects to
    print(f"\nCheck → Variable blocks (Z={Z}):")
    chk_to_var = result['chk_to_var']
    for c in range(M):
        blocks = set(v // Z for v in chk_to_var[c])
        print(f"  c{c}: blocks {sorted(blocks)}")


def main():
    parser = argparse.ArgumentParser(description="Construct LDPC H matrix")

    parser.add_argument('--q', type=int, required=True, help='Alphabet size (GF(q))')
    parser.add_argument('--L', type=int, required=True, help='Codeword length')
    parser.add_argument('--M', type=int, required=True, help='Number of parity checks')
    parser.add_argument('--d_v', type=int, required=True, help='Variable node degree')
    parser.add_argument('--d_c', type=int, required=True, help='Check node degree')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--show', action='store_true', help='Display H matrix structure')

    args = parser.parse_args()

    # Construct H
    result = construct_H(args.q, args.L, args.M, args.d_v, args.d_c, args.seed)

    # Show if requested
    if args.show:
        show_H(result)

    # Save
    torch.save(result, args.output)
    print(f"\nSaved to {args.output}")
    print(f"  Contains: H_matrix, H1, H2, H2_inv, Pi, var_to_chk, chk_to_var")
    print(f"  Parameters: q={args.q}, L={args.L}, M={args.M}, k={result['k']}, Z={result['Z']}")


if __name__ == "__main__":
    main()

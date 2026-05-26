"""
Galois Field GF(q) arithmetic operations.

Provides vectorized GF(q) operations for syndrome computation in LDPC/RS codes.
"""

import torch
import math


class GFq:
    """Galois Field GF(q) arithmetic using lookup tables."""

    def __init__(self, q=16):
        """
        Initialize GF(q) with q = 2^m (e.g., q=16 means GF(2^4)).
        Uses primitive polynomial for field construction.
        """
        self.q = q
        self.m = int(math.log2(q))
        assert 2 ** self.m == q, "q must be a power of 2"

        # Primitive polynomials for common fields - MUST match galois library defaults!
        primitives = {
            4: 0b10011,        # GF(16): x^4 + x + 1
            5: 0b100101,       # GF(32): x^5 + x^2 + 1
            6: 0b1011011,      # GF(64): x^6 + x^4 + x^3 + x + 1 (galois default)
            7: 0b10000011,     # GF(128): x^7 + x + 1 (galois default)
            8: 0b100011101,    # GF(256): x^8 + x^4 + x^3 + x^2 + 1 (galois default)
            9: 0b1000010001,   # GF(512): x^9 + x^4 + 1
            10: 0b10000001001, # GF(1024): x^10 + x^3 + 1
        }
        self.primitive = primitives.get(self.m, self._find_primitive())

        # Build exp and log tables
        self.exp_table = [0] * (2 * q)
        self.log_table = [0] * q

        x = 1
        for i in range(q - 1):
            self.exp_table[i] = x
            self.log_table[x] = i
            x <<= 1
            if x >= q:
                x ^= self.primitive

        # Extend exp table for easier multiplication
        for i in range(q - 1, 2 * q):
            self.exp_table[i] = self.exp_table[i - (q - 1)]

    def _find_primitive(self):
        """Find a primitive polynomial (fallback)."""
        return 0b10011

    def add(self, a, b):
        """Addition in GF(q) is XOR."""
        return a ^ b

    def mul(self, a, b):
        """Multiplication in GF(q) using log/exp tables."""
        if a == 0 or b == 0:
            return 0
        return self.exp_table[self.log_table[a] + self.log_table[b]]

    def batch_mul(self, H_row, x_row):
        """
        Compute dot product H_row · x_row over GF(q).
        H_row: [N] tensor of field elements
        x_row: [N] tensor of field elements
        Returns: scalar in GF(q)
        """
        result = 0
        for h, x in zip(H_row.tolist(), x_row.tolist()):
            result = self.add(result, self.mul(int(h), int(x)))
        return result


# Global GF instance cache
_gf_cache = {}

def get_gf(q):
    """Get or create GF(q) instance."""
    if q not in _gf_cache:
        _gf_cache[q] = GFq(q)
    return _gf_cache[q]


# Cache for vectorized GF tables (as torch tensors)
_gf_torch_cache = {}

def get_gf_torch_tables(q, device):
    """Get exp/log tables as torch tensors for vectorized operations."""
    cache_key = (q, str(device))
    if cache_key not in _gf_torch_cache:
        gf = get_gf(q)
        exp_table = torch.tensor(gf.exp_table, dtype=torch.long, device=device)
        log_table = torch.tensor(gf.log_table, dtype=torch.long, device=device)
        _gf_torch_cache[cache_key] = (exp_table, log_table)
    return _gf_torch_cache[cache_key]


def syndrome_over_gfq_batch(H, X, q):
    """
    Vectorized batch syndrome computation for [B, K, N] input.

    Args:
        H: [M, N] parity-check matrix
        X: [B, K, N] symbols
        q: field size

    Returns:
        [B, K, M] syndrome values

    Uses fully vectorized GF(q) operations:
    - Multiplication via log/exp tables
    - Addition via XOR reduction
    """
    B, K, N = X.shape
    M = H.shape[0]
    device = X.device

    # Get GF tables as torch tensors
    exp_table, log_table = get_gf_torch_tables(q, device)

    # Replace MASK token (q) with 0 for syndrome computation
    X_clean = X.clone()
    X_clean[X_clean >= q] = 0

    # Expand H for batch multiplication: [M, N] -> [1, 1, M, N]
    H_exp = H.unsqueeze(0).unsqueeze(0)  # [1, 1, M, N]

    # Expand X for parity check: [B, K, N] -> [B, K, 1, N]
    X_exp = X_clean.unsqueeze(2)  # [B, K, 1, N]

    # GF(q) multiplication: a * b = exp[log[a] + log[b]], with 0 * x = 0
    # Create masks for zero elements
    H_zero = (H_exp == 0)
    X_zero = (X_exp == 0)
    either_zero = H_zero | X_zero  # [B, K, M, N]

    # Clamp to avoid index errors (zeros will be masked out anyway)
    H_safe = H_exp.clamp(min=1)
    X_safe = X_exp.clamp(min=1)

    # Log lookup: [B, K, M, N]
    log_H = log_table[H_safe]
    log_X = log_table[X_safe]

    # Exp of sum (modular arithmetic handled by extended exp_table)
    log_sum = log_H + log_X  # [B, K, M, N]
    products = exp_table[log_sum]  # [B, K, M, N]

    # Zero out where either operand was zero
    products = products.masked_fill(either_zero, 0)

    # GF(q) addition is XOR - reduce over N dimension
    syndromes = products[..., 0]  # [B, K, M]
    for n in range(1, N):
        syndromes = syndromes ^ products[..., n]

    return syndromes


def syndrome_over_gfq(H, x_row, q):
    """
    Compute syndrome s = H @ x over GF(q) for a single codeword.

    Args:
        H: [M, N] parity-check matrix (int tensor)
        x_row: [N] symbols in {0..q-1}, MASK tokens treated as 0
        q: field size

    Returns:
        [M] syndrome values in {0..q-1}
    """
    X = x_row.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
    syndromes = syndrome_over_gfq_batch(H, X, q)
    return syndromes[0, 0]  # [M]


def per_token_syndrome_contribution(H, X, q):
    """
    Compute per-token contribution to each syndrome equation.

    For each token n and parity check m: contribution[m] = H[m, n] * x[n]
    This tells us what each token contributes to each syndrome.

    Args:
        H: [M, N] parity-check matrix
        X: [B, K, N] symbols
        q: field size

    Returns:
        [B, K, N, M] per-token contribution to each syndrome
    """
    B, K, N = X.shape
    M = H.shape[0]
    device = X.device

    # Get GF tables as torch tensors
    exp_table, log_table = get_gf_torch_tables(q, device)

    # Replace MASK token (q) with 0
    X_clean = X.clone()
    X_clean[X_clean >= q] = 0

    # Expand for broadcasting: H[M, N] -> [1, 1, M, N], X[B, K, N] -> [B, K, 1, N]
    H_exp = H.unsqueeze(0).unsqueeze(0)  # [1, 1, M, N]
    X_exp = X_clean.unsqueeze(2)  # [B, K, 1, N]

    # GF(q) multiplication
    H_zero = (H_exp == 0)
    X_zero = (X_exp == 0)
    either_zero = H_zero | X_zero

    H_safe = H_exp.clamp(min=1)
    X_safe = X_exp.clamp(min=1)

    log_H = log_table[H_safe]
    log_X = log_table[X_safe]
    log_sum = log_H + log_X
    products = exp_table[log_sum]  # [B, K, M, N]
    products = products.masked_fill(either_zero, 0)

    # Transpose to [B, K, N, M] - per-token contribution to each syndrome
    return products.permute(0, 1, 3, 2)


def build_gf_perm_table(q):
    """
    Build GF(Q) permutation table for coefficient normalization.

    perm_table[a, s] = a^{-1} * s in GF(Q)

    This is used for normalizing/denormalizing messages by H coefficients.
    When coefficient is a, we multiply by a^{-1} to normalize.

    Args:
        q: field size (must be power of 2)

    Returns:
        [Q, Q] tensor where perm_table[a, s] = a^{-1} * s
    """
    gf = get_gf(q)

    # Build inverse table: inv[a] = a^{-1}
    # In GF(q), a^{-1} = a^{q-2} (Fermat's little theorem)
    # Or use: a^{-1} = exp[q-1 - log[a]]
    inv_table = [0] * q
    for a in range(1, q):
        # a^{-1} = exp[(q-1) - log[a]] since exp has period q-1
        inv_table[a] = gf.exp_table[(q - 1) - gf.log_table[a]]

    # Build permutation table
    perm_table = torch.zeros(q, q, dtype=torch.long)
    for a in range(q):
        for s in range(q):
            if a == 0:
                # 0^{-1} is undefined, map to 0 (won't be used)
                perm_table[a, s] = 0
            else:
                # perm[a, s] = a^{-1} * s
                a_inv = inv_table[a]
                perm_table[a, s] = gf.mul(a_inv, s)

    return perm_table


def build_pcm_mask(H, K, device):
    """
    Build PCM (parity-check matrix) mask for syndrome cross-attention.

    Tokens (k, n) can only attend to check nodes j where H[j, n] != 0.
    Mask is [K*N, K*M] shape, block-diagonal per user k.

    Args:
        H: [M, N] parity-check matrix
        K: number of users/slots
        device: torch device

    Returns:
        [K*N, K*M] boolean mask, True = allowed attention
    """
    M, N = H.shape
    # H^T mask for symbol→syndrome attention
    mask_single = (H.T != 0)  # [N, M], True where H[m,n] != 0

    # Expand block-diagonal for K users
    mask_blocks = [mask_single for _ in range(K)]
    mask_full = torch.block_diag(*mask_blocks).to(device)  # [K*N, K*M]

    return mask_full  # True = allowed, False = blocked

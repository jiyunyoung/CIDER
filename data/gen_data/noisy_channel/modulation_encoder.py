"""
Modulation Encoder: Unified encoder for SMV and MMV systems.

Maps user codewords to transmitted signal using a sensing matrix.
See docs/modulation_pipeline.md for full pipeline documentation.

Signal Model:
    x = sum_k sqrt(Psym) * A[:, idx_k]

where:
- x: Transmitted signal (n_s,) complex
- A: Sensing matrix (n_s, Q) complex
- idx_k: Symbol index for user k
- Psym: Power per symbol = (B * 10^(Eb/10)) / L

Supported sensing matrices:
- partial_dft: Partial DFT matrix (random row selection) - RECOMMENDED
- qr_gaussian: QR-orthogonalized Gaussian matrix
- hadamard: Partial Hadamard matrix (real-valued, ±1 entries)
"""

from typing import List, Dict, Any, Optional, Set
import torch
import math


def create_partial_dft_matrix(
    n_s: int,
    Q: int,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.complex64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a partial DFT sensing matrix.

    Selects n_s rows randomly from a Q x Q DFT matrix.

    Args:
        n_s: Number of measurements (rows to select)
        Q: Alphabet size (DFT matrix size)
        seed: Random seed for row selection
        dtype: Complex data type
        device: Torch device

    Returns:
        A: (n_s, Q) complex partial DFT matrix, normalized so E[|A_ij|^2] = 1/n_s
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Create full DFT matrix: F[k,n] = exp(-2*pi*i * k * n / Q) / sqrt(Q)
    k = torch.arange(Q, dtype=torch.float32, device=device).unsqueeze(1)
    n = torch.arange(Q, dtype=torch.float32, device=device).unsqueeze(0)

    # DFT matrix (normalized to have E[|F_ij|^2] = 1/Q)
    angle = -2.0 * math.pi * k * n / Q
    F = torch.complex(torch.cos(angle), torch.sin(angle)) / math.sqrt(Q)
    F = F.to(dtype)

    # Randomly select n_s rows
    row_indices = torch.randperm(Q, device=device)[:n_s]
    A = F[row_indices, :]

    # Scale to get E[|A_ij|^2] = 1/n_s
    A = A * math.sqrt(Q / n_s)

    return A


def create_qr_gaussian_matrix(
    n_s: int,
    Q: int,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.complex64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a QR-orthogonalized Gaussian sensing matrix.

    Steps:
    1. Generate a Q x Q complex Gaussian matrix
    2. QR-decompose it to obtain orthonormal Q (Q x Q)
    3. Take first n_s rows -> A (n_s, Q)
    4. Scale so E[|A_ij|^2] = 1/n_s

    Args:
        n_s: Number of measurements (rows to select)
        Q: Alphabet size
        seed: Random seed
        dtype: Complex data type
        device: Torch device

    Returns:
        A: (n_s, Q) complex QR-orthogonalized matrix, normalized so E[|A_ij|^2] = 1/n_s
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Step 1: random complex Gaussian Q x Q
    R_real = torch.randn(Q, Q, dtype=torch.float32, device=device)
    R_imag = torch.randn(Q, Q, dtype=torch.float32, device=device)
    R = torch.complex(R_real, R_imag).to(dtype)

    # Step 2: QR decomposition (Q_mat has orthonormal columns)
    # Q_mat has E[|Q_ij|^2] = 1/Q (since columns have unit norm)
    Q_mat, _ = torch.linalg.qr(R)

    # Step 3: take first n_s rows
    A = Q_mat[:n_s, :]

    # Step 4: Scale so E[|A_ij|^2] = 1/n_s
    # Currently E[|A_ij|^2] ≈ 1/Q, need to multiply by sqrt(Q/n_s)
    A = A * math.sqrt(Q / n_s)

    return A


def create_hadamard_matrix(
    n_s: int,
    Q: int,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.complex64,
    device: Optional[torch.device] = None,
    complex_phases: bool = False,
) -> torch.Tensor:
    """
    Create a partial Hadamard sensing matrix.

    Uses Sylvester's construction for Hadamard matrices (size must be power of 2).
    Selects n_s rows randomly from a Q x Q Hadamard matrix.

    Args:
        n_s: Number of measurements (rows to select)
        Q: Alphabet size (must be power of 2)
        seed: Random seed for row selection
        dtype: Complex data type
        device: Torch device
        complex_phases: If True, multiply each column by random phase e^{jθ}

    Returns:
        A: (n_s, Q) complex matrix
    """
    # Check Q is power of 2
    if Q & (Q - 1) != 0:
        raise ValueError(f"Hadamard matrix requires Q to be power of 2, got Q={Q}")

    if seed is not None:
        torch.manual_seed(seed)

    # Sylvester's construction: H_1 = [1], H_{2n} = [[H_n, H_n], [H_n, -H_n]]
    def sylvester_hadamard(n):
        if n == 1:
            return torch.ones(1, 1, device=device)
        else:
            H_half = sylvester_hadamard(n // 2)
            return torch.cat([
                torch.cat([H_half, H_half], dim=1),
                torch.cat([H_half, -H_half], dim=1)
            ], dim=0)

    # Create Q x Q Hadamard matrix
    H = sylvester_hadamard(Q)  # entries are ±1

    # Randomly select n_s rows
    row_indices = torch.randperm(Q, device=device)[:n_s]
    A_real = H[row_indices, :]

    # Normalize so E[|A_ij|^2] = 1/n_s
    A_real = A_real / math.sqrt(n_s)

    if complex_phases:
        # Multiply each column by random phase e^{jθ}
        phases = torch.rand(Q, device=device) * 2 * math.pi
        phase_factors = torch.complex(torch.cos(phases), torch.sin(phases))
        A = A_real.to(torch.complex64) * phase_factors.unsqueeze(0)
        A = A.to(dtype)
    else:
        # Real Hadamard (imaginary part = 0)
        A = torch.complex(A_real.to(torch.float32), torch.zeros_like(A_real, dtype=torch.float32))
        A = A.to(dtype)

    return A


def create_complex_hadamard_matrix(
    n_s: int,
    Q: int,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.complex64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a complex Hadamard sensing matrix (Hadamard with random column phases).

    This maintains orthogonality while providing complex diversity like DFT.
    NOTE: Uses arbitrary phases - NOT compatible with standard QAM constellations.

    Args:
        n_s: Number of measurements (rows to select)
        Q: Alphabet size (must be power of 2)
        seed: Random seed
        dtype: Complex data type
        device: Torch device

    Returns:
        A: (n_s, Q) complex matrix with |A_ij| = 1/sqrt(n_s)
    """
    return create_hadamard_matrix(n_s, Q, seed, dtype, device, complex_phases=True)


def create_qpsk_hadamard_matrix(
    n_s: int,
    Q: int,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.complex64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a QPSK-compatible Hadamard sensing matrix.

    Entries are from {±1, ±j}/sqrt(n_s) - compatible with QPSK constellation.
    Each column is multiplied by a random phase from {1, j, -1, -j}.

    Args:
        n_s: Number of measurements (rows to select)
        Q: Alphabet size (must be power of 2)
        seed: Random seed
        dtype: Complex data type
        device: Torch device

    Returns:
        A: (n_s, Q) complex matrix with entries from {±1, ±j}/sqrt(n_s)
    """
    # Check Q is power of 2
    if Q & (Q - 1) != 0:
        raise ValueError(f"Hadamard matrix requires Q to be power of 2, got Q={Q}")

    if seed is not None:
        torch.manual_seed(seed)

    # Sylvester's construction
    def sylvester_hadamard(n):
        if n == 1:
            return torch.ones(1, 1, device=device)
        else:
            H_half = sylvester_hadamard(n // 2)
            return torch.cat([
                torch.cat([H_half, H_half], dim=1),
                torch.cat([H_half, -H_half], dim=1)
            ], dim=0)

    H = sylvester_hadamard(Q)

    # Randomly select n_s rows
    row_indices = torch.randperm(Q, device=device)[:n_s]
    A_real = H[row_indices, :] / math.sqrt(n_s)

    # QPSK phases: {0, π/2, π, 3π/2} → {1, j, -1, -j}
    phase_idx = torch.randint(0, 4, (Q,), device=device)
    qpsk_phases = torch.tensor([1+0j, 0+1j, -1+0j, 0-1j], device=device, dtype=torch.complex64)
    phase_factors = qpsk_phases[phase_idx]

    A = A_real.to(torch.complex64) * phase_factors.unsqueeze(0)
    return A.to(dtype)


def create_sensing_matrix(
    n_s: int,
    Q: int,
    matrix_type: str = 'partial_dft',
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.complex64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a sensing matrix of the specified type.

    Args:
        n_s: Number of measurements
        Q: Alphabet size
        matrix_type: 'partial_dft', 'qr_gaussian', 'hadamard', or 'complex_hadamard'
        seed: Random seed
        dtype: Complex data type
        device: Torch device

    Returns:
        A: (n_s, Q) complex sensing matrix, normalized so E[|A_ij|^2] = 1/n_s
    """
    if matrix_type == 'partial_dft':
        return create_partial_dft_matrix(n_s, Q, seed, dtype, device)
    elif matrix_type == 'qr_gaussian':
        return create_qr_gaussian_matrix(n_s, Q, seed, dtype, device)
    elif matrix_type == 'hadamard':
        return create_hadamard_matrix(n_s, Q, seed, dtype, device)
    elif matrix_type == 'complex_hadamard':
        return create_complex_hadamard_matrix(n_s, Q, seed, dtype, device)
    elif matrix_type == 'qpsk_hadamard':
        return create_qpsk_hadamard_matrix(n_s, Q, seed, dtype, device)
    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}. "
                         f"Use 'partial_dft', 'qr_gaussian', 'hadamard', 'complex_hadamard', or 'qpsk_hadamard'.")


class ModulationEncoder:
    """
    Unified Modulation Encoder for SMV and MMV systems.

    Maps user codewords (symbol indices) to a transmitted signal x.
    The transmitted signal is a superposition of sensing matrix columns.
    """

    def __init__(
        self,
        A: torch.Tensor,
        B: int,
        Eb: float,
        L: int,
    ):
        """
        Initialize modulation encoder.

        Args:
            A: Sensing matrix (n_s, Q) complex, normalized so E[|A_ij|^2] = 1/n_s
            B: Bits per user (log2 of message dimension)
            Eb: Energy per bit in dB (Eb/N0)
            L: Outer code length (number of slots/fading blocks)
        """
        self.A = A
        self.B = B
        self.Eb = Eb
        self.L = L
        self.device = A.device
        self.dtype = A.dtype

        self.n_s = A.shape[0]
        self.Q = A.shape[1]

        # Compute power per symbol (per slot)
        # Psym = (B * 10^(Eb/10)) / L
        Eb_linear = 10.0 ** (Eb / 10.0)
        self.Psym = (B * Eb_linear) / L

    def encode(self, codewords: List[List[int]], slot: int = 0) -> Dict[str, Any]:
        """
        Encode user codewords to transmitted signal for a specific slot.

        For each user k with symbol index idx_k:
            x += sqrt(Psym) * A[:, idx_k]

        Args:
            codewords: List of codewords [[c_0_0, c_0_1, ...], [c_1_0, c_1_1, ...], ...]
                       Each inner list has L symbol indices (one per slot)
            slot: Which slot to encode (default 0, for L=1 use slot=0)

        Returns:
            Dictionary with:
            - x: Transmitted signal (n_s,) complex
            - active_indices: Set of active symbol indices
            - codewords: Input codewords
            - slot: Which slot was encoded
            - metadata: {K, L, Q, n_s, B, Eb, Psym}
        """
        K = len(codewords)

        # Initialize transmitted signal
        x = torch.zeros(self.n_s, dtype=self.dtype, device=self.device)

        # Track active indices
        active_indices: Set[int] = set()

        # Superposition: each user contributes A[:, idx] * sqrt(Psym)
        sqrt_Psym = math.sqrt(self.Psym)
        for k in range(K):
            idx = codewords[k][slot] if slot < len(codewords[k]) else codewords[k][0]
            active_indices.add(idx)
            x = x + sqrt_Psym * self.A[:, idx]

        return {
            'x': x,
            'active_indices': active_indices,
            'codewords': codewords,
            'slot': slot,
            'metadata': {
                'K': K,
                'L': self.L,
                'Q': self.Q,
                'n_s': self.n_s,
                'B': self.B,
                'Eb': self.Eb,
                'Psym': float(self.Psym),
            }
        }

    def encode_symbols(self, active_symbols: List[int]) -> Dict[str, Any]:
        """
        Directly encode a list of active symbols (simpler interface for L=1).

        Args:
            active_symbols: List of symbol indices that are active

        Returns:
            Same as encode()
        """
        codewords = [[sym] for sym in active_symbols]
        return self.encode(codewords, slot=0)

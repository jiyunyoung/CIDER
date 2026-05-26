"""
Batched Modulation Decoder for fast data generation.

Processes multiple samples in parallel across the batch dimension.
Each sample still runs sequential AMP iterations, but all samples
in the batch are processed together.

Usage:
    from modulation_decoder_batch import BatchedModulationDecoder

    decoder = BatchedModulationDecoder(K=2, max_iter=10, sigma2=1.0)
    norm_mag_batch = decoder.forward_batch(Y_batch, A, metadata)
    # Y_batch: [batch_size, n_s]
    # norm_mag_batch: [batch_size, Q]
"""

import torch
import math
from typing import Optional, Dict, Any


@torch.no_grad()
def _amp_mmse_step_batch(
    y: torch.Tensor,        # (batch, n_s)
    A: torch.Tensor,        # (n_s, Q)
    At: torch.Tensor,       # (Q, n_s)
    x: torch.Tensor,        # (batch, Q)
    z: torch.Tensor,        # (batch, n_s)
    lam: float,
    Psym: float,
    n_s: int,
    Q: int,
    return_posterior: bool = False,
):
    """
    Batched AMP iteration with Bayes-optimal MMSE denoiser.

    All tensors have batch dimension as first axis.

    If return_posterior=True, also returns t2 for computing posteriors.
    """
    # tau2: (batch,)
    tau2 = torch.mean((z.conj() * z).real, dim=1, keepdim=True)  # (batch, 1)
    tau2 = torch.clamp(tau2, min=1e-20)

    # s = x + At @ z: (batch, Q)
    # z: (batch, n_s), At: (Q, n_s) -> need batched matmul
    s = x + torch.matmul(z, At.T)  # (batch, n_s) @ (n_s, Q) = (batch, Q)

    u = Psym / tau2  # (batch, 1)
    t1 = 1.0 + 1.0 / u  # (batch, 1)

    abs_s2 = (s.conj() * s).real  # (batch, Q)
    exp_term = torch.exp(-abs_s2 / tau2 * (1.0 / t1))  # (batch, Q)
    t2 = 1.0 + ((1.0 - lam) / lam) * (1.0 + u) * exp_term  # (batch, Q)

    denom = t1 * t2  # (batch, Q)
    x_new = s / denom  # (batch, Q)

    # Compute eta derivative for Onsager correction
    inv_t1 = 1.0 / t1  # (batch, 1)
    inv_t2 = 1.0 / t2  # (batch, Q)
    termA = 1.0 / denom  # (batch, Q)
    termB = (s * inv_t1) * (s.conj() * (inv_t1 / tau2))  # (batch, Q)
    termC = inv_t2 - inv_t2 * inv_t2  # (batch, Q)
    eta_der = termA + termB * termC  # (batch, Q)

    b = (Q / n_s) * torch.mean(eta_der.real, dim=1, keepdim=True)  # (batch, 1)

    # z_new = y - A @ x_new + b * z
    # A @ x_new: (n_s, Q) @ (batch, Q).T -> need (batch, n_s)
    Ax = torch.matmul(x_new, A.T)  # (batch, Q) @ (Q, n_s) = (batch, n_s)
    z_new = y - Ax + b * z  # (batch, n_s)

    if return_posterior:
        return x_new, z_new, t2
    return x_new, z_new


@torch.no_grad()
def amp_smv_batch(
    y: torch.Tensor,        # (batch, n_s)
    A: torch.Tensor,        # (n_s, Q)
    lam: float,
    Psym: float,
    max_iter: int = 10,
    return_posterior: bool = False,
):
    """
    Batched AMP algorithm for single measurement vector (SMV).

    Args:
        y: Received signals (batch, n_s) complex
        A: Sensing matrix (n_s, Q) complex
        lam: Sparsity rate (K/Q)
        Psym: Symbol power
        max_iter: Number of iterations
        return_posterior: If True, also return t2 from final iteration

    Returns:
        x: Estimated sparse signals (batch, Q) complex
        t2: (optional) Posterior denominator (batch, Q) for computing P(active) = 1/t2
    """
    batch_size = y.shape[0]
    n_s, Q = A.shape
    dtype = A.dtype
    device = A.device

    x = torch.zeros(batch_size, Q, dtype=dtype, device=device)
    z = y.clone()
    At = A.conj().T  # (Q, n_s)

    # Run all but last iteration
    for _ in range(max(0, max_iter - 1)):
        x, z = _amp_mmse_step_batch(y, A, At, x, z, lam, Psym, n_s, Q, return_posterior=False)

    # Final iteration with optional posterior
    if return_posterior:
        x, z, t2 = _amp_mmse_step_batch(y, A, At, x, z, lam, Psym, n_s, Q, return_posterior=True)
        return x, t2
    else:
        x, z = _amp_mmse_step_batch(y, A, At, x, z, lam, Psym, n_s, Q, return_posterior=False)
        return x


class BatchedModulationDecoder:
    """
    Batched Modulation Decoder for fast parallel processing.

    Processes entire batch in parallel while maintaining sequential
    AMP iterations within each sample.
    """

    def __init__(
        self,
        K: int,
        max_iter: int = 10,
        sigma2: float = 1.0,
    ):
        """
        Initialize batched decoder.

        Args:
            K: Number of active users
            max_iter: Maximum AMP iterations (default: 10)
            sigma2: Noise variance (default: 1.0)
        """
        self.K = K
        self.max_iter = max_iter
        self.sigma2 = sigma2

    def forward_batch(
        self,
        Y_batch: torch.Tensor,
        A: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        output_type: str = 'norm_mag',
    ) -> torch.Tensor:
        """
        Run batched decoder on received signals.

        Args:
            Y_batch: Received signals (batch, n_s) complex
            A: Sensing matrix (n_s, Q) complex
            metadata: Dict with 'K', 'Q', 'gamma'/'Psym', 'sigma2'
            output_type: Output format
                - 'norm_mag': Normalized magnitude |x|^2 / sigma2 (default, legacy)
                - 'logits': Log-posterior -log(t2), higher = more likely active

        Returns:
            output: (batch, Q) tensor in specified format
        """
        batch_size = Y_batch.shape[0]
        Q = A.shape[1]

        # Get parameters
        K = metadata.get('K', self.K) if metadata else self.K
        lam = K / Q

        if metadata and 'gamma' in metadata:
            Psym = metadata['gamma']
        elif metadata and 'Psym' in metadata:
            Psym = metadata['Psym']
        else:
            Psym = 1.0

        sigma2 = metadata.get('sigma2', self.sigma2) if metadata else self.sigma2

        if output_type == 'logits':
            # Run batched AMP with posterior
            x, t2 = amp_smv_batch(Y_batch, A, lam, Psym, max_iter=self.max_iter, return_posterior=True)
            # Log-posterior: -log(t2), higher means more likely active
            # P(active) = 1/t2, so log P(active) = -log(t2)
            logits = -torch.log(t2 + 1e-20)
            return logits
        else:
            # Legacy: normalized magnitude
            x = amp_smv_batch(Y_batch, A, lam, Psym, max_iter=self.max_iter, return_posterior=False)
            norm_mag = (x.abs() ** 2) / sigma2
            return norm_mag


def test_batched_decoder():
    """Test batched decoder against original."""
    from modulation_decoder import ModulationDecoder
    from modulation_encoder import create_sensing_matrix
    import time

    print("Testing Batched Modulation Decoder")
    print("=" * 50)

    # Parameters
    K = 2
    Q = 64
    n_s = 24
    batch_size = 1000
    Psym = 10.0
    sigma2 = 1.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"K={K}, Q={Q}, n_s={n_s}, batch_size={batch_size}")
    print()

    # Create sensing matrix
    A = create_sensing_matrix(n_s=n_s, Q=Q, matrix_type='partial_dft', seed=42, device=device)

    # Generate random test data
    torch.manual_seed(42)

    # Create batch of received signals
    Y_batch = torch.randn(batch_size, n_s, dtype=torch.complex64, device=device)

    metadata = {'K': K, 'Q': Q, 'gamma': Psym, 'sigma2': sigma2}

    # Test batched decoder
    batched_decoder = BatchedModulationDecoder(K=K, max_iter=10, sigma2=sigma2)

    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.time()
    norm_mag_batched = batched_decoder.forward_batch(Y_batch, A, metadata)
    torch.cuda.synchronize() if device == 'cuda' else None
    batched_time = time.time() - t0

    print(f"Batched decoder: {batched_time*1000:.2f} ms for {batch_size} samples")
    print(f"  = {batch_size/batched_time:.0f} samples/sec")
    print()

    # Test original decoder (serial)
    original_decoder = ModulationDecoder(K=K, max_iter=10, sigma2=sigma2)

    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.time()
    norm_mag_list = []
    for i in range(min(100, batch_size)):  # Only test 100 for speed
        Y = Y_batch[i]
        _, _, norm_mag = original_decoder.forward(Y, A, metadata)
        norm_mag_list.append(norm_mag)
    torch.cuda.synchronize() if device == 'cuda' else None
    original_time = time.time() - t0

    print(f"Original decoder: {original_time*1000:.2f} ms for 100 samples")
    print(f"  = {100/original_time:.0f} samples/sec")
    print()

    # Compare results
    norm_mag_original = torch.stack(norm_mag_list)
    norm_mag_batched_subset = norm_mag_batched[:100]

    max_diff = (norm_mag_original - norm_mag_batched_subset).abs().max().item()
    mean_diff = (norm_mag_original - norm_mag_batched_subset).abs().mean().item()

    print(f"Comparison (first 100 samples):")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print()

    # Speedup
    speedup = (original_time / 100) / (batched_time / batch_size)
    print(f"Speedup: {speedup:.1f}x")


if __name__ == '__main__':
    test_batched_decoder()

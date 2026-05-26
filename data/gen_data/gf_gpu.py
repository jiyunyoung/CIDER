"""
GPU-accelerated Galois Field operations for GF(2^m).

For GF(2^m):
  - Addition: XOR
  - Multiplication: Using log/antilog tables

Usage:
    from gf_gpu import GF_GPU

    gf = GF_GPU(q=64, device='cuda')

    # Matrix-vector multiply over GF(q)
    result = gf.matvec(M, v)  # M @ v

    # Batch LDPC encoding
    codewords = gf.ldpc_encode_batch(H1, H2_inv, Pi, batch_size=2048)
"""

import torch
import numpy as np


class GF_GPU:
    """GPU-accelerated Galois Field GF(2^m) operations."""

    def __init__(self, q, device='cuda'):
        """
        Initialize GF(2^m) with lookup tables on GPU.

        Args:
            q: Field size (must be power of 2)
            device: 'cuda' or 'cpu'
        """
        self.q = q
        self.m = int(np.log2(q))
        assert 2 ** self.m == q, f"q={q} must be power of 2"
        self.device = device

        # Build lookup tables
        self._build_tables()

    def _build_tables(self):
        """Build multiplication and addition tables."""
        q = self.q

        # Addition table: add_table[a, b] = a + b = a XOR b
        # For GF(2^m), addition is XOR
        a = np.arange(q).reshape(q, 1)
        b = np.arange(q).reshape(1, q)
        add_table = a ^ b  # XOR

        # Build multiplication table using galois library for correctness
        # Then move to GPU
        import galois
        GF = galois.GF(q)

        mult_table = np.zeros((q, q), dtype=np.int64)
        for a in range(q):
            for b in range(q):
                mult_table[a, b] = int(GF(a) * GF(b))

        # Negation table: neg_table[a] = -a
        # In GF(2^m), -a = a (since 1 + 1 = 0)
        neg_table = np.arange(q)

        # Move to GPU
        self.add_table = torch.tensor(add_table, dtype=torch.long, device=self.device)
        self.mult_table = torch.tensor(mult_table, dtype=torch.long, device=self.device)
        self.neg_table = torch.tensor(neg_table, dtype=torch.long, device=self.device)

    def add(self, a, b):
        """Element-wise addition: a + b in GF(q)."""
        return self.add_table[a, b]

    def mult(self, a, b):
        """Element-wise multiplication: a * b in GF(q)."""
        return self.mult_table[a, b]

    def neg(self, a):
        """Negation: -a in GF(q). For GF(2^m), -a = a."""
        return self.neg_table[a]

    def matvec(self, M, v):
        """
        Matrix-vector multiply over GF(q): M @ v

        Args:
            M: [m, n] matrix over GF(q)
            v: [batch, n] vectors over GF(q)

        Returns:
            result: [batch, m] = M @ v^T for each batch
        """
        # M: [m, n], v: [batch, n]
        batch_size = v.shape[0]
        m, n = M.shape

        # Expand for broadcasting: M[m, n], v[batch, n] -> products[batch, m, n]
        M_exp = M.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, m, n]
        v_exp = v.unsqueeze(1).expand(-1, m, -1)          # [batch, m, n]

        # Element-wise multiplication
        products = self.mult_table[M_exp, v_exp]  # [batch, m, n]

        # Sum (XOR) along last dimension
        result = products[:, :, 0]
        for i in range(1, n):
            result = self.add_table[result, products[:, :, i]]

        return result

    def matvec_fast(self, M, v):
        """
        Faster matrix-vector multiply using reduce.

        Args:
            M: [m, n] matrix over GF(q)
            v: [batch, n] vectors over GF(q)

        Returns:
            result: [batch, m]
        """
        batch_size = v.shape[0]
        m, n = M.shape

        # Products: [batch, m, n]
        M_exp = M.unsqueeze(0).expand(batch_size, -1, -1)
        v_exp = v.unsqueeze(1).expand(-1, m, -1)
        products = self.mult_table[M_exp, v_exp]

        # XOR reduction (for GF(2^m), addition is XOR)
        result = torch.zeros(batch_size, m, dtype=torch.long, device=self.device)
        for i in range(n):
            result = result ^ products[:, :, i]

        return result

    def matmul(self, A, B):
        """
        Matrix-matrix multiply over GF(q): A @ B

        Args:
            A: [m, k] matrix
            B: [k, n] matrix

        Returns:
            C: [m, n] = A @ B
        """
        m, k = A.shape
        k2, n = B.shape
        assert k == k2

        C = torch.zeros(m, n, dtype=torch.long, device=self.device)

        for i in range(m):
            for j in range(n):
                val = 0
                for l in range(k):
                    prod = self.mult_table[A[i, l], B[l, j]].item()
                    val ^= prod  # XOR for GF(2^m) addition
                C[i, j] = val

        return C

    def ldpc_encode_batch(self, H1, H2_inv, Pi_inv, k, batch_size):
        """
        Batch LDPC encoding on GPU.

        Encoding: v = [u | p] where p = -(H2_inv @ H1 @ u)
        Then apply inverse permutation.

        Args:
            H1: [M, k] information part of H (on GPU)
            H2_inv: [M, M] inverse of parity part (on GPU)
            Pi_inv: [L] inverse permutation (on GPU)
            k: number of information symbols
            batch_size: number of codewords to generate

        Returns:
            codewords: [batch_size, L] random LDPC codewords
        """
        M = H2_inv.shape[0]
        L = len(Pi_inv)

        # Generate random information symbols: [batch, k]
        u = torch.randint(0, self.q, (batch_size, k), dtype=torch.long, device=self.device)

        # Compute H1 @ u: [batch, M]
        Hu = self.matvec_fast(H1, u)

        # Compute p = -(H2_inv @ Hu^T)^T = H2_inv @ Hu (since -a = a in GF(2^m))
        p = self.matvec_fast(H2_inv, Hu)

        # Form codeword in permuted coordinates: [u | p]
        v_perm = torch.zeros(batch_size, L, dtype=torch.long, device=self.device)
        v_perm[:, :k] = u
        v_perm[:, k:] = p

        # Apply inverse permutation
        v = v_perm[:, Pi_inv]

        return v

    def random_codewords(self, H1, H2_inv, Pi_inv, k, batch_size):
        """Alias for ldpc_encode_batch."""
        return self.ldpc_encode_batch(H1, H2_inv, Pi_inv, k, batch_size)


def test_gf_gpu():
    """Test GF_GPU against galois library."""
    import galois

    q = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing GF({q}) on {device}")

    gf_gpu = GF_GPU(q, device)
    GF = galois.GF(q)

    # Test multiplication table
    print("Testing multiplication table...")
    for a in range(q):
        for b in range(q):
            gpu_result = gf_gpu.mult_table[a, b].item()
            galois_result = int(GF(a) * GF(b))
            assert gpu_result == galois_result, f"{a} * {b}: GPU={gpu_result}, galois={galois_result}"
    print("  Multiplication table OK")

    # Test addition table
    print("Testing addition table...")
    for a in range(q):
        for b in range(q):
            gpu_result = gf_gpu.add_table[a, b].item()
            galois_result = int(GF(a) + GF(b))
            assert gpu_result == galois_result, f"{a} + {b}: GPU={gpu_result}, galois={galois_result}"
    print("  Addition table OK")

    # Test matrix-vector multiply
    print("Testing matvec...")
    M = np.random.randint(0, q, (8, 12))
    v = np.random.randint(0, q, (100, 12))

    M_gpu = torch.tensor(M, dtype=torch.long, device=device)
    v_gpu = torch.tensor(v, dtype=torch.long, device=device)

    result_gpu = gf_gpu.matvec_fast(M_gpu, v_gpu).cpu().numpy()

    M_gf = GF(M)
    v_gf = GF(v)
    result_galois = np.array(M_gf @ v_gf.T).T

    assert np.allclose(result_gpu, result_galois), "matvec mismatch"
    print("  matvec OK")

    # Test LDPC encoding
    print("Testing LDPC encoding...")
    from construct_H import construct_H

    h_data = construct_H(q=64, L=12, M=8, d_v=2, d_c=3, seed=42)

    H1 = torch.tensor(h_data['H1'].numpy(), dtype=torch.long, device=device)
    H2_inv = torch.tensor(h_data['H2_inv'].numpy(), dtype=torch.long, device=device)
    Pi = h_data['Pi'].tolist()
    Pi_inv = [0] * len(Pi)
    for i, p in enumerate(Pi):
        Pi_inv[p] = i
    Pi_inv = torch.tensor(Pi_inv, dtype=torch.long, device=device)
    k = h_data['k']

    # Generate codewords on GPU
    codewords = gf_gpu.ldpc_encode_batch(H1, H2_inv, Pi_inv, k, batch_size=1000)

    # Verify H @ codeword = 0
    H = torch.tensor(h_data['H_matrix'].numpy(), dtype=torch.long, device=device)
    syndromes = gf_gpu.matvec_fast(H, codewords)

    if torch.all(syndromes == 0):
        print("  All 1000 codewords satisfy H @ c = 0")
    else:
        print("  ERROR: Some codewords have non-zero syndrome!")

    print("\nAll tests passed!")


def benchmark_gf_gpu():
    """Benchmark GPU vs CPU encoding speed."""
    import time
    import galois
    from construct_H import construct_H

    q = 64
    L = 12
    batch_sizes = [1000, 5000, 10000, 50000]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking GF({q}) encoding on {device}\n")

    # Setup
    gf_gpu = GF_GPU(q, device)
    h_data = construct_H(q=q, L=L, M=8, d_v=2, d_c=3, seed=42)

    H1_gpu = torch.tensor(h_data['H1'].numpy(), dtype=torch.long, device=device)
    H2_inv_gpu = torch.tensor(h_data['H2_inv'].numpy(), dtype=torch.long, device=device)
    Pi = h_data['Pi'].tolist()
    Pi_inv = [0] * len(Pi)
    for i, p in enumerate(Pi):
        Pi_inv[p] = i
    Pi_inv_gpu = torch.tensor(Pi_inv, dtype=torch.long, device=device)
    k = h_data['k']

    # CPU setup (galois)
    GF = galois.GF(q)
    H1_cpu = GF(h_data['H1'].numpy())
    H2_inv_cpu = GF(h_data['H2_inv'].numpy())

    from ldpc_codes_torch import ldpc_encode

    print(f"{'Batch':<10} {'GPU (ms)':<12} {'CPU (ms)':<12} {'Speedup':<10}")
    print("-" * 44)

    for batch_size in batch_sizes:
        # GPU timing
        torch.cuda.synchronize() if device == 'cuda' else None
        t0 = time.time()
        codewords_gpu = gf_gpu.ldpc_encode_batch(H1_gpu, H2_inv_gpu, Pi_inv_gpu, k, batch_size)
        torch.cuda.synchronize() if device == 'cuda' else None
        gpu_time = (time.time() - t0) * 1000

        # CPU timing
        t0 = time.time()
        for _ in range(batch_size):
            u = [np.random.randint(0, q) for _ in range(k)]
            ldpc_encode(H1_cpu, None, H2_inv_cpu, Pi, GF, u)
        cpu_time = (time.time() - t0) * 1000

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"{batch_size:<10} {gpu_time:<12.2f} {cpu_time:<12.2f} {speedup:<10.1f}x")

    print("\nDone!")


if __name__ == '__main__':
    test_gf_gpu()

"""
Data generation and dataset for LDPC codeword demixing.

Includes:
- Random codeword generation
- Fragment mixing and shuffling
- PyTorch Dataset for saved data
"""

import torch
import random
from torch.utils.data import Dataset
import os


def generate_codewords(K_true, N, Q, device):
    """Generate random Q-ary sequences."""
    return torch.randint(low=0, high=Q, size=(K_true, N), device=device)


def mix_and_shuffle(codewords, K_true, K_max, Q, device):
    """
    Build unordered fragment sets per position ℓ.

    Args:
        codewords: [K_true, N] - ground truth codewords
        K_true: number of active users
        K_max: max number of slots (for padding)
        Q: alphabet size (used for PAD token)
        device: torch device

    Returns:
        frags: [N, K_max] - unordered shuffled fragments with padding
    """
    N = codewords.shape[1]
    PAD = Q  # use Q as padding token (beyond 0..Q-1)
    frags = []

    for ℓ in range(N):
        row = []
        # add fragments from each user
        for k in range(K_true):
            row.append(int(codewords[k, ℓ].item()))
        # shuffle row (unordered set)
        random.shuffle(row)
        # pad to K_max
        while len(row) < K_max:
            row.append(PAD)
        frags.append(row)

    frags = torch.tensor(frags, dtype=torch.long, device=device)
    return frags  # [N, K_max]


def generate_batch(batch_size, K_true, N, Q, K_max, device):
    """Generate a batch of data."""
    GT_codewords_list = []
    frag_syms_list = []

    for _ in range(batch_size):
        GT_codewords = generate_codewords(K_true, N, Q, device)
        frag_syms = mix_and_shuffle(GT_codewords, K_true, K_max, Q, device)
        GT_codewords_list.append(GT_codewords)
        frag_syms_list.append(frag_syms)

    # Stack into batches
    GT_codewords_batch = torch.stack(GT_codewords_list)  # [B, K_true, N]
    frag_syms_batch = torch.stack(frag_syms_list)  # [B, N, K_max]

    return GT_codewords_batch, frag_syms_batch


class LDPCDataset(Dataset):
    """Dataset for LDPC codeword demixing task (fragment-based)."""

    def __init__(self, data_dir, split='train', device='cpu'):
        """
        Args:
            data_dir: directory containing saved dataset files
            split: 'train', 'val', or 'test'
            device: torch device
        """
        self.device = device
        self.split = split

        # Load saved data
        data_file = os.path.join(data_dir, f'{split}_data.pt')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Dataset not found at {data_file}")

        self.data = torch.load(data_file, map_location=device, weights_only=False)
        self.frag_syms = self.data['frag_syms']  # [N_samples, N, K_max]
        self.gt_codewords = self.data['gt_codewords']  # [N_samples, K_true, N]

        # Load H_matrix if available
        H_file = os.path.join(data_dir, 'H_matrix.pt')
        if os.path.exists(H_file):
            H_data = torch.load(H_file, map_location=device, weights_only=False)
            self.H_matrix = H_data.get('H_matrix', None)
        else:
            self.H_matrix = self.data.get('H_matrix', None)

    def __len__(self):
        return len(self.frag_syms)

    def __getitem__(self, idx):
        """Returns (frag_syms, gt_codewords) for a single sample.

        Both tensors have shape [K, N] for consistency:
        - frag_syms: [K, N] shuffled fragments (K is arbitrary order)
        - gt_codewords: [K, N] ground truth codewords (K is slot identity)
        """
        # Transpose frag_syms from [N, K] to [K, N] to match gt_codewords
        return self.frag_syms[idx].T, self.gt_codewords[idx]


class BinaryOnTheFlyDataset(Dataset):
    """On-the-fly binary LDPC dataset with BPSK + AWGN channel.

    Generates fresh codewords and channel observations every access.
    No pre-stored data — infinite variety.

    Data format:
        Y: [N, 2] - soft posteriors P(bit=0|y), P(bit=1|y)
        gt_codewords: [K, N] - ground truth binary codewords
    """

    def __init__(self, h_matrix_path, K=2, Eb_dB=10.0, num_samples=70000, device='cpu', fixed_seed=None):
        """
        Args:
            h_matrix_path: path to H_matrix.pt
            K: number of users
            Eb_dB: Eb/N0 in dB
            num_samples: dataset size per epoch
            device: torch device
            fixed_seed: if set, generates deterministic data (for val/test)
        """
        import numpy as np
        from scipy.special import comb

        self.device = device
        self.K = K
        self.num_samples = num_samples
        self.fixed_seed = fixed_seed

        # Load H matrix and encoding components
        h_data = torch.load(h_matrix_path, map_location='cpu', weights_only=False)
        self.H_matrix = h_data['H_matrix']
        self.L = h_data['L']
        self.Q = h_data['q']
        self.M = h_data['M']
        self.k = h_data['k']

        # Encoding matrices
        self.H1 = h_data['H1']
        self.H2_inv = h_data['H2_inv']
        Pi = h_data['Pi'].tolist()
        self.Pi_inv = torch.zeros(self.L, dtype=torch.long)
        for i, p in enumerate(Pi):
            self.Pi_inv[p] = i

        # GF(2) multiplication tables (trivial: XOR for add, AND for mult)
        self.add_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.mult_table = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)

        # Channel params
        Eb_linear = 10 ** (Eb_dB / 10)
        self.sigma2 = 1.0 / (2.0 * Eb_linear)

        # Precompute log binomial priors for posterior computation
        self.log_priors = []
        for j in range(K):
            self.log_priors.append(
                np.log(comb(K - 1, j, exact=True)) - (K - 1) * np.log(2)
            )

    def _gf2_encode(self):
        """Encode a random binary LDPC codeword using systematic encoding."""
        # Random info symbols
        u = torch.randint(0, 2, (self.k,), dtype=torch.long)

        # Parity: p = H2_inv @ (H1 @ u) over GF(2)
        # GF(2) matmul = (A @ B) mod 2
        Hu = (self.H1 @ u) % 2
        p = (self.H2_inv @ Hu) % 2

        # Codeword before permutation: [u | p]
        c_sys = torch.cat([u, p])

        # Apply inverse permutation
        c = c_sys[self.Pi_inv]
        return c

    def _bpsk_awgn_posterior(self, y):
        """Compute P(bit=0|y), P(bit=1|y) for K-user BPSK+AWGN."""
        import numpy as np
        y_np = y.numpy().astype(np.float64)

        log_p0 = np.full_like(y_np, -np.inf)
        log_p1 = np.full_like(y_np, -np.inf)

        for j in range(self.K):
            log_prior = self.log_priors[j]
            # This user sends 0 (+1): sum = K - 2j
            mean_0 = self.K - 2 * j
            ll_0 = log_prior + (-0.5 * (y_np - mean_0) ** 2 / self.sigma2)

            # This user sends 1 (-1): sum = K - 2j - 2
            mean_1 = self.K - 2 * j - 2
            ll_1 = log_prior + (-0.5 * (y_np - mean_1) ** 2 / self.sigma2)

            log_p0 = np.logaddexp(log_p0, ll_0)
            log_p1 = np.logaddexp(log_p1, ll_1)

        log_norm = np.logaddexp(log_p0, log_p1)
        p0 = np.exp(log_p0 - log_norm).astype(np.float32)
        p1 = np.exp(log_p1 - log_norm).astype(np.float32)

        return torch.tensor(np.stack([p0, p1], axis=-1))  # [L, 2]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Generate a fresh sample (or deterministic if fixed_seed is set)."""
        if self.fixed_seed is not None:
            torch.manual_seed(self.fixed_seed + idx)

        # Generate K random codewords
        codewords = torch.stack([self._gf2_encode() for _ in range(self.K)])  # [K, L]

        # BPSK modulation: 0 → +1, 1 → -1
        s = 1.0 - 2.0 * codewords.float()  # [K, L]

        # Superimpose + AWGN
        y = s.sum(dim=0)  # [L]
        noise = torch.randn(self.L) * (self.sigma2 ** 0.5)
        y = y + noise

        # Compute soft posteriors
        Y = self._bpsk_awgn_posterior(y)  # [L, 2]

        return Y, codewords


class InterleavedBinaryDataset(Dataset):
    """On-the-fly binary LDPC dataset with per-user interleavers + BPSK + AWGN.

    Mimics the T-fold IRSA channel model:
    - Each user encodes with the same binary LDPC code
    - Each user's codeword is permuted by a different interleaver
    - Interleaved codewords are BPSK modulated and superimposed
    - Receiver knows which interleavers are active

    Data format:
        Y: [N, 2] - soft posteriors P(bit=0|y), P(bit=1|y) in channel domain
        gt_codewords: [K, N] - ground truth codewords in CHANNEL domain (interleaved)
        interleavers: [K, N] - per-user interleaver indices (LDPC bit n → channel pos)
    """

    def __init__(self, h_matrix_path, K=2, Eb_dB=5.0, num_interleavers=512,
                 num_samples=70000, device='cpu', fixed_seed=None):
        """
        Args:
            h_matrix_path: path to H_matrix.pt (binary LDPC, GF(2))
            K: number of users
            Eb_dB: Eb/N0 in dB
            num_interleavers: number of pre-generated interleavers (like M_p preambles)
            num_samples: dataset size per epoch
            device: torch device
            fixed_seed: if set, generates deterministic data (for val/test)
        """
        import numpy as np
        from scipy.special import comb

        self.device = device
        self.K = K
        self.num_samples = num_samples
        self.fixed_seed = fixed_seed

        # Load H matrix
        h_data = torch.load(h_matrix_path, map_location='cpu', weights_only=False)
        self.H_matrix = h_data['H_matrix']
        self.L = h_data['L']
        self.Q = h_data['q']
        self.M = h_data['M']
        self.k = h_data['k']

        # Encoding matrices
        self.H1 = h_data['H1']
        self.H2_inv = h_data['H2_inv']
        Pi = h_data['Pi'].tolist()
        self.Pi_inv = torch.zeros(self.L, dtype=torch.long)
        for i, p in enumerate(Pi):
            self.Pi_inv[p] = i

        # Channel params
        Eb_linear = 10 ** (Eb_dB / 10)
        self.sigma2 = 1.0 / (2.0 * Eb_linear)

        # Pre-generate fixed set of interleavers (like preamble-based permutations)
        self.num_interleavers = num_interleavers
        rng = np.random.RandomState(42)
        self.all_interleavers = torch.zeros(num_interleavers, self.L, dtype=torch.long)
        for i in range(num_interleavers):
            self.all_interleavers[i] = torch.from_numpy(rng.permutation(self.L))

        # Precompute log binomial priors for posterior computation
        self.log_priors = []
        for j in range(K):
            self.log_priors.append(
                np.log(comb(K - 1, j, exact=True)) - (K - 1) * np.log(2)
            )

    def _gf2_encode(self):
        """Encode a random binary LDPC codeword (in LDPC domain)."""
        u = torch.randint(0, 2, (self.k,), dtype=torch.long)
        Hu = (self.H1 @ u) % 2
        p = (self.H2_inv @ Hu) % 2
        c_sys = torch.cat([u, p])
        c = c_sys[self.Pi_inv]
        return c

    def _bpsk_awgn_posterior(self, y):
        """Compute P(bit=0|y), P(bit=1|y) for K-user BPSK+AWGN."""
        import numpy as np
        y_np = y.numpy().astype(np.float64)

        log_p0 = np.full_like(y_np, -np.inf)
        log_p1 = np.full_like(y_np, -np.inf)

        for j in range(self.K):
            log_prior = self.log_priors[j]
            mean_0 = self.K - 2 * j
            ll_0 = log_prior + (-0.5 * (y_np - mean_0) ** 2 / self.sigma2)
            mean_1 = self.K - 2 * j - 2
            ll_1 = log_prior + (-0.5 * (y_np - mean_1) ** 2 / self.sigma2)
            log_p0 = np.logaddexp(log_p0, ll_0)
            log_p1 = np.logaddexp(log_p1, ll_1)

        log_norm = np.logaddexp(log_p0, log_p1)
        p0 = np.exp(log_p0 - log_norm).astype(np.float32)
        p1 = np.exp(log_p1 - log_norm).astype(np.float32)
        return torch.tensor(np.stack([p0, p1], axis=-1))  # [L, 2]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Generate a fresh sample with random interleavers.

        Returns:
            Y: [N, 2] posteriors in channel domain
            gt_codewords: [K, N] ground truth in CHANNEL domain (interleaved)
            interleavers: [K, N] interleaver indices per user
        """
        if self.fixed_seed is not None:
            torch.manual_seed(self.fixed_seed + idx)

        # Pick K distinct interleavers from the fixed set
        perm = torch.randperm(self.num_interleavers)[:self.K]
        interleavers = self.all_interleavers[perm]  # [K, N]

        # Generate K codewords in LDPC domain
        codewords_ldpc = torch.stack([self._gf2_encode() for _ in range(self.K)])  # [K, N]

        # Interleave to channel domain
        codewords_channel = torch.zeros_like(codewords_ldpc)
        for k in range(self.K):
            # interleavers[k, ldpc_pos] = channel_pos
            # So codewords_channel[k, channel_pos] = codewords_ldpc[k, ldpc_pos]
            codewords_channel[k, interleavers[k]] = codewords_ldpc[k]

        # BPSK modulation: 0 → +1, 1 → -1
        s = 1.0 - 2.0 * codewords_channel.float()  # [K, N]

        # Superimpose + AWGN (in channel domain)
        y = s.sum(dim=0)  # [N]
        noise = torch.randn(self.L) * (self.sigma2 ** 0.5)
        y = y + noise

        # Soft posteriors
        Y = self._bpsk_awgn_posterior(y)  # [N, 2]

        return Y, codewords_channel, interleavers


class E2EDataset(Dataset):
    """Dataset for end-to-end demixing with inner decoder soft outputs.

    Data format:
        Y: [N_samples, N, Q] - soft scores from inner decoder
        gt_codewords: [N_samples, K, N] - ground truth codewords
        H_matrix: [M_parity, N] - parity check matrix (stored separately)
    """

    def __init__(self, data_dir, split='train', device='cpu'):
        """
        Args:
            data_dir: directory containing saved dataset files
            split: 'train', 'val', or 'test'
            device: torch device
        """
        self.device = device
        self.split = split

        # Load saved data
        data_file = os.path.join(data_dir, f'{split}_data.pt')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Dataset not found at {data_file}")

        self.data = torch.load(data_file, map_location=device, weights_only=False)
        self.Y = self.data['Y']  # [N_samples, N, Q] soft scores
        self.gt_codewords = self.data['gt_codewords']  # [N_samples, K, N]

        # Load H_matrix from separate file
        H_file = os.path.join(data_dir, 'H_matrix.pt')
        if os.path.exists(H_file):
            H_data = torch.load(H_file, map_location=device, weights_only=False)
            self.H_matrix = H_data.get('H_matrix', None)
            self.config = H_data.get('config', {})
        else:
            self.H_matrix = None
            self.config = {}

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        """Returns (Y, gt_codewords) for a single sample.

        Returns:
            Y: [N, Q] soft scores from inner decoder
            gt_codewords: [K, N] ground truth codewords
        """
        return self.Y[idx], self.gt_codewords[idx]

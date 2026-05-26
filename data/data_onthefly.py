"""
On-the-fly Q-ary LDPC dataset with sensing matrix + AMP inner channel.

Supports mixed Eb/N0 training — each sample gets a random Eb/N0 from a range.
Fresh codewords and noise every access.
"""

import torch
import numpy as np
from torch.utils.data import Dataset

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gen_data'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gen_data', 'noisy_channel'))

from gf_gpu import GF_GPU
from noisy_channel.modulation_encoder import ModulationEncoder, create_sensing_matrix
from noisy_channel.modulation_decoder_batch import BatchedModulationDecoder


class QaryOnTheFlyDataset(Dataset):
    """On-the-fly Q-ary LDPC dataset with inner channel simulation.

    Data format:
        Y: [N, Q] - soft likelihoods from inner decoder
        gt_codewords: [K, N] - ground truth codewords in GF(Q)
    """

    def __init__(self, h_matrix_path, K=2, Eb_dB=10.0, n_s=24, sigma2=1.0,
                 matrix_type='partial_dft', num_samples=70000,
                 fixed_seed=None, Eb_range=None, device='cpu'):
        """
        Args:
            h_matrix_path: path to H_matrix.pt
            K: number of users
            Eb_dB: default Eb/N0 in dB (used when Eb_range is None)
            n_s: inner code length (sensing matrix rows)
            sigma2: noise variance for inner decoder
            matrix_type: sensing matrix type
            num_samples: epoch size
            fixed_seed: deterministic data for val/test
            Eb_range: (min_dB, max_dB) — sample uniform Eb/N0 per sample
            device: 'cpu' or 'cuda' for encoding (DataLoader workers use cpu)
        """
        self.K = K
        self.num_samples = num_samples
        self.fixed_seed = fixed_seed
        self.Eb_range = Eb_range
        self.Eb_dB = Eb_dB
        self.sigma2 = sigma2
        self.n_s = n_s

        # Load H matrix
        h_data = torch.load(h_matrix_path, map_location='cpu', weights_only=False)
        self.H_matrix = h_data['H_matrix']
        self.L = h_data['L']
        self.Q = h_data['q']
        self.M = h_data['M']
        self.k = h_data['k']

        # Encoding components
        self.gf_gpu = GF_GPU(self.Q, 'cpu')
        self.H1 = torch.tensor(h_data['H1'].numpy(), dtype=torch.long)
        self.H2_inv = torch.tensor(h_data['H2_inv'].numpy(), dtype=torch.long)
        Pi = h_data['Pi'].tolist()
        self.Pi_inv = torch.zeros(self.L, dtype=torch.long)
        for i, p in enumerate(Pi):
            self.Pi_inv[p] = i

        # Sensing matrix (fixed across samples)
        self.A = create_sensing_matrix(
            n_s=n_s, Q=self.Q, matrix_type=matrix_type, seed=42, device='cpu')

        # Inner decoder
        self.decoder = BatchedModulationDecoder(K=K, max_iter=10, sigma2=sigma2)

        # Precompute bits per symbol
        self.bits_per_symbol = int(np.log2(self.Q))
        self.B = self.k * self.bits_per_symbol

    def _make_encoder(self, Eb_dB):
        """Create encoder with given Eb/N0."""
        return ModulationEncoder(A=self.A, B=self.B, Eb=Eb_dB, L=self.L)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.fixed_seed is not None:
            torch.manual_seed(self.fixed_seed + idx)
            np.random.seed(self.fixed_seed + idx)

        # Sample Eb/N0
        if self.Eb_range is not None:
            Eb_dB = np.random.uniform(self.Eb_range[0], self.Eb_range[1])
        else:
            Eb_dB = self.Eb_dB

        encoder = self._make_encoder(Eb_dB)
        sqrt_Psym = np.sqrt(encoder.Psym)
        metadata = {'K': self.K, 'Q': self.Q, 'gamma': encoder.Psym, 'sigma2': self.sigma2}

        # Generate K codewords
        codewords = self.gf_gpu.ldpc_encode_batch(
            self.H1, self.H2_inv, self.Pi_inv, self.k, self.K)  # [K, L]

        # Batched encoding across all L positions:
        # A[:, codewords] -> [n_s, K, L]; sum over K, transpose -> [L, n_s]
        cw_long = codewords.long()
        A_sel = self.A[:, cw_long]                     # [n_s, K, L]
        x_all = float(sqrt_Psym) * A_sel.sum(dim=1).T  # [L, n_s] complex

        noise_r = torch.randn(self.L, self.n_s) * np.sqrt(self.sigma2 / 2)
        noise_i = torch.randn(self.L, self.n_s) * np.sqrt(self.sigma2 / 2)
        noise = torch.complex(noise_r, noise_i).to(self.A.dtype)
        y_recv = x_all + noise                         # [L, n_s]

        # Single batched AMP call: [L, n_s] -> [L, Q]
        # Use 'logits' (-log(t2), log-posterior) to match cached datasets.
        Y = self.decoder.forward_batch(y_recv, self.A, metadata,
                                       output_type='logits').float()

        return Y, codewords

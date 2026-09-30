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
                 fixed_seed=None, Eb_range=None, device='cpu',
                 fading='none', rician_k_dB=None, fading_coherence='slot',
                 return_yrecv=False, gamma_scale=1.0,
                 noise_type='gaussian', noise_nu=3.0, noise_eps=0.1,
                 noise_kappa=10.0):
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
            fading: 'none' (AWGN superposition, the paper's setting),
                'rayleigh' (h ~ CN(0,1) per user), or 'rician' (LoS + diffuse,
                strength set by rician_k_dB).
            rician_k_dB: Rician K-factor in dB, required when fading='rician'.
                K -> +inf recovers AWGN, K = -inf recovers Rayleigh, so a sweep
                over this knob traces a graceful-degradation curve.
            fading_coherence: 'slot' (i.i.d. per user per slot, block fading) or
                'frame' (one coefficient per user, constant across the L slots).
            return_yrecv: if True, __getitem__ also returns the raw received
                slots y_recv [L, n_s] (complex) and the scalar Psym, so a
                front-end active-user-count estimator can be built from the same
                observation the AMP inner detector consumes. Default False keeps
                the (Y, codewords) signature the dataloader expects.

        Fading is applied to the transmitted superposition only; the AMP inner
        detector is left unchanged and receives no CSI. This is deliberate — it
        is the mismatched, non-coherent case, which is the honest hard setting
        for a decoder trained under AWGN. E[|h|^2] = 1 in every mode, so the
        nominal Eb/N0 is preserved and curves stay comparable across modes.
        """
        self.K = K
        self.num_samples = num_samples
        self.fixed_seed = fixed_seed
        self.Eb_range = Eb_range
        self.Eb_dB = Eb_dB
        self.sigma2 = sigma2
        self.n_s = n_s

        if fading not in ('none', 'rayleigh', 'rician'):
            raise ValueError(f"unknown fading mode: {fading!r}")
        if fading == 'rician' and rician_k_dB is None:
            raise ValueError("fading='rician' requires rician_k_dB")
        if fading_coherence not in ('slot', 'frame'):
            raise ValueError(f"unknown fading_coherence: {fading_coherence!r}")
        self.fading = fading
        self.rician_k_dB = rician_k_dB
        self.fading_coherence = fading_coherence
        self.return_yrecv = return_yrecv
        # Non-Gaussian channel noise. All variants are power-normalized so that
        # E[|n|^2] = sigma2 (nominal Eb/N0 unchanged, curves comparable across
        # types), differing only in TAIL shape. The AMP detector assumes Gaussian
        # noise (Gaussian-optimal MMSE denoiser) and CIDER was trained on AWGN, so
        # this is a distribution mismatch at both stages. 'gaussian' (default) is
        # bit-exact with the AWGN path. noise_nu = Student-t dof (heavy tails),
        # noise_eps / noise_kappa = impulse rate / power ratio (Bernoulli-Gaussian).
        if noise_type not in ('gaussian', 'laplace', 'student_t', 'uniform',
                              'impulsive'):
            raise ValueError(f"unknown noise_type: {noise_type!r}")
        self.noise_type = noise_type
        self.noise_nu = float(noise_nu)
        self.noise_eps = float(noise_eps)
        self.noise_kappa = float(noise_kappa)

        # Detector power (gamma = Psym) MIS-calibration. The channel always
        # transmits the true Psym; only the AMP detector's assumed power is
        # scaled by gamma_scale. gamma_scale != 1 models a received-power / AGC /
        # open-loop-power-control error: the receiver believes users arrive at
        # gamma_scale x the true power (10*log10(gamma_scale) dB off). Default 1.0
        # keeps the perfectly-calibrated AWGN path bit-exact.
        self.gamma_scale = gamma_scale

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

    def _sample_fading(self):
        """Per-user channel gains h, shape [K, L], normalized to E[|h|^2] = 1.

        Returns None under fading='none' so the AWGN path stays bit-exact with
        previously generated datasets.
        """
        if self.fading == 'none':
            return None

        # 'frame' coherence draws one gain per user and broadcasts over slots.
        n_slots = 1 if self.fading_coherence == 'frame' else self.L
        diffuse = torch.complex(torch.randn(self.K, n_slots),
                                torch.randn(self.K, n_slots)) / np.sqrt(2.0)

        if self.fading == 'rayleigh':
            h = diffuse
        else:
            # Rician: h = sqrt(k/(k+1)) * LoS + sqrt(1/(k+1)) * diffuse.
            # The LoS is the deterministic specular path: UNIT magnitude, but a
            # per-user phase theta_k ~ U[0, 2pi). Different users sit at different
            # locations, so their LoS phases are independent -- fixing them all to
            # 0 (a phase-aligned special case) would misstate the multi-user
            # interference. theta_k is QUASI-STATIC: drawn once per user and held
            # across the L slots (LoS phase is geometric), while the diffuse part
            # keeps varying per slot. |LoS| = 1 so E[|h|^2] = 1 is preserved.
            k_lin = 10.0 ** (self.rician_k_dB / 10.0)
            theta = torch.rand(self.K, 1) * (2.0 * np.pi)          # [K, 1] per user
            los = torch.complex(torch.cos(theta), torch.sin(theta))  # e^{j theta_k}
            h = (np.sqrt(k_lin / (k_lin + 1.0)) * los             # broadcasts [K,1]->[K,n_slots]
                 + np.sqrt(1.0 / (k_lin + 1.0)) * diffuse)

        if n_slots == 1:
            h = h.expand(self.K, self.L)
        return h.to(self.A.dtype)

    def _sample_noise(self):
        """Complex channel noise [L, n_s], E[|n|^2] = sigma2 for every type.

        Each real/imag component carries variance sigma2/2. Only the tail shape
        changes: gaussian (baseline) < uniform (light tails) < laplace < student_t
        (heavy) and impulsive (spiky). 'gaussian' preserves the exact RNG order of
        the original AWGN path, so that path stays bit-exact.
        """
        L, n_s, s = self.L, self.n_s, self.sigma2
        v = s / 2.0                                        # per-component variance

        if self.noise_type == 'gaussian':
            nr = torch.randn(L, n_s) * np.sqrt(v)
            ni = torch.randn(L, n_s) * np.sqrt(v)
        elif self.noise_type == 'uniform':
            a = np.sqrt(3.0 * v)                           # U(-a,a): var = a^2/3
            nr = (torch.rand(L, n_s) * 2 - 1) * a
            ni = (torch.rand(L, n_s) * 2 - 1) * a
        elif self.noise_type == 'laplace':
            b = np.sqrt(v / 2.0)                           # Laplace(b): var = 2 b^2
            nr = torch.distributions.Laplace(0.0, b).sample((L, n_s))
            ni = torch.distributions.Laplace(0.0, b).sample((L, n_s))
        elif self.noise_type == 'student_t':
            nu = self.noise_nu                             # var = nu/(nu-2), nu>2
            sc = np.sqrt(v * (nu - 2.0) / nu)
            nr = torch.distributions.StudentT(nu).sample((L, n_s)) * sc
            ni = torch.distributions.StudentT(nu).sample((L, n_s)) * sc
        else:  # 'impulsive' Bernoulli-Gaussian (Middleton-like): eps spikes, kappa x power
            eps, kap = self.noise_eps, self.noise_kappa
            v_bg = v / ((1 - eps) + eps * kap)             # keep total var = v
            def imp():
                mask = (torch.rand(L, n_s) < eps).float()
                var = v_bg * (1 - mask) + v_bg * kap * mask
                return torch.randn(L, n_s) * torch.sqrt(var)
            nr, ni = imp(), imp()

        return torch.complex(nr, ni).to(self.A.dtype)

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
        sqrt_Psym = np.sqrt(encoder.Psym)                     # TRUE transmit amplitude
        # Detector's ASSUMED power = true Psym * gamma_scale (calibration error);
        # the transmitted signal below still uses the true sqrt_Psym.
        metadata = {'K': self.K, 'Q': self.Q,
                    'gamma': encoder.Psym * self.gamma_scale, 'sigma2': self.sigma2}

        # Generate K codewords
        codewords = self.gf_gpu.ldpc_encode_batch(
            self.H1, self.H2_inv, self.Pi_inv, self.k, self.K)  # [K, L]

        # Batched encoding across all L positions:
        # A[:, codewords] -> [n_s, K, L]; sum over K, transpose -> [L, n_s]
        cw_long = codewords.long()
        A_sel = self.A[:, cw_long]                     # [n_s, K, L]

        h = self._sample_fading()                      # [K, L] or None
        if h is not None:
            A_sel = A_sel * h.unsqueeze(0)             # scale each user's symbol
        x_all = float(sqrt_Psym) * A_sel.sum(dim=1).T  # [L, n_s] complex

        noise = self._sample_noise()                   # AWGN by default; bit-exact
        y_recv = x_all + noise                         # [L, n_s]

        # Single batched AMP call: [L, n_s] -> [L, Q]
        # Use 'logits' (-log(t2), log-posterior) to match cached datasets.
        Y = self.decoder.forward_batch(y_recv, self.A, metadata,
                                       output_type='logits').float()

        if self.return_yrecv:
            # y_recv [L, n_s] complex; Psym scalar. Both feed the front-end
            # energy estimator; codewords carry the true load for calibration.
            return Y, codewords, y_recv, torch.tensor(float(encoder.Psym))

        return Y, codewords

"""
Torch/GPU port of FactorizedBPDecoder.decode_batch (SIC-BP and FFT-BP).

This is a FAITHFUL, NAIVE port: it mirrors the NumPy batched path in
rules/sic_bp.py one-for-one, including the Python loops over N variable
nodes and M check nodes. Only the per-message tensor ops (WHT, normalize,
GF permute, explain-away, syndrome) move to torch, so they can run on GPU.

"Naive" = the Tanner graph is still traversed with Python loops, so each
BP iteration issues O(N + M) small kernel launches. Whether that is faster
than CPU NumPy is exactly the empirical question; the measured wall-clock
(launch overhead included, since it is inside the timed region) is the answer.

Correctness is validated against the NumPy reference by comparing decoded
argmax grids on identical inputs (see __main__).

Self-test: python -m rules.sic_bp_torch --device cpu --n 32
"""
from __future__ import annotations

import numpy as np
import torch

from rules.sic_bp import (FactorizedBPDecoder, DEFAULT_MAX_ITERS, DEFAULT_DAMPING,
                          DEFAULT_EXPLAIN_STRENGTH)


def _wht_torch(f: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard transform along last dim; mirrors sic_bp._walsh_hadamard_transform."""
    a = f.clone()
    Q = a.shape[-1]
    lead = a.shape[:-1]
    h = 1
    while h < Q:
        a = a.reshape(*lead, Q // (2 * h), 2, h)
        x = a[..., 0, :].clone()
        y = a[..., 1, :].clone()
        a[..., 0, :] = x + y
        a[..., 1, :] = x - y
        a = a.reshape(*lead, Q)
        h *= 2
    return a


class TorchBPDecoder:
    """GPU/CPU torch BP decoder wrapping a NumPy FactorizedBPDecoder's tables."""

    def __init__(self, np_decoder: FactorizedBPDecoder,
                 device: str = 'cuda', dtype: torch.dtype = torch.float32):
        d = self.dev = torch.device(device)
        self.dtype = dtype
        self.Q = np_decoder.Q
        self.N = np_decoder.N
        self.M = np_decoder.M
        self.K = np_decoder.K
        self.damping = float(np_decoder.damping)
        self.explain_strength = float(np_decoder.explain_strength)
        self.eps = float(np_decoder.eps)
        self.max_iters = int(np_decoder.max_iters)
        # Was previously dropped: every GPU run silently used the WHT path
        # regardless of the variant requested, so "SIC-BP" and "FFT-BP"
        # measured the same code.
        self.use_wht = bool(getattr(np_decoder, 'use_wht', True))
        # z1 ^ z lookup for the direct (non-WHT) XOR convolution.
        _ar = torch.arange(self.Q, device=d)
        self._xor_idx = (_ar.unsqueeze(1) ^ _ar.unsqueeze(0)).long()  # [Q,Q]

        # GF tables -> device
        self.perm = torch.as_tensor(np_decoder._perm_table, dtype=torch.long, device=d)  # [Q,Q]
        self.inv = torch.as_tensor(np_decoder._inv_table, dtype=torch.long, device=d)    # [Q]
        self.mul = torch.as_tensor(np_decoder._mul_table, dtype=torch.long, device=d)    # [Q,Q]

        # Graph adjacency (kept as Python lists -> the "naive" loop structure)
        g = np_decoder.graph
        self.var_to_checks = [list(x) for x in g.var_to_checks]
        self.check_to_vars = [list(x) for x in g.check_to_vars]
        self.check_coeffs = [list(x) for x in g.check_coeffs]

    # ---- helpers -------------------------------------------------------
    def _norm_prob(self, p: torch.Tensor) -> torch.Tensor:
        p = p.clamp_min(0.0)
        s = p.sum(-1, keepdim=True)
        s = torch.where(s < self.eps, torch.ones_like(s), s)
        return p / s

    def _norm_log(self, logp: torch.Tensor) -> torch.Tensor:
        m = logp.max(-1, keepdim=True).values
        p = torch.exp(logp - m)
        s = p.sum(-1, keepdim=True)
        s = torch.where(s < self.eps, torch.ones_like(s), s)
        return p / s

    def _bp_iteration(self, channel_prob, v2c, c2v):
        """One BP iteration, batched over B. Mirrors _bp_iteration_batch."""
        Q, N, M = self.Q, self.N, self.M
        damp, eps = self.damping, self.eps
        v2c_new = torch.zeros_like(v2c)

        # variable -> check
        for n in range(N):
            checks = self.var_to_checks[n]
            prod = channel_prob[:, n, :].clone()  # [B,Q]
            for m in checks:
                prod = prod * c2v[:, m, n, :].clamp_min(eps)
            prod = self._norm_prob(prod)
            for m in checks:
                msg = prod / c2v[:, m, n, :].clamp_min(eps)
                msg = self._norm_prob(msg)
                v2c_new[:, n, m, :] = (1 - damp) * v2c[:, n, m, :] + damp * msg

        # check -> variable
        #   use_wht=True  -> FFT-BP: XOR convolution via WHT,  O(Q log Q)
        #   use_wht=False -> SIC-BP: direct XOR convolution,   O(Q^2)
        # Both compute the same extrinsic messages (see paper App. I); they
        # differ only in how the check-to-variable convolution is evaluated,
        # which is exactly the runtime distinction the manuscript draws.
        c2v_new = torch.zeros_like(c2v)
        for m in range(M):
            vars_in_check = self.check_to_vars[m]
            coeffs = self.check_coeffs[m]
            d_c = len(vars_in_check)
            gathered = torch.stack(
                [v2c_new[:, v, m, :] for v in vars_in_check], dim=1)  # [B,d_c,Q]
            transformed = torch.empty_like(gathered)
            for i, h in enumerate(coeffs):
                h_inv = int(self.inv[h].item())
                transformed[:, i, :] = gathered[:, i, :][:, self.perm[h_inv]]

            if self.use_wht:
                wht_all = _wht_torch(transformed)          # [B,d_c,Q]
                wht_prod = wht_all.prod(dim=1)             # [B,Q]
                for i, (v, h) in enumerate(zip(vars_in_check, coeffs)):
                    wht_ext = wht_prod / (wht_all[:, i, :] + 1e-30)
                    conv = _wht_torch(wht_ext) / Q
                    conv = self._norm_prob(conv.clamp_min(0))
                    out = conv[:, self.perm[h]]
                    c2v_new[:, m, v, :] = (1 - damp) * c2v[:, m, v, :] + damp * out
            else:
                # Direct XOR convolution, mirroring the numpy branch in
                # sic_bp.py:622-637 including its per-step renormalization.
                #   r[z] = sum_{z1} p[z1] * q[z1 ^ z]
                # self._xor_idx[z1, z] = z1 ^ z, so q[:, _xor_idx] is
                # [B,Q,Q] with entry [b,z1,z] = q[b, z1^z].
                for i, (v, h) in enumerate(zip(vars_in_check, coeffs)):
                    others = [transformed[:, j, :]
                              for j in range(d_c) if j != i]
                    if not others:
                        conv = torch.full_like(transformed[:, 0, :], 1.0 / Q)
                    else:
                        conv = others[0]
                        for p_z in others[1:]:
                            conv = torch.einsum(
                                'bi,biz->bz', conv, p_z[:, self._xor_idx])
                            conv = self._norm_prob(conv)
                    conv = self._norm_prob(conv.clamp_min(0))
                    out = conv[:, self.perm[h]]
                    c2v_new[:, m, v, :] = (1 - damp) * c2v[:, m, v, :] + damp * out

        # posteriors
        posteriors = channel_prob.clone()
        for n in range(N):
            for m in self.var_to_checks[n]:
                posteriors[:, n, :] = posteriors[:, n, :] * c2v_new[:, m, n, :].clamp_min(eps)
            posteriors[:, n, :] = self._norm_prob(posteriors[:, n, :])
        return v2c_new, c2v_new, posteriors

    def _syndromes_ok(self, hard: torch.Tensor) -> torch.Tensor:
        """hard [B,K,N] long -> bool [B]: every row satisfies every check."""
        B, K, N = hard.shape
        ok = torch.ones(B, K, dtype=torch.bool, device=hard.device)
        for m in range(self.M):
            synd = torch.zeros(B, K, dtype=torch.long, device=hard.device)
            for n, h in zip(self.check_to_vars[m], self.check_coeffs[m]):
                synd = synd ^ self.mul[h][hard[:, :, n]]   # GF(2^m) add == XOR
            ok = ok & (synd == 0)
        return ok.all(dim=1)

    def _unique_rows(self, hard: torch.Tensor) -> torch.Tensor:
        """hard [B,K,N] -> bool [B]: all K rows distinct."""
        B, K, N = hard.shape
        u = torch.ones(B, dtype=torch.bool, device=hard.device)
        for i in range(K):
            for j in range(i + 1, K):
                same = (hard[:, i, :] == hard[:, j, :]).all(dim=1)
                u = u & ~same
        return u

    # ---- main ----------------------------------------------------------
    @torch.no_grad()
    def decode_batch(self, Y_batch, early_exit: bool = True) -> torch.Tensor:
        """Y_batch: [B,N,Q] (np or torch) -> codewords [B,K,N] long (on device)."""
        if isinstance(Y_batch, np.ndarray):
            Y = torch.as_tensor(Y_batch, dtype=self.dtype, device=self.dev)
        else:
            Y = Y_batch.to(self.dev, self.dtype)
        B, N, Q = Y.shape
        K, M, eps = self.K, self.M, self.eps

        Y_prob = self._norm_log(Y)                                  # [B,N,Q]
        channel = Y.unsqueeze(1).repeat(1, K, 1, 1)                 # [B,K,N,Q]
        v2c = torch.full((B, K, N, M, Q), 1.0 / Q, dtype=self.dtype, device=self.dev)
        c2v = torch.full((B, K, M, N, Q), 1.0 / Q, dtype=self.dtype, device=self.dev)
        posteriors = Y_prob.unsqueeze(1).repeat(1, K, 1, 1)         # [B,K,N,Q]

        done = torch.zeros(B, dtype=torch.bool, device=self.dev) if early_exit else None
        final_post = posteriors.clone() if early_exit else None

        for _ in range(self.max_iters):
            for k in range(K):
                ch_k = self._norm_log(channel[:, k])
                v2c[:, k], c2v[:, k], posteriors[:, k] = self._bp_iteration(
                    ch_k, v2c[:, k], c2v[:, k])
                for k2 in range(K):
                    if k2 == k:
                        continue
                    new_ch = Y_prob.clone()
                    explain = 1.0 - self.explain_strength * posteriors[:, k] + self.explain_strength / Q
                    new_ch = new_ch * explain
                    for k3 in range(k):
                        if k3 != k2:
                            e3 = 1.0 - self.explain_strength * posteriors[:, k3] + self.explain_strength / Q
                            new_ch = new_ch * e3
                    new_ch = self._norm_prob(new_ch)
                    channel[:, k2] = torch.log(new_ch + eps)

            if early_exit:
                hard = posteriors.argmax(dim=-1)                    # [B,K,N]
                conv = self._syndromes_ok(hard) & self._unique_rows(hard)
                newly = conv & ~done
                if newly.any():
                    final_post[newly] = posteriors[newly]
                    done = done | newly
                if bool(done.all()):
                    break

        if early_exit:
            mask = done.view(B, 1, 1, 1)
            posteriors = torch.where(mask, final_post, posteriors)
        return posteriors.argmax(dim=-1)


if __name__ == '__main__':
    # Correctness gate: torch (cpu, float64) vs NumPy reference.
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.expanduser('~/data/demix/tiny_LDPC'))
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--max_iters', type=int, default=DEFAULT_MAX_ITERS)
    ap.add_argument('--use_wht', action='store_true', default=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    data = torch.load(os.path.join(args.data_dir, 'test_data.pt'), weights_only=True)
    hd = torch.load(os.path.join(args.data_dir, 'H_matrix.pt'), weights_only=True)
    Y = data['Y'][:args.n].numpy()
    H = hd['H_matrix'].numpy()
    K = data['gt_codewords'].shape[1]
    Q, N, M = Y.shape[-1], Y.shape[1], H.shape[0]

    npdec = FactorizedBPDecoder(Q=Q, N=N, K=K, M=M, H=H, max_iters=args.max_iters,
                                damping=DEFAULT_DAMPING,
                                explain_strength=DEFAULT_EXPLAIN_STRENGTH, use_wht=True)
    ref = npdec.decode_batch(Y)                                     # [n,K,N]

    for dev in ['cpu', args.device]:
        if dev == 'cuda' and not torch.cuda.is_available():
            continue
        dt = torch.float64 if dev == 'cpu' else torch.float32
        tdec = TorchBPDecoder(npdec, device=dev, dtype=dt)
        out = tdec.decode_batch(Y).cpu().numpy()
        # Hungarian-free: compare as sets per sample (row order may differ)
        agree = 0
        for b in range(args.n):
            r = {tuple(ref[b, k]) for k in range(K)}
            o = {tuple(out[b, k]) for k in range(K)}
            agree += (r == o)
        print(f'[{dev:4s} {str(dt):15s}] exact set-match: {agree}/{args.n} '
              f'({100*agree/args.n:.1f}%)')

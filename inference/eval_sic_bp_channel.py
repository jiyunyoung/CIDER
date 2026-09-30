#!/usr/bin/env python3
"""
Rule-based BP decoders (SIC-BP / FFT-BP) under the SAME mismatched channels as
CIDER. No retraining applies -- BP is rule-based -- but it consumes the same
soft evidence Y the AMP inner detector produces, so feeding it the on-the-fly
fading Y makes the CIDER-vs-SIC-BP comparison apples-to-apples per channel.

Produces the SIC-BP row of the main-text tab:fading_stress (uncompensated
fading with unknown per-user gains, K=2, L=12, Eb/N0 = 10 dB). The CIDER row
comes from inference/eval_snr_sweep.py --fading ... on the same test stream
(fixed_seed=199999, disjoint from the training-data seed 42; sensing A seed 42).

Decoder: rules.sic_bp_torch.TorchBPDecoder (batched; matches the NumPy
reference on 100% of samples), wrapping a rules.sic_bp.FactorizedBPDecoder
that holds the GF tables and Tanner graph.

Conditions: AWGN, Rician (per-user uniform LoS phase + slot-wise diffuse part,
E|h|^2 = 1) at each --kappas value, and Rayleigh.

tab:fading_stress SIC-BP row:
    python inference/eval_sic_bp_channel.py --K 2 --eb 10 --variant sic \
        --num-samples 5000 --batch-size 256

Usage:
    python inference/eval_sic_bp_channel.py                     # FFT-BP, defaults
    python inference/eval_sic_bp_channel.py --variant sic --kappas 10 0 -10
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_onthefly import QaryOnTheFlyDataset
from rules.sic_bp import (FactorizedBPDecoder, DEFAULT_MAX_ITERS,
                          DEFAULT_DAMPING, DEFAULT_EXPLAIN_STRENGTH)
from rules.sic_bp_torch import TorchBPDecoder
from models.cider_iterative_v2 import _assign, _hamming_cost_matrix


def cer_from_preds(preds, gt):
    """PUPE / CER for hard predictions [B,K,N] vs gt [B,K,N] (Hungarian-matched)."""
    preds = preds.long(); gt = gt.long()
    col = _assign(_hamming_cost_matrix(preds, gt))
    B, K, N = preds.shape
    gt_perm = torch.gather(gt, 1, col.unsqueeze(-1).expand(B, K, N))
    matches = (preds == gt_perm)
    n_row_err = (~matches.all(dim=-1)).sum().item()
    n_sym_err = (~matches).sum().item()
    return n_row_err, B * K, n_sym_err, matches.numel()


def build_conditions(kappas):
    conds = [('none', None, 'AWGN')]
    for k in kappas:
        conds.append(('rician', float(k), f'Rician{k:+g}dB'))
    conds.append(('rayleigh', None, 'Rayleigh'))
    return conds


def gen_YX(h_path, K, eb, n_s, sigma2, fading, kdb, num_samples, batch_size):
    """On-the-fly fading data -> (Y [S,N,Q], X0 [S,K,N]) tensors on CPU."""
    kw = {}
    if fading != 'none':
        kw['fading'] = fading
        kw['fading_coherence'] = 'slot'
        if fading == 'rician':
            kw['rician_k_dB'] = kdb
    ds = QaryOnTheFlyDataset(h_path, K=K, Eb_dB=eb, n_s=n_s, sigma2=sigma2,
                             num_samples=num_samples, fixed_seed=199999, **kw)
    Ys, Xs = [], []
    for Y, X in DataLoader(ds, batch_size=batch_size, shuffle=False):
        Ys.append(Y); Xs.append(X)
    return torch.cat(Ys), torch.cat(Xs), ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, default=2)
    p.add_argument('--eb', type=float, default=10.0)
    p.add_argument('--n-s', type=int, default=24)
    p.add_argument('--sigma2', type=float, default=1.0)
    p.add_argument('--max-iters', type=int, default=DEFAULT_MAX_ITERS,
                   help='BP iterations')
    p.add_argument('--variant', choices=['fft', 'sic'], default='fft',
                   help="fft = use_wht True (FFT-BP, O(Q log Q), fast on CPU); "
                        "sic = use_wht False (SIC-BP, direct O(Q^2)). Same CER, "
                        "differ only in check-node convolution speed.")
    p.add_argument('--damping', type=float, default=DEFAULT_DAMPING)
    p.add_argument('--explain-strength', type=float,
                   default=DEFAULT_EXPLAIN_STRENGTH)
    p.add_argument('--kappas', type=float, nargs='+', default=[20, 10, 5, 0, -5, -10])
    p.add_argument('--num-samples', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--h-matrix',
                   default=os.path.expanduser('~/data/demix/tiny_LDPC/H_matrix.pt'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--csv', default=None,
                   help='Output CSV (default: '
                        'logs/{variant}_bp_channel_K{K}_it{max_iters}.csv).')
    args = p.parse_args()
    args.h_matrix = os.path.expanduser(args.h_matrix)

    # Build the decoder once (geometry is channel-independent). H from the H file.
    h = torch.load(args.h_matrix, weights_only=False)
    H = h['H_matrix']
    Hnp = H.numpy() if isinstance(H, torch.Tensor) else np.asarray(H)
    Q, N, M = int(h['q']), int(h['L']), int(h['M'])
    use_wht = (args.variant == 'fft')
    bp_name = 'FFT-BP' if use_wht else 'SIC-BP'
    npdec = FactorizedBPDecoder(Q=Q, N=N, K=args.K, M=M, H=Hnp,
                                max_iters=args.max_iters, damping=args.damping,
                                explain_strength=args.explain_strength,
                                use_wht=use_wht)
    tdec = TorchBPDecoder(npdec, device=args.device)
    print(f"device={args.device}  K={args.K}  Eb/N0={args.eb} dB  "
          f"{bp_name} (use_wht={use_wht}) max_iters={args.max_iters} "
          f"damping={args.damping} explain={args.explain_strength}  "
          f"num_samples={args.num_samples}")

    print(f"\n{'channel':<14}{'kappa_dB':>9}{'CER':>10}{'SER':>10}{'sec':>8}")
    print("-" * 52)
    rows = []
    for fading, kdb, label in build_conditions(args.kappas):
        Y, X, _ = gen_YX(args.h_matrix, args.K, args.eb, args.n_s, args.sigma2,
                         fading, kdb, args.num_samples, args.batch_size)
        S = Y.shape[0]
        re = r = se = s = 0
        t0 = time.time()
        for lo in range(0, S, args.batch_size):
            Yb = Y[lo:lo + args.batch_size]
            preds = tdec.decode_batch(Yb).cpu()          # [b,K,N] long
            a, b, c, d = cer_from_preds(preds, X[lo:lo + args.batch_size])
            re += a; r += b; se += c; s += d
        dt = time.time() - t0
        cer, ser = re / max(1, r), se / max(1, s)
        kd = '' if kdb is None else f'{kdb:+g}'
        print(f"{label:<14}{kd:>9}{cer:>10.4f}{ser:>10.4f}{dt:>8.1f}")
        rows.append((label, kd, cer, ser))

    # CSV
    out = args.csv or f"logs/{args.variant}_bp_channel_K{args.K}_it{args.max_iters}.csv"
    out = os.path.expanduser(out)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        f.write("channel,kappa_dB,CER,SER\n")
        for lbl, kd, cer, ser in rows:
            f.write(f"{lbl},{kd},{cer:.4f},{ser:.4f}\n")
    print(f"\nsaved {out}")


if __name__ == '__main__':
    main()

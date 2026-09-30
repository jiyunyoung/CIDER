#!/usr/bin/env python3
"""
Generate per-K test sets that also store raw received slots y_recv, for the
front-end active-user-count (K) estimator experiment (Table
tab:estimated_load, via inference/eval_k_estimator.py).

For each K in --K_list this writes  <out>/K<K>/kest_data.pt  holding:
    Y            [n, L, Q]      AMP evidence (== the paper's S interface)
    gt_codewords [n, K, L]      true codewords (true load K per sample)
    y_recv       [n, L, n_s]    raw received slots (complex) -- estimator input
    Psym         scalar         symbol power for this K/Eb
    meta         dict           K, Eb_dB, n_s, sigma2, L, Q

The estimator (inference/eval_k_estimator.py) consumes y_recv only; Y and
gt_codewords let the same file drive the composed end-to-end pipeline.

Seeding: frame i of load K is drawn with per-sample seed
seed + 1000*K + i (QaryOnTheFlyDataset fixed_seed), far from the seed-42
training stream. The sensing matrix A stays at seed 42 (set inside
QaryOnTheFlyDataset). Note that for n > 1000 the per-sample seed ranges of
neighbouring K overlap (K's frame i+1000 and (K+1)'s frame i share a seed);
this is kept as-is so the published sets are reproduced bit-for-bit.

Usage (the sets behind tab:estimated_load, 3000 frames per K):
    python data/gen_data/gen_kest_data.py \
        --h_matrix ~/data/demix/tiny_LDPC/H_matrix.pt \
        --out data/gen_data/datasets/kest_Eb10 \
        --K_list 2 3 4 5 6 7 8 --Eb 10 --n 3000
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.data_onthefly import QaryOnTheFlyDataset


def build_k(h_matrix, K, Eb, n_s, sigma2, n, seed, fading, rician_k_dB):
    kw = {}
    if fading != 'none':
        kw['fading'] = fading
        if fading == 'rician':
            kw['rician_k_dB'] = rician_k_dB
    ds = QaryOnTheFlyDataset(
        h_matrix, K=K, Eb_dB=Eb, n_s=n_s, sigma2=sigma2,
        num_samples=n, fixed_seed=seed + 1000 * K,
        return_yrecv=True, **kw)

    Ys, X0s, Ys_recv = [], [], []
    Psym = None
    for i in range(n):
        Y, cw, y_recv, psym = ds[i]
        Ys.append(Y)
        X0s.append(cw)
        Ys_recv.append(y_recv)
        Psym = float(psym)
    return (torch.stack(Ys), torch.stack(X0s), torch.stack(Ys_recv),
            Psym, dict(K=K, Eb_dB=Eb, n_s=n_s, sigma2=sigma2,
                       L=ds.L, Q=ds.Q))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h_matrix', default='~/data/demix/tiny_LDPC/H_matrix.pt')
    p.add_argument('--out', default='data/gen_data/datasets/kest_Eb10')
    p.add_argument('--K_list', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument('--Eb', type=float, default=10.0)
    p.add_argument('--n_s', type=int, default=24)
    p.add_argument('--sigma2', type=float, default=1.0)
    p.add_argument('--n', type=int, default=3000, help='samples per K')
    p.add_argument('--seed', type=int, default=770077,
                   help='Base per-sample seed (frame i of load K uses '
                        'seed + 1000*K + i). Keep it away from the training '
                        'seed 42.')
    p.add_argument('--fading', default='none',
                   choices=['none', 'rayleigh', 'rician'])
    p.add_argument('--rician_k_dB', type=float, default=None)
    args = p.parse_args()

    h_matrix = os.path.expanduser(args.h_matrix)
    out = Path(os.path.expanduser(args.out))
    print(f"Generating K-estimator test sets -> {out}")
    print(f"  K_list={args.K_list}  Eb={args.Eb}  n/K={args.n}  fading={args.fading}")

    # Also print the energy-vs-K trend as a built-in sanity check: excess
    # per-slot energy should be ~linear in K with slope ~= Psym.
    trend = []
    for K in args.K_list:
        Y, X0, y_recv, Psym, meta = build_k(
            h_matrix, K, args.Eb, args.n_s, args.sigma2, args.n,
            args.seed, args.fading, args.rician_k_dB)

        k_dir = out / f"K{K}"
        k_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dict(Y=Y, gt_codewords=X0, y_recv=y_recv,
                        Psym=Psym, meta=meta), k_dir / "kest_data.pt")

        # excess energy per slot, averaged over slots and samples
        e_slot = (y_recv.abs() ** 2).sum(-1)            # [n, L]
        excess = e_slot.mean().item() - args.n_s * args.sigma2
        trend.append((K, excess, Psym))
        print(f"  K={K}: saved {tuple(Y.shape)}  Psym={Psym:.3f}  "
              f"mean excess energy/slot={excess:8.3f}  (excess/K={excess/K:7.3f})")

    print("\nEnergy-vs-K check (excess/K should be ~constant ~= Psym):")
    for K, ex, ps in trend:
        print(f"  K={K}  excess/K={ex/K:8.3f}  Psym={ps:.3f}")


if __name__ == '__main__':
    main()

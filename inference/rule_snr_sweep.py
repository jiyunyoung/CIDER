"""
SNR sweep evaluator for rule-based decoders (sic_bp).

Usage:
    python inference/rule_snr_sweep.py
    python inference/rule_snr_sweep.py --K 2 3 4 5 --snr -4 -3 -2 -1 0 1 2 3 4
    python inference/rule_snr_sweep.py --max-iters 100 --explain-strength 1.0

SNR convention: per-user per-complex-channel-use SNR.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rules.sic_bp import FactorizedBPDecoder
from data.data_onthefly import QaryOnTheFlyDataset
from inference.eval_rules import hungarian_match


def make_loader(h_path, K, Eb_dB, n_s, sigma2, num_samples, batch_size,
                num_workers, seed=199999):
    ds = QaryOnTheFlyDataset(
        h_path, K=K, Eb_dB=Eb_dB, n_s=n_s, sigma2=sigma2,
        num_samples=num_samples, fixed_seed=seed, Eb_range=None,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
        persistent_workers=(num_workers > 0),
    )


def evaluate(decoder, loader, K, N, per_sample=True):
    """Run decoder on loader, return (SER, CER).
    """
    correct_sym = total_sym = correct_cw = total_cw = 0
    for Y_batch, X_batch in loader:
        Y_np = Y_batch.numpy()           # [B, N, Q]
        X_np = X_batch.numpy()           # [B, K, N]
        B = Y_np.shape[0]

        if per_sample:
            for b in range(B):
                preds_b, _ = decoder.decode(Y_np[b])  # [K, N], early-exit
                _, sym_err, per_cw_err = hungarian_match(preds_b[:K], X_np[b, :K])
                correct_sym += K * N - int(sym_err)
                total_sym  += K * N
                for err in per_cw_err:
                    if err == 0:
                        correct_cw += 1
                    total_cw += 1
        else:
            preds = decoder.decode_batch(Y_np)  # [B, K, N]
            for b in range(B):
                _, sym_err, per_cw_err = hungarian_match(preds[b, :K], X_np[b, :K])
                correct_sym += K * N - int(sym_err)
                total_sym  += K * N
                for err in per_cw_err:
                    if err == 0:
                        correct_cw += 1
                    total_cw += 1

    ser = 1.0 - correct_sym / max(1, total_sym)
    cer = 1.0 - correct_cw / max(1, total_cw)
    return ser, cer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, nargs='+', default=[2, 3, 4, 5])
    p.add_argument('--snr', type=float, nargs='+',
                   default=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
    p.add_argument('--num-samples', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--h-matrix',
                   default=os.path.expanduser('~/data/demix/tiny_LDPC/H_matrix.pt'))
    p.add_argument('--n-s', type=int, default=24)
    p.add_argument('--sigma2', type=float, default=1.0)
    p.add_argument('--offset', type=float, default=10.79,
                   help='Eb/N0 = SNR + offset (dB). 10.79 = tiny_ldpc rate.')
    p.add_argument('--csv', default='logs/rule_snr_sweep_results.csv')
    p.add_argument('--seed', type=int, default=199999,
                   help='Fixed seed for on-the-fly test data')
    # SIC-BP hyperparameters
    p.add_argument('--max-iters', type=int, default=50)
    p.add_argument('--damping', type=float, default=0.1)
    p.add_argument('--explain-strength', type=float, default=1.0)
    p.add_argument('--batched', action='store_true',
                   help='Use decode_batch() instead of per-sample decode(). '
                        'Faster but slightly different float ordering.')
    args = p.parse_args()

    print(f"SNR range: {args.snr}")
    print(f"K range:   {args.K}")
    print(f"num_samples: {args.num_samples}, batch_size: {args.batch_size}")
    print(f"Eb/N0 offset: +{args.offset} dB")
    print(f"sic_bp: max_iters={args.max_iters}, damping={args.damping}, "
          f"explain_strength={args.explain_strength}")

    h_data = torch.load(args.h_matrix, weights_only=False)
    H = h_data['H_matrix'].numpy() if isinstance(h_data['H_matrix'], torch.Tensor) \
        else np.asarray(h_data['H_matrix'])
    Q = int(h_data['q'])
    N = int(h_data['L'])
    M = int(h_data['M'])

    results = []  # (K, SNR, Eb, SER, CER, ms_per_sample)
    for K in args.K:
        print(f"\n{'='*64}\nK={K}\n{'='*64}")
        decoder = FactorizedBPDecoder(
            Q=Q, N=N, K=K, M=M, H=H,
            max_iters=args.max_iters,
            damping=args.damping,
            explain_strength=args.explain_strength,
        )

        for snr in args.snr:
            eb = round(snr + args.offset, 4)
            t0 = time.perf_counter()
            loader = make_loader(args.h_matrix, K, eb, args.n_s, args.sigma2,
                                 args.num_samples, args.batch_size,
                                 args.num_workers, seed=args.seed)
            ser, cer = evaluate(decoder, loader, K, N,
                                per_sample=not args.batched)
            dt = time.perf_counter() - t0
            ms_per = dt * 1000 / args.num_samples
            print(f"  SNR={snr:+5.1f}  Eb/N0={eb:6.2f}  "
                  f"SER={ser:.4f}  CER={cer:.4f}  ({ms_per:7.2f} ms/sample)")
            results.append((K, snr, eb, ser, cer, ms_per))
            del loader

    # Combined table
    print(f"\n{'='*64}\nCombined Summary  (sic_bp)\n{'='*64}")
    print(f"{'K':<4} {'SNR':>6} {'Eb/N0':>8} {'SER':>10} {'CER':>10} {'ms/samp':>10}")
    print("-" * 64)
    for K, snr, eb, ser, cer, ms in results:
        print(f"{K:<4} {snr:>+6.1f} {eb:>8.2f} {ser:>10.4f} {cer:>10.4f} {ms:>10.2f}")

    # CSV
    os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
    with open(args.csv, 'w') as f:
        f.write("K,SNR,EbN0,SER,CER,ms_per_sample\n")
        for r in results:
            f.write(','.join(str(x) for x in r) + '\n')
    print(f"\nSaved CSV: {args.csv}")


if __name__ == '__main__':
    main()

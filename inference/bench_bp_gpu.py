#!/usr/bin/env python3
"""
Usage:
    python inference/bench_bp_gpu.py --scales tiny small moderate large --variants sic fft
"""
import argparse
import os
import sys
import time
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.eval_rules import hungarian_match  # noqa: E402
from rules.sic_bp import (FactorizedBPDecoder, DEFAULT_MAX_ITERS,  # noqa: E402
                          DEFAULT_DAMPING, DEFAULT_EXPLAIN_STRENGTH)
from rules.sic_bp_torch import TorchBPDecoder  # noqa: E402

# scale -> (data subdir under --data_root, L)
SCALE_CFG = {
    'tiny':     ('tiny_LDPC',     12),
    'small':    ('small_LDPC',    18),
    'moderate': ('moderate_LDPC', 24),
    'large':    ('large_LDPC',    48),
    'peg':      ('tiny_LDPC_PEG', 12),
    'tree':     ('tiny_tree',     12),
}


def load(data_dir: str, num_samples: int):
    d = torch.load(os.path.join(data_dir, 'test_data.pt'), weights_only=True)
    h = torch.load(os.path.join(data_dir, 'H_matrix.pt'), weights_only=True)
    return (d['Y'][:num_samples].numpy(),
            d['gt_codewords'][:num_samples].numpy(),
            h['H_matrix'].numpy())


def score(preds, X0, K, N):
    cs = tc = ccw = tcw = 0
    for b in range(preds.shape[0]):
        _, se, pce = hungarian_match(preds[b, :K], X0[b, :K])
        cs += K * N - int(se); tc += K * N
        for e in pce:
            ccw += (e == 0); tcw += 1
    return 1 - cs / tc, 1 - ccw / tcw


def time_cpu(npdec, Y, bs):
    B = Y.shape[0]
    start = time.perf_counter()
    outs = [npdec.decode_batch(Y[lo:lo + bs]) for lo in range(0, B, bs)]
    el = time.perf_counter() - start
    return np.concatenate(outs, 0), 1000 * el / B


def time_gpu(tdec, Y, bs, repeats=3):
    B = Y.shape[0]
    # warmup
    _ = tdec.decode_batch(Y[:min(bs, B)])
    torch.cuda.synchronize()
    times: List[float] = []
    preds = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        chunks = [tdec.decode_batch(Y[lo:lo + bs]) for lo in range(0, B, bs)]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        preds = chunks
    out = torch.cat(preds, 0).cpu().numpy()
    return out, 1000 * float(np.mean(times)) / B


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scales', nargs='+', default=['tiny', 'small', 'moderate', 'large'],
                   choices=list(SCALE_CFG))
    p.add_argument('--variants', nargs='+', default=['sic', 'fft'],
                   choices=['sic', 'fft'])
    p.add_argument('--batch_size', '--bs', type=int, default=8)
    p.add_argument('--num_samples', '--n', type=int, default=512)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--data_root', default=os.path.expanduser('~/data/demix'),
                   help='Directory holding {tiny,small,moderate,large}_LDPC/test_data.pt')
    p.add_argument('--max_iters', type=int, nargs='+', default=[DEFAULT_MAX_ITERS],
                   help='One value for all scales, or one per --scales entry')
    p.add_argument('--damping', type=float, default=DEFAULT_DAMPING,
                   help='Weight on the new message (1 = undamped)')
    p.add_argument('--explain_strength', type=float, default=DEFAULT_EXPLAIN_STRENGTH)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                   help="'cuda' times the torch decoder on the GPU; 'cpu' times the NumPy decoder")
    p.add_argument('--dtype', default='float32',
                   choices=['float32', 'float16', 'bfloat16'],
                   help="GPU dtype. float16 UNDERFLOWS BP (garbage decodes); "
                        "bfloat16 is the accuracy-preserving 16-bit analog of "
                        "CIDER's 16-mixed (fp32 accumulation).")
    args = p.parse_args()
    gpu_dtype = getattr(torch, args.dtype)
    if len(args.max_iters) == 1:
        iters_of = {s: args.max_iters[0] for s in args.scales}
    elif len(args.max_iters) == len(args.scales):
        iters_of = dict(zip(args.scales, args.max_iters))
    else:
        p.error('--max_iters takes one value or one per --scales entry')

    print('=' * 78)
    print('Belief Propagation')
    print('=' * 78)

    for scale in args.scales:
        subdir, L = SCALE_CFG[scale]
        iters = iters_of[scale]
        data_dir = os.path.join(args.data_root, subdir)
        if not os.path.exists(os.path.join(data_dir, 'test_data.pt')):
            print(f'[skip] {scale}: no data at {data_dir}')
            continue
        Y, X0, H = load(data_dir, args.num_samples)
        K, N, Q, M = X0.shape[1], Y.shape[1], Y.shape[-1], H.shape[0]
        print(f'{scale} (L={N}, K={K})')

        for var in args.variants:
            use_wht = (var == 'fft')
            label = 'FFT-BP' if use_wht else 'SIC-BP'
            npdec = FactorizedBPDecoder(Q=Q, N=N, K=K, M=M, H=H, max_iters=iters,
                                        damping=args.damping,
                                        explain_strength=args.explain_strength,
                                        use_wht=use_wht)
            if args.device == 'cuda':
                tdec = TorchBPDecoder(npdec, device='cuda', dtype=gpu_dtype)
                preds, ms = time_gpu(tdec, Y, args.batch_size, args.repeats)
            else:
                preds, ms = time_cpu(npdec, Y, args.batch_size)
            ser, cer = score(preds, X0, K, N)
            print(f'  {label:<7} SER {ser:.4f}  CER {cer:.4f}  {ms:.2f} ms/sample')


if __name__ == '__main__':
    main()

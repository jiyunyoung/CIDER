#!/usr/bin/env python3
"""
GPU vs CPU timing for batched BP (SIC-BP and FFT-BP), decode-only.

Uses the validated torch port (rules/sic_bp_torch.TorchBPDecoder, 100% exact
set-match vs the NumPy reference) for the GPU path, and the NumPy decode_batch
for the CPU path. Early exit is ON for both, matching Table 1's convention.

Timed region = decode only (Hungarian matching runs after, like the BP baseline
in Table 1). CUDA work is bracketed with torch.cuda.synchronize(), so kernel-
launch overhead and the early-exit host syncs are correctly counted.

Usage:
    python inference/bench_bp_gpu.py --scales tiny small moderate large
    python inference/bench_bp_gpu.py --scales tiny --variants sic fft

Table 1 timing protocol (same GPU, batch size 8, 512 test samples):
    python inference/bench_bp_gpu.py --scales tiny small moderate large \
        --variants sic fft --batch_size 8 --num_samples 512 --repeats 1 --no_cpu
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
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_samples', type=int, default=256)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--data_root', default=os.path.expanduser('~/data/demix'),
                   help='Directory holding {tiny,small,moderate,large}_LDPC/test_data.pt')
    p.add_argument('--max_iters', type=int, nargs='+', default=[DEFAULT_MAX_ITERS],
                   help='One value for all scales, or one per --scales entry')
    p.add_argument('--damping', type=float, default=DEFAULT_DAMPING,
                   help='Weight on the new message (1 = undamped)')
    p.add_argument('--explain_strength', type=float, default=DEFAULT_EXPLAIN_STRENGTH)
    p.add_argument('--no_cpu', action='store_true', help='GPU only (skip CPU ref)')
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
    print('Batched BP: GPU (torch) vs CPU (numpy), decode-only, early-exit ON')
    print('=' * 78)
    print(f'host: {os.uname().nodename}   gpu: '
          f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')
    print(f'batch_size={args.batch_size}  num_samples={args.num_samples}  '
          f'repeats={args.repeats}  damping={args.damping}  '
          f'explain_strength={args.explain_strength}')
    print()

    rows = []
    for scale in args.scales:
        subdir, L = SCALE_CFG[scale]
        iters = iters_of[scale]
        data_dir = os.path.join(args.data_root, subdir)
        if not os.path.exists(os.path.join(data_dir, 'test_data.pt')):
            print(f'[skip] {scale}: no data at {data_dir}')
            continue
        Y, X0, H = load(data_dir, args.num_samples)
        K, N, Q, M = X0.shape[1], Y.shape[1], Y.shape[-1], H.shape[0]
        print('-' * 78)
        print(f'{scale}  (L={N}, K={K}, Q={Q}, M={M}, max_iters={iters})')
        print('-' * 78)

        for var in args.variants:
            use_wht = (var == 'fft')
            label = 'FFT-BP' if use_wht else 'SIC-BP'
            npdec = FactorizedBPDecoder(Q=Q, N=N, K=K, M=M, H=H, max_iters=iters,
                                        damping=args.damping,
                                        explain_strength=args.explain_strength,
                                        use_wht=use_wht)
            cpu_ms = gpu_ms = None
            cpu_cer = gpu_cer = None
            if not args.no_cpu:
                pc, cpu_ms = time_cpu(npdec, Y, args.batch_size)
                _, cpu_cer = score(pc, X0, K, N)
            if torch.cuda.is_available():
                tdec = TorchBPDecoder(npdec, device='cuda', dtype=gpu_dtype)
                pg, gpu_ms = time_gpu(tdec, Y, args.batch_size, args.repeats)
                _, gpu_cer = score(pg, X0, K, N)
            sp = (cpu_ms / gpu_ms) if (cpu_ms and gpu_ms) else None
            rows.append(dict(scale=scale, L=N, method=label,
                             cpu_ms=cpu_ms, gpu_ms=gpu_ms, sp=sp,
                             cpu_cer=cpu_cer, gpu_cer=gpu_cer))
            cs = f'{cpu_ms:8.2f}' if cpu_ms else '     -  '
            gs = f'{gpu_ms:8.2f}' if gpu_ms else '     -  '
            sps = f'{sp:6.2f}x' if sp else '   -  '
            print(f'  {label}:  CPU {cs} ms   GPU {gs} ms   GPU/CPU {sps}   '
                  f'CER cpu={cpu_cer} gpu={gpu_cer}')
        print()

    print('=' * 78)
    print(f'{"scale":<9} {"L":>3} {"method":<7} {"CPU ms":>9} {"GPU ms":>9} '
          f'{"GPU speedup":>12}')
    for r in rows:
        cs = f'{r["cpu_ms"]:.2f}' if r['cpu_ms'] else '-'
        gs = f'{r["gpu_ms"]:.2f}' if r['gpu_ms'] else '-'
        sp = f'{r["sp"]:.2f}x' if r['sp'] else '-'
        print(f'{r["scale"]:<9} {r["L"]:>3} {r["method"]:<7} {cs:>9} {gs:>9} {sp:>12}')


if __name__ == '__main__':
    main()

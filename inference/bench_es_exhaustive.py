#!/usr/bin/env python3
"""
Usage:
    python inference/bench_es_exhaustive.py 2 tiny
"""
import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))
from rules.top_j_beam import BeamTopJDecoder

# K==2 length scales -> (dataset dir, L)
SCALE_BY_L = {12: 'tiny_LDPC', 18: 'small_LDPC', 24: 'moderate_LDPC', 48: 'large_LDPC'}
# scale name -> L (the length arg accepts either a name or a raw number)
L_BY_NAME = {'tiny': 12, 'small': 18, 'moderate': 24, 'large': 48, 'peg': 12, 'tree': 12}
# scale name -> dataset dir under --data_root (peg/tree share L=12 with tiny)
DIR_BY_NAME = {'peg': 'tiny_LDPC_PEG', 'tree': 'tiny_tree'}


def parse_L(tok):
    if tok in L_BY_NAME:
        return L_BY_NAME[tok]
    try:
        return int(tok)
    except ValueError:
        sys.exit(f'L must be a length or one of {list(L_BY_NAME)} (got {tok!r})')


START_PASSES = 16


class _Timeout(Exception):
    pass


signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Timeout()))


def load(K, L, data_root, protocol_dir, subdir=None):
    if K == 2:
        if subdir is None and L not in SCALE_BY_L:
            sys.exit(f'no K=2 dataset for L={L} (have {sorted(SCALE_BY_L)})')
        base = os.path.join(data_root, subdir or SCALE_BY_L[L])
        d = torch.load(f'{base}/test_data.pt', weights_only=True)
        H = torch.load(f'{base}/H_matrix.pt', weights_only=True)['H_matrix'].numpy()
    else:
        if L != 12:
            sys.exit(f'K={K} data only exists at L=12 (protocol_Eb10)')
        base = Path(protocol_dir) / f'K{K}'
        d = torch.load(base / 'test_data.pt', weights_only=True)
        hd = torch.load(base / 'H_matrix.pt', weights_only=True)
        H = hd.get('H_matrix', hd.get('H')).numpy()
    return d['Y'].numpy(), d['gt_codewords'].numpy(), H


def ser_cer(preds, X0, K):
    """SER and CER after minimum-Hamming Hungarian row matching."""
    sym_err = sym_tot = cw = tot = 0
    for b in range(len(preds)):
        c = np.array([[(preds[b, i] != X0[b, j]).sum() for j in range(K)] for i in range(K)])
        r, co = linear_sum_assignment(c)
        for i, j in zip(r, co):
            sym_err += int(c[i, j]); sym_tot += X0.shape[-1]
            cw += int(c[i, j] == 0); tot += 1
    return sym_err / sym_tot, 1 - cw / tot


def main():
    p = argparse.ArgumentParser(description='GPU exhaustive Top-J search.')
    p.add_argument('K', type=int)
    p.add_argument('L', help='code length, or scale name: tiny/small/moderate/large/peg/tree')
    p.add_argument('--J', type=int, default=None, help='proposal width: top-J symbols per position (default: K)')
    p.add_argument('--n', type=int, default=512, help='samples for CER/latency')
    p.add_argument('--bs', type=int, default=8)
    p.add_argument('--dnf_s', type=int, default=60, help='per-sample DNF threshold (s)')
    p.add_argument('--data_root', default='~/data/demix',
                   help='Root of the K=2 datasets ({tiny,small,moderate,large}_LDPC)')
    p.add_argument('--protocol_dir', default='data/gen_data/datasets/protocol_Eb10',
                   help='K>=3 datasets (K<K>/test_data.pt), as in inference/eval_protocol.py')
    p.add_argument('--device', default='cuda',
                   help="Device for the search (default: cuda; 'cpu' for smoke tests)")
    args = p.parse_args()

    K, L = args.K, parse_L(args.L)
    J = args.J if args.J is not None else K
    if J < K:
        sys.exit(f'J={J} < K={K}: need at least K candidates per position')

    # start at START_PASSES rounded up to a power of J
    s = 0
    while J ** s < START_PASSES and s < L:
        s += 1

    dev = torch.device(args.device)

    def sync():
        if dev.type == 'cuda':
            torch.cuda.synchronize()

    Y, X0, H = load(K, L, os.path.expanduser(args.data_root),
                    os.path.expanduser(args.protocol_dir),
                    subdir=DIR_BY_NAME.get(args.L))
    n = min(args.n, len(Y))
    Yt = torch.as_tensor(Y[:n], dtype=torch.float32, device=dev)

    print('=' * 78)
    print('Exhaustive Top-J')
    print('=' * 78)
    name = f'{args.L} ' if args.L in L_BY_NAME else ''
    print(f'{name}(L={L}, K={K})')
    label = f'Top-{J}'

    # On OOM, retry with the next achievable split (passes x J) until it fits.
    while True:
        passes = J ** s
        beam = J ** (L - s)           # exhaustive per pass, never pruned

        es = None
        try:
            es = BeamTopJDecoder(64, L, K, H.shape[0], H, beam_width=beam,
                                 proposal_width=J, device=dev, seq_split=s)
            # DNF gate: warm up and time one sample under a dnf_s cap
            signal.alarm(args.dnf_s)
            es.decode_batch(Yt[:1]); sync()                  # warmup / OOM probe
            signal.alarm(args.dnf_s)
            t0 = time.perf_counter()
            es.decode_batch(Yt[:1]); sync()
            signal.alarm(0)
            probe_ms = (time.perf_counter() - t0) * 1000     # bs=1, upper-bound-ish

            # size n so the measurement takes about dnf_s
            n_meas = int(min(n, max(args.bs,
                                    (args.dnf_s * 1000) / probe_ms // args.bs * args.bs)))

            t0 = time.perf_counter()
            preds = np.concatenate([np.asarray(es.decode_batch(Yt[lo:lo + args.bs]))
                                    for lo in range(0, n_meas, args.bs)], 0)
            sync()
            ms = (time.perf_counter() - t0) * 1000 / n_meas
            ser, c = ser_cer(preds, X0[:n_meas], K)
            print(f'  {label:<7} SER {ser:.4f}  CER {c:.4f}  {ms:.2f} ms/sample')
        except _Timeout:
            signal.alarm(0)
            print(f'  {label:<7} DNF (>{args.dnf_s} s/sample)')
        except RuntimeError as e:
            signal.alarm(0)
            del es
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
            msg = str(e).lower()
            if 'out of memory' in msg or "can't allocate memory" in msg:
                if s < L:
                    s += 1
                    print(f'  (out of memory at {passes:,} passes; retrying with {J ** s:,})')
                    continue
                print(f'  {label:<7} out of memory even at the maximum split')
            else:
                print(f'  {label:<7} error: {str(e)[:80]}')
        break


if __name__ == '__main__':
    main()

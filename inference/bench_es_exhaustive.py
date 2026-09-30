#!/usr/bin/env python3
"""
GPU-vectorized TRUE exhaustive Top-J search (memory-chunked). CER + latency.

Produces the "Top-J (Top 2)" Time column of the paper's main Table 1
(tab:main_results_classical: K=2, Q=64, J=2, 60 s/sample cap) and the Top-J
column of tab:response_gpu_operating (min J reaching CER <= .05, ">60 s"
entries). Decoder: rules/top_j_beam.py BeamTopJDecoder, run exhaustively over
the top-J list (never pruned). Table 1's Top-J SER/CER are the original-protocol
numbers (tab:app_classical_topj_results); this script re-times them on GPU.

Positional args:  <K>  <code_length L>  <passes>   [J]

  K            number of users
  L            code length (blocklength). Selects the dataset:
                 K==2 : <--data_root>/{tiny,small,moderate,large}_LDPC (L=12/18/24/48)
                 K>=3 : <--protocol_dir>/K<K>                          (L must be 12)
  passes       how many sequential chunks to split the search into (the
               memory<->latency dial). The J^L candidate codewords are covered
               by this many disjoint passes; result is identical to one big
               parallel sweep. Achievable pass counts are powers of J, so this
               is ROUNDED UP to the nearest J^s (the script prints what it used).
                 few passes  -> big per-pass beam -> fast, but may OOM
                 many passes -> small per-pass beam -> never OOM, but DNF
  J            proposal width, top-J symbols kept per position (default: K).
               Must be >= K to recover K codewords. The decode is exhaustive
               over that top-J list (never pruned).

Reports per-sample CER, latency, and pass count.
  DNF : per-sample decode exceeds --dnf_s (default 60s ~ paper 24h/1000 samples)
  OOM : one pass needs more VRAM than available -> ask for MORE passes
  >1s : exceeds a real-time receiver budget (~250x CIDER)

Table 1 (K=2, J=2, RTX 3090, --bs 8) -- closest LOGGED ANALOGUE, NOT an exact
reproduction. No logged command reproduces Table 1's Top-J (Top 2) times
(2.52 / 4.83 / 77.61 ms at L=12/18/24). The decoded output (hence CER) does not
depend on the pass count, but latency does, strongly (L=12, bs=8: 1 pass
0.25 ms, 4 passes 0.71 ms, 64 passes 9.17 ms). With the fewest passes that fit
in 24 GB VRAM:
  python inference/bench_es_exhaustive.py 2 tiny 1 --n 512       # L=12: expect ~0.2 ms/sample
  python inference/bench_es_exhaustive.py 2 small 1 --n 512      # L=18: expect ~1.2 ms/sample
  python inference/bench_es_exhaustive.py 2 moderate 32 --n 512  # L=24: expect ~82.0 ms/sample (fewer passes OOM)
  python inference/bench_es_exhaustive.py 2 large 1073741824     # L=48: DNF (>60 s), the ">60,000" entry
  (L=48 has 2^48 leaves: pass counts small enough to finish OOM, and pass
   counts that fit in memory exceed the 60 s cap.)
At 64 passes (seq_split=6) all scales give ~9.1 / 14.0 / 88.3 ms instead. Neither
setting matches the paper's 2.52 / 4.83 / 77.61. Note that 77.61 equals the
original CPU time 77611.5 ms / 1000 (tab:app_classical_topj_results). Table 1's
Top-J SER/CER are the original-protocol numbers, not values from this script.

tab:response_gpu_operating (Top-J column, K=2, L=18, J=3 min for CER <= .05):
  python inference/bench_es_exhaustive.py 2 small 729 3 --n 32   # logged: 1197.9 ms/sample, CER .0156

Examples:
  python inference/bench_es_exhaustive.py 3 tiny 1        # K=3, L=12, 1 pass (no split)
  python inference/bench_es_exhaustive.py 3 tiny 16 4     # K=3, L=12, ~16 passes, Top-4
  python inference/bench_es_exhaustive.py 2 large 64      # K=2, L=48, ~64 passes
  python inference/bench_es_exhaustive.py 2 24 1          # numbers still work (L=24)
  python inference/bench_es_exhaustive.py 2 tiny 1 --device cpu --n 16   # CPU smoke test
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
L_BY_NAME = {'tiny': 12, 'small': 18, 'moderate': 24, 'large': 48}


def parse_L(tok):
    if tok in L_BY_NAME:
        return L_BY_NAME[tok]
    try:
        return int(tok)
    except ValueError:
        sys.exit(f'L must be a length or one of {list(L_BY_NAME)} (got {tok!r})')


class _Timeout(Exception):
    pass


signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Timeout()))


def load(K, L, data_root, protocol_dir):
    if K == 2:
        if L not in SCALE_BY_L:
            sys.exit(f'no K=2 dataset for L={L} (have {sorted(SCALE_BY_L)})')
        base = os.path.join(data_root, SCALE_BY_L[L])
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


def cer(preds, X0, K):
    cw = tot = 0
    for b in range(len(preds)):
        c = np.array([[(preds[b, i] != X0[b, j]).sum() for j in range(K)] for i in range(K)])
        r, co = linear_sum_assignment(c)
        for i, j in zip(r, co):
            cw += int((preds[b, i] == X0[b, j]).all()); tot += 1
    return 1 - cw / tot


def main():
    p = argparse.ArgumentParser(description='GPU exhaustive Top-J search.')
    p.add_argument('K', type=int)
    p.add_argument('L', help='code length, or scale name: tiny/small/moderate/large')
    p.add_argument('passes', type=int, help='desired # of sequential chunks (rounded up to J^s)')
    p.add_argument('J', type=int, nargs='?', help='proposal width (default: K)')
    p.add_argument('--n', type=int, default=256, help='samples for CER/latency')
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

    # passes must be J^s; round the request UP to the nearest achievable power.
    req = max(1, args.passes)
    s = 0
    while J ** s < req:
        s += 1
    if s > L:
        s = L
    passes = J ** s
    if passes != req:
        print(f'(requested {req} passes -> using {passes} = {J}^{s}, nearest power of J)')

    dev = torch.device(args.device)

    def sync():
        if dev.type == 'cuda':
            torch.cuda.synchronize()

    Y, X0, H = load(K, L, os.path.expanduser(args.data_root),
                    os.path.expanduser(args.protocol_dir))
    beam = J ** (L - s)           # exhaustive per pass, never pruned
    print(f'exhaustive Top-J  |  K={K} L={L} J={J} (Q=64)  '
          f'{passes:,} passes (split={s}),  {beam:,} paths/pass  [{dev}]')
    print('-' * 68)

    try:
        es = BeamTopJDecoder(64, L, K, H.shape[0], H, beam_width=beam,
                             proposal_width=J, device=dev, seq_split=s)
        n = min(args.n, len(Y))
        Yt = torch.as_tensor(Y[:n], dtype=torch.float32, device=dev)
        # DNF gate: warm up + time ONE sample, both under a hard dnf_s cap.
        # This is the literal "if a sample takes > dnf_s, quit" -- the alarm
        # covers the warmup too, so a truly slow config quits within dnf_s
        # instead of hanging in an untimed warmup.
        signal.alarm(args.dnf_s)
        es.decode_batch(Yt[:1]); sync()                  # warmup / OOM probe
        signal.alarm(args.dnf_s)
        t0 = time.perf_counter()
        es.decode_batch(Yt[:1]); sync()
        signal.alarm(0)
        probe_ms = (time.perf_counter() - t0) * 1000     # bs=1, upper-bound-ish

        # Passed the gate. Size the measurement so total wall time stays ~<=
        # dnf_s: large n for fast configs, few samples for slow ones. So a
        # feasible-but-slow config never falsely DNFs and never runs forever.
        n_meas = int(min(n, max(args.bs,
                                (args.dnf_s * 1000) / probe_ms // args.bs * args.bs)))

        t0 = time.perf_counter()
        preds = np.concatenate([np.asarray(es.decode_batch(Yt[lo:lo + args.bs]))
                                for lo in range(0, n_meas, args.bs)], 0)
        sync()
        ms = (time.perf_counter() - t0) * 1000 / n_meas
        c = cer(preds, X0[:n_meas], K)
        rt = '  [>1s: not real-time]' if ms > 1000 else ''
        print(f'CER = {c:.4f}   (n={n_meas})')
        print(f'latency = {ms:.1f} ms/sample{rt}')
    except _Timeout:
        signal.alarm(0)
        print(f'DNF  (>{args.dnf_s} s/sample, {passes:,} passes)')
    except RuntimeError as e:
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        msg = str(e).lower()
        if 'out of memory' in msg or "can't allocate memory" in msg:
            print(f'OOM  ({beam:,} paths/pass too large) -> ask for more passes')
        else:
            print(f'ERR  {str(e)[:80]}')


if __name__ == '__main__':
    main()

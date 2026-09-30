#!/usr/bin/env python3
"""
CIDER decode-time benchmark.

Usage:
    python inference/bench_scales_optimized.py --scales tiny small moderate large
"""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import open_dict
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.eval_protocol import load_backbone, sample_with_diffusion

# name: (checkpoint dir, data dir, L, T)
SCALES = {
    'tiny':     ('checkpoints/tiny_ldpc_tiny_cider',         'tiny_LDPC',     12, 12),
    'small':    ('checkpoints/small_ldpc_small_cider',       'small_LDPC',    18, 16),
    'moderate': ('checkpoints/moderate_ldpc_moderate_cider', 'moderate_LDPC', 24, 20),
    'large':    ('checkpoints/large_ldpc_large_cider',       'large_LDPC',    48, 28),
    'peg':      ('checkpoints/tiny_ldpc_peg_tiny_cider',     'tiny_LDPC_PEG', 12, 12),
    'tree':     ('checkpoints/tiny_tree_tiny_cider',         'tiny_tree',     12, 12),
    'l72':      ('checkpoints/l72_ldpc_large_cider',          'l72_ldpc',     72, 40),
    'xlarge':   ('checkpoints/xlarge_ldpc_large_cider',       'xlarge_ldpc',  96, 60),
}

# name: (H subdir, n_s, Eb, sigma2, matrix_type, K), generated on the fly
ONTHEFLY = {
    'l72':    ('l72_ldpc',    24, 10.0, 1.0, 'partial_dft', 2),
    'xlarge': ('xlarge_ldpc', 24, 10.0, 1.0, 'partial_dft', 2),
}


def gen_onthefly(h_path, n, n_s, Eb, sigma2, matrix_type, K, seed):
    from data.data_onthefly import QaryOnTheFlyDataset
    ds = QaryOnTheFlyDataset(h_matrix_path=str(h_path), K=K, Eb_dB=Eb, n_s=n_s,
                             sigma2=sigma2, matrix_type=matrix_type,
                             num_samples=n, fixed_seed=seed)
    Ys, X0s = [], []
    for i in range(n):
        Y, cw = ds[i]                      # Y [N,Q], cw [K,N]
        Ys.append(Y); X0s.append(cw)
    return torch.stack(Ys), torch.stack(X0s), ds.H_matrix


def enable_sdpa(backbone):
    n = 0
    for m in backbone.modules():
        if type(m).__name__ == 'EdgeSelfAttention':
            m.use_sdpa = True
            n += 1
    return n


def score(preds, X0):
    B, K, N = preds.shape
    cs = cw = 0
    for b in range(B):
        cost = (preds[b].unsqueeze(1) != X0[b].unsqueeze(0)).sum(-1)
        r, c = linear_sum_assignment(cost.cpu().numpy())
        for i, j in zip(r, c):
            dd = int(cost[i, j])
            cs += N - dd
            if dd == 0:
                cw += 1
    return cs, B * K * N, cw, B * K


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scales', nargs='+',
                   default=['tiny', 'small', 'moderate', 'large'],
                   choices=list(SCALES))
    p.add_argument('--data_root', default='~/data/demix',
                   help='Root of the cached K=2 test sets ({tiny,small,moderate,'
                        'large}_LDPC/test_data.pt)')
    p.add_argument('--h_dir', default='data/scale_sweep',
                   help='Root holding {l72_ldpc,xlarge_ldpc}/H_matrix.pt for the '
                        'on-the-fly L=72/96 scales (not shipped; see the module '
                        'docstring for how to build them)')
    p.add_argument('--onthefly_seed', type=int, default=199999,
                   help='fixed_seed for on-the-fly L=72/96 evidence (per-sample '
                        'seed = this + index). Default: the held-out test seed '
                        'used by dataloader.py; the paper rows used 42.')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_samples', type=int, default=None,
                   help='Cap samples per scale (default: entire cached test '
                        'set; on-the-fly scales default to 256 if unset).')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True,
                   help='Run decode under torch.autocast(float16) (CUDA only).')
    p.add_argument('--sdpa', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fast_sampler', action=argparse.BooleanOptionalAction, default=True,
                   help='Vectorized discrete sampler (_sample_discrete_fast).')
    p.add_argument('--per_frame', action='store_true',
                   help='Also measure batch-size-1 latency (p50/p90/p99) on a '
                        '--per_frame_n subset.')
    p.add_argument('--per_frame_n', type=int, default=200)
    p.add_argument('--ckpt', default=None,
                   help='Override checkpoint for ALL --scales (the "source" '
                        'weights). This measures MISMATCHED-length decode time: '
                        "source weights run on each scale's (N, M, T). The model "
                        'is length-agnostic (params depend only on Q,K,D; H enters '
                        'at runtime), so this is the transfer setup, timed with the '
                        'optimized stack. Default: each scale uses its matched ckpt.')
    p.add_argument('--steps', type=int, nargs='+', default=None,
                   help='Override T: one value for all --scales, or one per scale '
                        "(1:1 with --scales). Default: the paper's per-scale "
                        'values (tab:app_exact_model_sizing, tab:response_transfer). '
                        'For mismatched length T should scale with N '
                        '(12->12, 18->16, 24->20, 48->28, 72->40, 96->60), else '
                        'transfer is understated.')
    p.add_argument('--save_results', default='logs/scales_optimized.json')
    args = p.parse_args()
    args.data_root = os.path.expanduser(args.data_root)
    args.h_dir = os.path.expanduser(args.h_dir)
    if args.ckpt:
        args.ckpt = os.path.expanduser(args.ckpt)
    if args.save_results:
        args.save_results = os.path.expanduser(args.save_results)

    # Resolve per-scale T: None -> paper defaults; 1 value -> all; list -> 1:1.
    if args.steps is None:
        steps_override = None
    elif len(args.steps) == 1:
        steps_override = {name: args.steps[0] for name in args.scales}
    elif len(args.steps) == len(args.scales):
        steps_override = dict(zip(args.scales, args.steps))
    else:
        raise SystemExit(f'--steps takes 1 value or one per --scales '
                         f'({len(args.scales)}); got {len(args.steps)}')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.tf32:
        torch.set_float32_matmul_precision('high')
    if args.fp16 and dev.type != 'cuda':
        print('WARNING: --fp16 is CUDA-only; running in FP32 on CPU')

    print('=' * 78)
    print('CIDER')
    print('=' * 78)

    rows, results = [], {}
    for name in args.scales:
        ck_dir, data_name, L, T_paper = SCALES[name]
        ck = Path(args.ckpt) if args.ckpt else Path(ck_dir) / 'best_model.ckpt'
        ddir = Path(args.data_root) / data_name
        onthefly = name in ONTHEFLY
        h_otf = (Path(args.h_dir) / ONTHEFLY[name][0] / 'H_matrix.pt'
                 if onthefly else None)
        data_ok = (h_otf.exists() if onthefly
                   else (ddir / 'test_data.pt').exists())
        if not ck.exists() or not data_ok:
            print(f'[skip] {name}: missing checkpoint or data')
            continue

        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            dm, cfg = load_backbone(str(ck), dev)
        bb = dm.backbone
        # matched iff the source ckpt's own training length equals this scale's N.
        src_N = int(cfg.data.get('N', L))
        src_K = int(cfg.data.get('K_max', cfg.data.get('K_true', cfg.data.get('K', 2))))
        kind = 'matched' if src_N == L else f'xfer(N{src_N}->{L})'
        T = steps_override[name] if steps_override else T_paper
        T_ckpt = cfg.model.get('inference_steps', None)

        with open_dict(dm.config):
            dm.config.model.fast_sampler = bool(args.fast_sampler)
        n_sdpa = enable_sdpa(bb) if args.sdpa else 0
        if args.compile:
            dm.backbone = torch.compile(bb, mode='reduce-overhead')

        # evidence prep (not timed)
        if onthefly:
            _, n_s, Eb, sigma2, mtype, K_gen = ONTHEFLY[name]
            n_gen = args.max_samples if args.max_samples else 256
            Y_all, X0_all, H = gen_onthefly(h_otf, n_gen, n_s, Eb, sigma2,
                                            mtype, K_gen, args.onthefly_seed)
            H = H.to(dev)
        else:
            data = torch.load(ddir / 'test_data.pt', weights_only=True)
            H = torch.load(ddir / 'H_matrix.pt',
                           weights_only=True)['H_matrix'].to(dev)
            Y_all, X0_all = data['Y'], data['gt_codewords']
        data_K = X0_all.shape[1]
        if data_K != src_K:
            print(f'  [skip] {name}: source ckpt is K={src_K} but data is '
                  f'K={data_K}. Length transfer needs matching K (slot_init is '
                  f'K-sized); this would be nonsense, not a decode-time number.')
            del dm
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
            continue

        # point the model at the target length (N, M)
        tgt_N, tgt_M = Y_all.shape[1], H.shape[0]
        dm.N, dm.M = tgt_N, tgt_M
        bb.N, bb.M = tgt_N, tgt_M
        n = len(Y_all) if args.max_samples is None else min(args.max_samples,
                                                            len(Y_all))
        Y_all, X0_all = Y_all[:n], X0_all[:n]

        bs = args.batch_size
        batches = [Y_all[s:s + bs].to(dev).float() for s in range(0, n, bs)]
        if dev.type == 'cuda':
            torch.cuda.synchronize()

        amp = (torch.autocast('cuda', dtype=torch.float16)
               if args.fp16 and dev.type == 'cuda' else contextlib.nullcontext())

        with torch.no_grad(), amp:                   # warmup (+ compile)
            for _ in range(args.warmup):
                sample_with_diffusion(dm, batches[0], H, num_steps=T)
        if dev.type == 'cuda':
            torch.cuda.synchronize()

        preds = []
        t0 = time.perf_counter()
        with torch.no_grad(), amp:
            for Yb in batches:
                preds.append(sample_with_diffusion(dm, Yb, H, num_steps=T))
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ms_per_sample = dt * 1000.0 / n

        cs = ts = cw = tw = 0
        for i, s in enumerate(range(0, n, bs)):
            a, b, c, dd = score(preds[i], X0_all[s:s + bs].to(dev))
            cs += a; ts += b; cw += c; tw += dd
        ser, cer = 1.0 - cs / ts, 1.0 - cw / tw
        del batches, preds

        pf = None
        if args.per_frame:
            frames = [Y_all[i:i + 1].to(dev).float()
                      for i in range(min(args.per_frame_n, n))]
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            with torch.no_grad(), amp:
                for _ in range(args.warmup):
                    sample_with_diffusion(dm, frames[0], H, num_steps=T)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            ts_ = []
            with torch.no_grad(), amp:
                for Yb in frames:
                    a0 = time.perf_counter()
                    sample_with_diffusion(dm, Yb, H, num_steps=T)
                    if dev.type == 'cuda':
                        torch.cuda.synchronize()
                    ts_.append((time.perf_counter() - a0) * 1000.0)
            ts_.sort()
            q = lambda x: ts_[min(len(ts_) - 1, int(x * len(ts_)))]
            pf = dict(p50=q(.5), p90=q(.9), p99=q(.99), n=len(ts_))
            del frames

        rows.append((name, L, T, n, ser, cer, ms_per_sample, pf, kind))
        results[name] = dict(L=L, T=T, n=n, ser=ser, cer=cer,
                             ms_per_sample=ms_per_sample,
                             batch_size=bs, per_frame=pf, kind=kind,
                             src_N=src_N, ckpt=str(ck),
                             tf32=args.tf32, sdpa=args.sdpa,
                             compile=args.compile, sdpa_modules=n_sdpa,
                             fast_sampler=args.fast_sampler, fp16=args.fp16,
                             onthefly_seed=(args.onthefly_seed if onthefly
                                            else None))
        print(f'{name} (L={L}, K={data_K})')
        print(f'  {"CIDER":<7} SER {ser:.4f}  CER {cer:.4f}  {ms_per_sample:.2f} ms/sample'
              + (f'  (bs=1 p50 {pf["p50"]:.2f} ms)' if pf else ''))

        dm.backbone = bb
        del dm
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    if args.save_results:
        o = Path(args.save_results)
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

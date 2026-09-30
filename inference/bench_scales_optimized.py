#!/usr/bin/env python3
"""
Per-sample decode time over the FULL test set, for tiny/small/moderate/large,
with the optimized inference stack.

Paper tables produced:
  * Main Table 1 (tab:main_results_classical), CIDER "Time" column
    (L=12/18/24/48, ms/sample at batch size 8). The SER/CER in that row are the
    original full-test-set numbers (tab:main_results_ser_cer); this script
    re-measures the time.
  * tab:response_transfer, L=72/96 rows (tiny L=12 checkpoint applied zero-shot,
    T=40/60, 256 on-the-fly examples each): transfer CER, and the 17.57/34.90
    ms latency quoted in the transfer appendix.

Optimizations applied (each independently toggleable so their contributions
are separable):
    --tf32      torch.set_float32_matmul_precision('high'); TF32 tensor cores
    --sdpa      fused scaled_dot_product_attention in EdgeSelfAttention
                (~10 kernels -> 1, x8 modules per forward). Verified to leave
                decoded grids bit-identical.
    --compile   torch.compile(mode='reduce-overhead') -> CUDA graphs, which
                collapse the ~5-7k per-decode kernel launches that dominate
                small-batch latency.
    --fast_sampler  vectorized discrete sampler (diffusion.py
                _sample_discrete_fast, i.e. model.fast_sampler=true).
    --fp16      torch.autocast(float16), CUDA only.

Reports two DIFFERENT quantities:
    throughput   per-sample ms at --batch_size (Table 1 style; bs=8)
    latency      per-sample ms at batch size 1 (what a receiver decoding one
                 frame at a time actually waits; --per_frame)

Only decode() is timed. Excluded: checkpoint load, dataset load, host->device
copy (batches are staged on-device first), Tanner-cache build, compile warmup,
and Hungarian matching (done afterwards on cached predictions).

All scales are K=2, so decoding goes through Diffusion._sample
(sample_with_diffusion), matching evaluate_batch's dispatch. NOTE: that path is
stochastic (random_slot_first=True), so SER/CER will vary slightly run to run
and exact-match comparisons against a reference are meaningless there.

Timing requires an IDLE GPU. Without CUDA (e.g. CUDA_VISIBLE_DEVICES="") it
runs on CPU, which is only useful as a smoke test (--compile/--fp16 are meant
for CUDA).

L=72/96 (l72/xlarge) have no cached test set: evidence is generated on the fly
from <--h_dir>/{l72_ldpc,xlarge_ldpc}/H_matrix.pt, which are NOT shipped with
the repo. The paper's H matrices are the rate-1/3 (d_v=2, d_c=3) Mobius-ladder
codes built with seed 42:
    python data/gen_data/construct_H.py --q 64 --L 72 --M 48 --d_v 2 --d_c 3 \
        --seed 42 --output data/scale_sweep/l72_ldpc/H_matrix.pt
    python data/gen_data/construct_H.py --q 64 --L 96 --M 64 --d_v 2 --d_c 3 \
        --seed 42 --output data/scale_sweep/xlarge_ldpc/H_matrix.pt

Paper reproduction (RTX 3090):
    # Table 1 CIDER Time column (full 15k test sets, bs=8). The logged run
    # closest to the paper used exactly these flags (no --fp16).
    python inference/bench_scales_optimized.py --tf32 --sdpa --compile \
        --fast_sampler --batch_size 8

    # tab:response_transfer L=72/96 rows (T=40/60 are the defaults for l72/xlarge)
    python inference/bench_scales_optimized.py --tf32 --sdpa --compile \
        --fast_sampler --per_frame --batch_size 8 \
        --ckpt checkpoints/tiny_ldpc_tiny_cider/best_model.ckpt \
        --scales l72 xlarge --max_samples 256
    (The paper's L=72/96 samples used on-the-fly seed 42; the default
     --onthefly_seed is now 199999, the release's held-out test seed, so CER
     will differ slightly at n=256. Pass --onthefly_seed 42 for those samples.)

Usage:
    python inference/bench_scales_optimized.py --tf32 --sdpa --compile
    python inference/bench_scales_optimized.py --scales tiny large --max_samples 2000
    python inference/bench_scales_optimized.py --tf32 --sdpa --compile --per_frame

    # MISMATCHED length: tiny-trained weights, timed on every length at that
    # length's T (T must scale with N, else transfer is understated).
    python inference/bench_scales_optimized.py --tf32 --sdpa --compile --per_frame \
        --ckpt checkpoints/tiny_ldpc_tiny_cider/best_model.ckpt \
        --scales tiny small moderate large --steps 12 16 20 28
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
# T is pinned to the paper's values and NOT read from the checkpoint:
# tab:app_exact_model_sizing for L<=48 (12/16/20/28) and tab:response_transfer
# for L=72/96 (40/60). The checkpoints carry inference_steps=10, which disagrees
# with the paper and yields ~5x worse error (tiny: SER 0.0057 at T=10 vs 0.0013
# at T=12, against the published 0.0011). Trusting the embedded config silently
# reproduces the wrong number. Override with --steps.
#
# l72/xlarge (L=72/96) extend the blocklength sweep. They have only an
# H_matrix.pt (under --h_dir, no cached test_data.pt), so their evidence is
# generated ON THE FLY (see ONTHEFLY below). The paper evaluates them with the
# tiny checkpoint (--ckpt); the matched checkpoint dirs below are not released.
SCALES = {
    'tiny':     ('checkpoints/tiny_ldpc_tiny_cider',         'tiny_LDPC',     12, 12),
    'small':    ('checkpoints/small_ldpc_small_cider',       'small_LDPC',    18, 16),
    'moderate': ('checkpoints/moderate_ldpc_moderate_cider', 'moderate_LDPC', 24, 20),
    'large':    ('checkpoints/large_ldpc_large_cider',       'large_LDPC',    48, 28),
    'l72':      ('checkpoints/l72_ldpc_large_cider',          'l72_ldpc',     72, 40),
    'xlarge':   ('checkpoints/xlarge_ldpc_large_cider',       'xlarge_ldpc',  96, 60),
}

# Scales with no cached test_data.pt: build evidence on the fly from
# <--h_dir>/<subdir>/H_matrix.pt, replicating the rate-1/3 generation params
# shared by every other scale (partial_dft sensing, n_s=24, Eb=10 dB, sigma2=1,
# K=2; see data/gen_data/ldpc_large.sh). The per-sample seed comes from
# --onthefly_seed. CPU-only, so it does not touch the GPU being timed.
# name -> (H subdir, n_s, Eb, sigma2, matrix_type, K).
ONTHEFLY = {
    'l72':    ('l72_ldpc',    24, 10.0, 1.0, 'partial_dft', 2),
    'xlarge': ('xlarge_ldpc', 24, 10.0, 1.0, 'partial_dft', 2),
}


def gen_onthefly(h_path, n, n_s, Eb, sigma2, matrix_type, K, seed):
    """Generate (Y [n,N,Q], X0 [n,K,N], H [M,N]) via the AMP inner channel.

    One sample at a time on CPU (QaryOnTheFlyDataset is per-item), so keep n
    modest with --max_samples for the long codes. fixed_seed makes the CER
    reproducible run to run; the decoded-time number is what we are after.
    """
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
    """Hungarian-matched correct symbols / codewords (untimed)."""
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
    # Default excludes l72/xlarge: those generate evidence on the fly (slow,
    # CPU), so opt into them explicitly with --scales.
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
    p.add_argument('--tf32', action='store_true')
    p.add_argument('--fp16', action='store_true',
                   help="Run decode under torch.autocast(float16) (CUDA only). "
                        "main.py's Trainer evaluates with precision='16-mixed' "
                        "(configs/training/default.yaml), i.e. the original "
                        "full-test-set timings. Off by default.")
    p.add_argument('--sdpa', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--fast_sampler', action='store_true',
                   help='Use the vectorized discrete sampler '
                        '(_sample_discrete_fast). Not bit-identical '
                        'to the original because of confidence ties; '
                        'validate SER/CER before reporting.')
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
    print('CIDER per-sample decode time, full test set, optimized stack')
    print('=' * 78)
    print(f'device      : {torch.cuda.get_device_name(0) if dev.type=="cuda" else "cpu"}')
    print(f'tf32={args.tf32}  fp16={args.fp16}  sdpa={args.sdpa}  '
          f'compile={args.compile}  fast_sampler={args.fast_sampler}  '
          f'batch_size={args.batch_size}')
    if args.ckpt:
        print(f'MISMATCHED-length decode: source ckpt {args.ckpt} run on every '
              f'--scales target (rows tagged matched / xfer(Nsrc->Ntgt))')
    print('decode-only; excludes model/data load, H2D copy, Hungarian matching')
    print('NOTE: requires an idle GPU, and K=2 decoding is stochastic\n')

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

        dm, cfg = load_backbone(str(ck), dev)
        bb = dm.backbone
        # matched iff the source ckpt's own training length equals this scale's N.
        src_N = int(cfg.data.get('N', L))
        src_K = int(cfg.data.get('K_max', cfg.data.get('K_true', cfg.data.get('K', 2))))
        kind = 'matched' if src_N == L else f'xfer(N{src_N}->{L})'
        T = steps_override[name] if steps_override else T_paper
        T_ckpt = cfg.model.get('inference_steps', None)
        if T_ckpt is not None and int(T_ckpt) != T:
            print(f'  [{name}] checkpoint says inference_steps='
                  f'{T_ckpt}; using T={T}')

        # The released Diffusion reads the fast-sampler switch from its config
        # (config.model.fast_sampler), same as main.py +fast_sampler=true.
        with open_dict(dm.config):
            dm.config.model.fast_sampler = bool(args.fast_sampler)
        n_sdpa = enable_sdpa(bb) if args.sdpa else 0
        if args.compile:
            dm.backbone = torch.compile(bb, mode='reduce-overhead')

        # --- evidence prep (NOT timed): cached load, or on-the-fly generation.
        # Either way this only produces Y/X0/H; the decode clock starts later,
        # after batches are staged on-device.
        if onthefly:
            _, n_s, Eb, sigma2, mtype, K_gen = ONTHEFLY[name]
            n_gen = args.max_samples if args.max_samples else 256
            print(f'  [{name}] generating {n_gen} on-the-fly samples on CPU '
                  f'(seed {args.onthefly_seed}; evidence prep, NOT part of '
                  f'decode latency)...', flush=True)
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

        # Point sampler + backbone at the TARGET geometry. self.N shapes the
        # masked grid (Diffusion._sample_discrete) and the backbone reshapes
        # hidden states with it (CIDER forward); both are otherwise inherited from the SOURCE
        # checkpoint's config, so a mismatched-length ckpt would build a wrong-N
        # grid and crash in the syndrome. The forward is genuinely N-agnostic
        # (the Tanner cache is rebuilt from H every call), so overriding N/M is
        # sufficient and needs no retraining -- same principle as
        # inference/eval_code_transfer.py build_transfer_model, which instead rebuilds the
        # model at the target (N, M). Set on the UNCOMPILED bb (not dm.backbone,
        # which may be a torch.compile wrapper) and BEFORE warmup, so compile
        # traces at the correct N. No-op when matched.
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
        print(f'{name:<9} L={L:<3} T={T:<3} n={n:<6} '
              f'SER={ser:.4f} CER={cer:.4f}  '
              f'{ms_per_sample:7.3f} ms/sample @bs={bs}'
              + (f'   |  bs=1 p50 {pf["p50"]:.2f} ms' if pf else '')
              + (f'   [{kind}]' if args.ckpt else ''))

        dm.backbone = bb
        del dm
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    print('\n' + '=' * 78)
    print(f'{"scale":<9} {"L":>3} {"T":>3} {"n":>6} {"SER":>8} {"CER":>8} '
          f'{"ms/sample":>10}' + ('  ' + f'{"bs=1 p50":>9}' if args.per_frame else ''))
    print('-' * 78)
    for name, L, T, n, ser, cer, ms, pf, kind in rows:
        line = (f'{name:<9} {L:>3} {T:>3} {n:>6} {ser:>8.4f} {cer:>8.4f} '
                f'{ms:>10.3f}')
        if args.per_frame and pf:
            line += f'  {pf["p50"]:>9.2f}'
        if args.ckpt:
            line += f'   {kind}'
        print(line)
    print('\nms/sample  = throughput at the given batch size (Table 1 style)')
    if args.per_frame:
        print('bs=1 p50   = single-frame latency (what a real-time receiver waits)')

    if args.save_results:
        o = Path(args.save_results)
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(results, indent=2))
        print(f'\nSaved: {o}')


if __name__ == '__main__':
    main()

"""
SNR sweep evaluator, with optional channel / sensing-matrix mismatch.

Loads each K's checkpoint exactly once, then loops SNR internally by
re-instantiating the on-the-fly dataset with a new Eb/N0 (fixed seed).
Avoids the per-SNR `main.py` subprocess + checkpoint-load overhead.

Uses the same inference dispatch as eval_protocol.py / eval_per_k.py:
    K<=2  : sample_with_diffusion (random_slot_first=True)
    K=3-5 : sample_first_alone
    K=6-8 : sample_with_quality_head if quality head loaded, else first_alone

--fading / --matrix-type evaluate the AWGN / partial-DFT trained checkpoint
under a channel or sensing codebook it never saw (no retraining, no CSI):

  * Main-text tab:fading_stress, CIDER row (K=2, L=12, Eb/N0 = 10 dB, i.e.
    SNR = -0.79 dB; Rician = per-user uniform LoS phase + slot-wise diffuse
    part, E|h|^2 = 1):
        python inference/eval_snr_sweep.py --K 2 --snr -0.79 \
            --num-samples 10000 --batch-size 256 \
            --fading none rician rayleigh --rician-k-dB 10 0 -10
    (--rician-k-dB 20 10 5 0 -5 -10 gives the full kappa curve.)
  * Appendix tab:evidence_shifts, "Sensing" row (same checkpoint, AWGN,
    10,000 examples per sensing family):
        python inference/eval_snr_sweep.py --K 2 --snr -0.79 \
            --num-samples 10000 --batch-size 256 \
            --matrix-type complex_hadamard qpsk_hadamard qr_gaussian

The SIC-BP row of tab:fading_stress is inference/eval_sic_bp_channel.py on
the same test stream. Test data use fixed_seed=199999 (per-sample seeding),
disjoint from the training-data seed (42); the sensing matrix A is seed 42.

Usage:
    python inference/eval_snr_sweep.py
    python inference/eval_snr_sweep.py --K 2 3 4 5 --snr -4 -2 0 2 4
    python inference/eval_snr_sweep.py --num-samples 2000 --batch-size 256
    python inference/eval_snr_sweep.py --K 6 7 8 --no-quality-head   # ablation

SNR convention: per-user per-complex-channel-use SNR (Es/N0).
For tiny_ldpc: Eb/N0 [dB] = SNR [dB] + 10*log10(L*n_s/B) ≈ SNR + 10.79 dB.
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_onthefly import QaryOnTheFlyDataset
from inference.eval_protocol import (INFERENCE_STEPS, evaluate_batch,
                                     load_backbone, load_quality_head)


def make_loader(h_path, K, Eb_dB, n_s, sigma2, num_samples, batch_size,
                num_workers, fading='none', rician_k_dB=None,
                fading_coherence='slot', matrix_type='partial_dft'):
    """Build the on-the-fly loader for one (K, channel condition, Eb/N0) point.

    fading / matrix_type are evaluation-time mismatches: the model is NOT
    retrained, so these measure robustness to a channel or codebook the
    decoder never saw. E[|h|^2] = 1 in every fading mode, so the nominal
    Eb/N0 is unchanged and curves stay comparable across conditions.
    """
    kw = {}
    if fading != 'none':
        kw['fading'] = fading
        kw['fading_coherence'] = fading_coherence
        if fading == 'rician':
            kw['rician_k_dB'] = rician_k_dB
    ds = QaryOnTheFlyDataset(
        h_path, K=K, Eb_dB=Eb_dB, n_s=n_s, sigma2=sigma2,
        matrix_type=matrix_type,
        num_samples=num_samples, fixed_seed=199999, Eb_range=None, **kw
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def build_conditions(fadings, rician_ks, matrix_types, coherence):
    """Expand the CLI axes into concrete (fading, rician_k_dB, matrix_type)
    conditions. 'rician' fans out over every requested K-factor, which is what
    traces a graceful-degradation curve from ~AWGN (+20 dB) to ~Rayleigh."""
    conds = []
    for mt in matrix_types:
        for f in fadings:
            if f == 'rician':
                for k in rician_ks:
                    conds.append((f, float(k), mt))
            else:
                conds.append((f, None, mt))
    return conds


def cond_label(fading, rician_k_dB, matrix_type):
    f = fading if fading != 'rician' else f'rician{rician_k_dB:+g}dB'
    return f'{f}/{matrix_type}'


@torch.no_grad()
def evaluate(loader, K, H, checkpoints, device):
    """Run evaluate_batch (eval_protocol) over all batches in loader."""
    correct_sym = total_sym = correct_cw = total_cw = 0
    for batch in loader:
        Y_batch = batch[0]   # evaluate_batch handles .to(device) + .float()
        X0_batch = batch[1]
        res = evaluate_batch(K, Y_batch, X0_batch, H, checkpoints, device)
        if res is None:
            continue
        cs, ts, cw, tw = res
        correct_sym += cs; total_sym += ts
        correct_cw += cw;  total_cw += tw

    ser = 1.0 - correct_sym / max(1, total_sym)
    cer = 1.0 - correct_cw / max(1, total_cw)
    return ser, cer


def strategy_label(K, has_qh):
    if K <= 2:
        return 'standard'
    if K <= 5 or not has_qh:
        return 'first_alone'
    return 'quality_head'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, nargs='+', default=[2, 3, 4, 5])
    p.add_argument('--snr', type=float, nargs='+',
                   default=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
    p.add_argument('--num-samples', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--ckpt-template',
                   default='checkpoints/protocol_scale/K{K}/best_model.ckpt')
    p.add_argument('--h-matrix',
                   default=os.path.expanduser('~/data/demix/tiny_LDPC/H_matrix.pt'))
    p.add_argument('--n-s', type=int, default=24)
    p.add_argument('--sigma2', type=float, default=1.0)
    p.add_argument('--offset', type=float, default=10.79,
                   help='Eb/N0 = SNR + offset (dB). 10.79 = tiny_ldpc rate.')
    p.add_argument('--fading', nargs='+', default=['none'],
                   choices=['none', 'rayleigh', 'rician'],
                   help='Channel conditions to sweep (evaluation-time '
                        'mismatch; the model is not retrained).')
    p.add_argument('--rician-k-dB', type=float, nargs='+',
                   default=[20, 10, 0, -10],
                   help='Rician K-factors, used when --fading includes '
                        'rician. +20dB ~ AWGN, -10dB ~ Rayleigh, so this '
                        'traces a degradation curve rather than one point.')
    p.add_argument('--fading-coherence', default='slot',
                   choices=['slot', 'frame'])
    p.add_argument('--matrix-type', nargs='+', default=['partial_dft'],
                   help='Sensing-matrix (= shared codebook) families: '
                        'partial_dft (training), complex_hadamard, '
                        'qpsk_hadamard, qr_gaussian, hadamard. Hadamard '
                        'variants need a power-of-two Q (Q=64 is fine).')
    p.add_argument('--csv', default='logs/snr_sweep_results.csv')
    p.add_argument('--steps', type=int, default=None,
                   help='Override inference steps for all K (default: per-K '
                        'from inference.eval_protocol.INFERENCE_STEPS).')
    p.add_argument('--no-quality-head', action='store_true',
                   help='Force K>=6 to use first-reveal-alone (skip qhead load).')
    args = p.parse_args()
    args.h_matrix = os.path.expanduser(args.h_matrix)
    args.ckpt_template = os.path.expanduser(args.ckpt_template)
    args.csv = os.path.expanduser(args.csv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"SNR range: {args.snr}")
    print(f"K range:   {args.K}")
    print(f"num_samples: {args.num_samples}, batch_size: {args.batch_size}")
    print(f"Eb/N0 offset: +{args.offset} dB")

    H = torch.load(args.h_matrix, weights_only=False)['H_matrix'].to(device)

    conditions = build_conditions(args.fading, args.rician_k_dB,
                                  args.matrix_type, args.fading_coherence)
    if len(conditions) > 1 or conditions[0] != ('none', None, 'partial_dft'):
        print("Channel/codebook conditions:")
        for c in conditions:
            print(f"  - {cond_label(*c)}")

    results = []  # (K, fading, rician_k_dB, matrix_type, SNR, Eb, SER, CER, ms)
    for K in args.K:
        ckpt = args.ckpt_template.format(K=K)
        if not os.path.exists(ckpt):
            print(f"[skip] K={K}: checkpoint missing at {ckpt}")
            continue

        # Override INFERENCE_STEPS dict so evaluate_batch sees the new value.
        if args.steps is not None:
            INFERENCE_STEPS[K] = args.steps
        steps = INFERENCE_STEPS.get(K, 16)

        diffusion_model, _ = load_backbone(ckpt, device)

        # Optionally load quality head for K>=6 (matches eval_protocol's logic).
        qhead = None
        if K >= 6 and not args.no_quality_head:
            qhead_path = ckpt.replace('best_model.ckpt',
                                      'best_quality_head.ckpt')
            if os.path.exists(qhead_path):
                qhead = load_quality_head(
                    qhead_path, diffusion_model.backbone.D, device
                )
                print(f"  loaded quality head: {qhead_path}")

        checkpoints = {K: (diffusion_model, qhead)}
        strat = strategy_label(K, qhead is not None)
        print(f"\n{'='*64}\nK={K}  ckpt={ckpt}\n"
              f"strategy={strat}  inference_steps={steps}\n{'='*64}")

        for (fad, rk, mt) in conditions:
            label = cond_label(fad, rk, mt)
            if len(conditions) > 1:
                print(f"  --- condition: {label} ---")
            for snr in args.snr:
                eb = round(snr + args.offset, 4)
                t0 = time.perf_counter()
                try:
                    loader = make_loader(
                        args.h_matrix, K, eb, args.n_s, args.sigma2,
                        args.num_samples, args.batch_size, args.num_workers,
                        fading=fad, rician_k_dB=rk,
                        fading_coherence=args.fading_coherence,
                        matrix_type=mt)
                    ser, cer = evaluate(loader, K, H, checkpoints, device)
                except Exception as e:
                    print(f"  [skip] {label} SNR={snr:+.1f}: "
                          f"{type(e).__name__}: {e}")
                    continue
                dt = time.perf_counter() - t0
                ms_per = dt * 1000 / args.num_samples
                print(f"  SNR={snr:+5.1f}  Eb/N0={eb:6.2f}  "
                      f"SER={ser:.4f}  CER={cer:.4f}  ({ms_per:6.2f} ms/sample)")
                results.append((K, fad, '' if rk is None else rk, mt,
                                snr, eb, ser, cer, ms_per))
                del loader

        del diffusion_model, qhead, checkpoints
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Combined table
    print(f"\n{'='*64}\nCombined Summary\n{'='*64}")
    print(f"{'K':<4} {'condition':<22} {'SNR':>6} {'Eb/N0':>8} "
          f"{'SER':>10} {'CER':>10} {'ms/samp':>9}")
    print("-" * 84)
    for K, fad, rk, mt, snr, eb, ser, cer, ms in results:
        lab = cond_label(fad, None if rk == '' else rk, mt)
        print(f"{K:<4} {lab:<22} {snr:>+6.1f} {eb:>8.2f} "
              f"{ser:>10.4f} {cer:>10.4f} {ms:>9.2f}")

    # CSV
    os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
    with open(args.csv, 'w') as f:
        f.write("K,fading,rician_k_dB,matrix_type,SNR,EbN0,SER,CER,"
                "ms_per_sample\n")
        for r in results:
            f.write(','.join(str(x) for x in r) + '\n')
    print(f"\nSaved CSV: {args.csv}")


if __name__ == '__main__':
    main()

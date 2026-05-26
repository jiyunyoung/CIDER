"""
SNR sweep evaluator.

Loads each K's checkpoint exactly once, then loops SNR internally by
re-instantiating the on-the-fly dataset with a new Eb/N0 (fixed seed).
Avoids the per-SNR `main.py` subprocess + checkpoint-load overhead.

Uses the same inference dispatch as eval_protocol.py / eval_per_k.py:
    K<=2  : sample_with_diffusion (random_slot_first=True)
    K=3-5 : sample_first_alone
    K=6-8 : sample_with_quality_head if quality head loaded, else first_alone

Usage:
    python inference/eval_snr_sweep.py
    python inference/eval_snr_sweep.py --K 2 3 4 5 --snr -4 -2 0 2 4
    python inference/eval_snr_sweep.py --num-samples 2000 --batch-size 256
    python inference/eval_snr_sweep.py --K 6 7 8 --no-quality-head   # ablation

SNR convention: per-user per-complex-channel-use SNR (Es/N0).
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
                num_workers):
    ds = QaryOnTheFlyDataset(
        h_path, K=K, Eb_dB=Eb_dB, n_s=n_s, sigma2=sigma2,
        num_samples=num_samples, fixed_seed=199999, Eb_range=None,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


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
    p.add_argument('--csv', default='logs/snr_sweep_results.csv')
    p.add_argument('--steps', type=int, default=None,
                   help='Override inference steps for all K (default: per-K '
                        'from inference.eval_protocol.INFERENCE_STEPS).')
    p.add_argument('--no-quality-head', action='store_true',
                   help='Force K>=6 to use first-reveal-alone (skip qhead load).')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"SNR range: {args.snr}")
    print(f"K range:   {args.K}")
    print(f"num_samples: {args.num_samples}, batch_size: {args.batch_size}")
    print(f"Eb/N0 offset: +{args.offset} dB")

    H = torch.load(args.h_matrix, weights_only=False)['H_matrix'].to(device)

    results = []  # (K, SNR, Eb, SER, CER, ms_per_sample)
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

        for snr in args.snr:
            eb = round(snr + args.offset, 4)
            t0 = time.perf_counter()
            loader = make_loader(args.h_matrix, K, eb, args.n_s, args.sigma2,
                                 args.num_samples, args.batch_size,
                                 args.num_workers)
            ser, cer = evaluate(loader, K, H, checkpoints, device)
            dt = time.perf_counter() - t0
            ms_per = dt * 1000 / args.num_samples
            print(f"  SNR={snr:+5.1f}  Eb/N0={eb:6.2f}  "
                  f"SER={ser:.4f}  CER={cer:.4f}  ({ms_per:6.2f} ms/sample)")
            results.append((K, snr, eb, ser, cer, ms_per))
            del loader

        del diffusion_model, qhead, checkpoints
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Combined table
    print(f"\n{'='*64}\nCombined Summary\n{'='*64}")
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

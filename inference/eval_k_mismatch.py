#!/usr/bin/env python3
"""
Load mismatch: decode K_true-user data with a decoder trained for K_model users.

Produces the appendix load-mismatch sweep (Table tab:load_mismatch, the
K_hat = K-1 / K / K+1 columns) that backs the discussion under
tab:estimated_load: what happens when the per-bin load is unknown or
mis-estimated? The per-K results elsewhere always pair the K-th decoder with
K-user data; this sweeps the off-diagonal.

Because the decoder emits K_model rows while the frame contains K_true users,
the metrics differ from the matched case:

    PUPE  fraction of TRUE users not exactly recovered by any emitted row.
          This is the standard URA per-user error probability (the paper's
          MDR) and reduces exactly to the paper's CER when K_model == K_true.
    FA    fraction of EMITTED rows that are not an exact codeword of some true
          user (spurious output, the paper's FAR). Only meaningful when
          K_model > K_true.
    SER   symbol error over the min(K_model, K_true) Hungarian-matched pairs.
          Reported for continuity with the paper, but under mismatch PUPE/FA
          are the metrics that mean something.

Rows and users are paired by a rectangular Hungarian assignment on Hamming
distance, the same criterion the paper uses in the square case. Decoding uses
the same per-K inference steps / dispatch / PRISM thresholds as
inference/eval_protocol.py (INFERENCE_STEPS, QUALITY_HEAD_PARAMS).

Data: <data_dir>/K<K>/{test_data.pt, H_matrix.pt}, e.g. test-only sets from
data/gen_data/generate_data_from_H.py (--num_train 0 --num_val 0), which moves
test-only runs off the seed-42 training stream by default.

Usage:
    # Full grid behind tab:load_mismatch (7x7, 2000 frames per cell)
    python inference/eval_k_mismatch.py \
        --checkpoint_dir checkpoints/protocol_scale \
        --data_dir data/gen_data/datasets/protocol_Eb10 \
        --model_K 2 3 4 5 6 7 8 --true_K 2 3 4 5 6 7 8 --max_samples 2000

    # Just the +/-1 mis-estimate band, which is the realistic failure mode
    python inference/eval_k_mismatch.py --model_K 4 --true_K 3 4 5

    tab:load_mismatch row K reads: K_hat=K-1 -> PUPE of cell (K-1)->K;
    K_hat=K -> PUPE of K->K; K_hat=K+1 -> PUPE / FA of (K+1)->K, set error
    = (misses + false alarms) / (K + K_hat).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.eval_protocol import (INFERENCE_STEPS, QUALITY_HEAD_PARAMS,
                                     load_all_checkpoints, sample_first_alone,
                                     sample_with_diffusion,
                                     sample_with_quality_head)


def decode(K_model, diffusion_model, qhead, Y, H, num_steps):
    """Run the inference strategy eval_protocol would pick for K_model."""
    backbone = diffusion_model.backbone
    if K_model <= 2:
        return sample_with_diffusion(diffusion_model, Y, H, num_steps=num_steps)
    if K_model <= 5 or qhead is None:
        return sample_first_alone(backbone, Y, H, num_steps=num_steps)
    thr = QUALITY_HEAD_PARAMS.get(K_model, {}).get(
        'thresholds', [0.99, 0.97, 0.94, 0.90])
    return sample_with_quality_head(backbone, qhead, Y, H,
                                    num_steps=num_steps, thresholds=thr)


def score_mismatched(preds, X0):
    """Rectangular Hungarian scoring.

    Args:
        preds: [B, K_model, N] emitted rows
        X0:    [B, K_true,  N] ground-truth users

    Returns:
        dict of running counters (hits, users, rows, matched symbol errors).
    """
    B, K_model, N = preds.shape
    K_true = X0.shape[1]
    n_pair = min(K_model, K_true)

    hits = 0            # true users exactly recovered
    sym_err = 0         # symbol errors over matched pairs
    sym_tot = 0

    for b in range(B):
        # cost[i, j] = Hamming(emitted row i, true user j)
        cost = (preds[b].unsqueeze(1) != X0[b].unsqueeze(0)).sum(-1)
        row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
        for i, j in zip(row_ind, col_ind):
            d = int(cost[i, j])
            sym_err += d
            sym_tot += N
            if d == 0:
                hits += 1

    return dict(hits=hits, users=B * K_true, rows=B * K_model,
                sym_err=sym_err, sym_tot=sym_tot, n_pair=n_pair)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', default='checkpoints/protocol_scale')
    p.add_argument('--data_dir', default='data/gen_data/datasets/protocol_Eb10')
    p.add_argument('--model_K', type=int, nargs='+', default=[2, 3, 4, 5])
    p.add_argument('--true_K', type=int, nargs='+', default=[2, 3, 4, 5])
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_samples', type=int, default=2000)
    p.add_argument('--save_results', default='logs/k_mismatch.json')
    args = p.parse_args()
    args.checkpoint_dir = os.path.expanduser(args.checkpoint_dir)
    args.data_dir = os.path.expanduser(args.data_dir)

    console = Console()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[bold]Device: {device}[/bold]")

    checkpoints = load_all_checkpoints(args.checkpoint_dir, device,
                                       K_list=args.model_K)

    # Load each true-K test set once.
    data, H = {}, None
    for K in args.true_K:
        path = Path(args.data_dir) / f"K{K}" / "test_data.pt"
        if not path.exists():
            console.print(f"[red]missing data for K_true={K}: {path}[/red]")
            continue
        d = torch.load(path, weights_only=True)
        data[K] = (d['Y'], d['gt_codewords'])
        if H is None:
            hp = Path(args.data_dir) / f"K{K}" / "H_matrix.pt"
            if hp.exists():
                hd = torch.load(hp, weights_only=True)
                H = hd.get('H_matrix', hd.get('H'))
    if H is None:
        console.print("[red]No H matrix found[/red]")
        return
    H = H.to(device)

    table = Table(show_header=True, header_style="bold")
    for col in ("K_model", "K_true", "n", "steps", "SER(matched)",
                "PUPE", "FA", "ms/sample"):
        table.add_column(col, justify="right")

    results = {}
    for Km in args.model_K:
        if Km not in checkpoints:
            console.print(f"[red]skip K_model={Km}: checkpoint missing[/red]")
            continue
        dm, qhead = checkpoints[Km]
        steps = INFERENCE_STEPS.get(Km, 16)

        for Kt in args.true_K:
            if Kt not in data:
                continue
            Y_all, X0_all = data[Kt]
            n = min(len(Y_all), args.max_samples)

            acc = dict(hits=0, users=0, rows=0, sym_err=0, sym_tot=0)
            t0 = time.perf_counter()
            for s in range(0, n, args.batch_size):
                e = min(s + args.batch_size, n)
                Y = Y_all[s:e].to(device).float()
                X0 = X0_all[s:e].to(device)
                with torch.no_grad():
                    preds = decode(Km, dm, qhead, Y, H, steps)
                r = score_mismatched(preds, X0)
                for k in acc:
                    acc[k] += r[k]
            dt = time.perf_counter() - t0

            ser = acc['sym_err'] / max(1, acc['sym_tot'])
            pupe = 1.0 - acc['hits'] / max(1, acc['users'])
            fa = 1.0 - acc['hits'] / max(1, acc['rows'])
            ms = dt * 1000 / max(1, n)

            mark = "[bold green]" if Km == Kt else ""
            end = "[/bold green]" if Km == Kt else ""
            table.add_row(f"{mark}{Km}{end}", f"{mark}{Kt}{end}", str(n),
                          str(steps), f"{ser:.4f}", f"{pupe:.4f}",
                          f"{fa:.4f}", f"{ms:.2f}")
            results[f"{Km}->{Kt}"] = dict(
                K_model=Km, K_true=Kt, n=n, steps=steps,
                ser_matched=ser, pupe=pupe, fa=fa, ms_per_sample=ms)

    console.print("\n[bold]Load-mismatch grid[/bold] "
                  "(green = matched, i.e. the paper's setting)")
    console.print(table)
    console.print("\nPUPE = fraction of true users never exactly recovered "
                  "(== paper CER on the diagonal)")
    console.print("FA   = fraction of emitted rows that are not a true "
                  "user's codeword")

    if args.save_results:
        out = Path(os.path.expanduser(args.save_results))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        console.print(f"\n[bold green]Saved: {out}[/bold green]")


if __name__ == '__main__':
    main()

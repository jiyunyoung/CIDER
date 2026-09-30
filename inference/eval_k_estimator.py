#!/usr/bin/env python3
"""
Front-end active-user-count (K) estimation + end-to-end composed pipeline.

Produces the estimator and Bank columns of Table tab:estimated_load
(P(K_hat=K), Bank MDR, Bank FAR). K is not assumed known: a front-end
estimator reads the number of active users from the raw received energy --
the same observation the AMP inner detector already consumes -- and
dispatches each frame to the matching per-K CIDER decoder ("Bank"). We report:

  (1) Estimator accuracy: confusion matrix of K_hat vs true K.
  (2) End-to-end PUPE (= MDR) and FAR of the composed pipeline (K_hat picks the
      decoder) against the ORACLE-K pipeline (true K picks the decoder). A
      small gap means the known-K assumption was never load-bearing.

The Single / Single-oracle columns of tab:estimated_load (one K=8 checkpoint,
dedup + top-K_hat pruning by summed slot-wise log-evidence) are NOT produced
by this script.

Estimator. Per slot the received energy obeys E[||y_l||^2] ~= Psym*K + n_s*s2
(unit-norm sensing columns, E|h|^2=1). The scalar feature

    e = mean_l ||y_l||^2

is strictly increasing in K, so we calibrate one centroid per K on a
calibration split and classify a test frame by the nearest centroid. This is
robust to the mild superlinearity from symbol collisions and needs no exact
knowledge of Psym or sigma2 at test time. To reflect that over-provisioning is
nearly free while under-provisioning is floored (see eval_k_mismatch.py), an
optional --round_up bias breaks ties toward the larger K.

Data: <data_dir>/K<K>/kest_data.pt from data/gen_data/gen_kest_data.py (per-
sample seeds far from the seed-42 training stream). The first --calib_frac of
each set calibrates the centroids; only the rest is decoded and scored.

Usage (tab:estimated_load: 3000 frames per K, 1500 calibrate / 1500 test):
    python data/gen_data/gen_kest_data.py \
        --h_matrix ~/data/demix/tiny_LDPC/H_matrix.pt \
        --out data/gen_data/datasets/kest_Eb10 \
        --K_list 2 3 4 5 6 7 8 --Eb 10 --n 3000
    python inference/eval_k_estimator.py \
        --data_dir data/gen_data/datasets/kest_Eb10 \
        --checkpoint_dir checkpoints/protocol_scale \
        --h_matrix ~/data/demix/tiny_LDPC/H_matrix.pt \
        --K_list 2 3 4 5 6 7 8
    # P(K_hat=K) = "K_hat acc", Bank MDR = "PUPE composed",
    # Bank FAR = "FAR composed" (also in the saved JSON under per_k).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.eval_protocol import INFERENCE_STEPS, load_all_checkpoints
from inference.eval_k_mismatch import decode, score_mismatched


def energy_feature(y_recv):
    """Mean per-slot received energy. y_recv: [n, L, n_s] complex -> [n]."""
    return (y_recv.abs() ** 2).sum(-1).mean(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/gen_data/datasets/kest_Eb10')
    p.add_argument('--checkpoint_dir', default='checkpoints/protocol_scale')
    p.add_argument('--h_matrix', default='~/data/demix/tiny_LDPC/H_matrix.pt',
                   help='H used when <data_dir>/K<K>/H_matrix.pt is absent '
                        '(gen_kest_data.py does not write one).')
    p.add_argument('--K_list', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument('--calib_frac', type=float, default=0.5,
                   help='Fraction of each K set used to calibrate centroids; '
                        'the rest is the disjoint test split.')
    p.add_argument('--round_up', action='store_true',
                   help='Bias the estimator up (classify to the nearest '
                        'centroid, then +1 if the frame sits above the '
                        'centroid) since over-provisioning is nearly free.')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--save_results', default='logs/k_estimator.json')
    args = p.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    args.checkpoint_dir = os.path.expanduser(args.checkpoint_dir)

    console = Console()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[bold]Device: {device}[/bold]")

    # ---- load data -------------------------------------------------------
    data, H = {}, None
    for K in args.K_list:
        f = Path(args.data_dir) / f"K{K}" / "kest_data.pt"
        if not f.exists():
            console.print(f"[red]missing {f}[/red]"); continue
        d = torch.load(f, weights_only=True)
        data[K] = d
    Ks = sorted(data)
    if not Ks:
        console.print("[red]no data loaded[/red]"); return

    # H comes from the checkpoint side; load it from the protocol data dir if
    # present, else from the tiny_LDPC H that generated these sets.
    for cand in [Path(args.data_dir) / f"K{Ks[0]}" / "H_matrix.pt",
                 Path(os.path.expanduser(args.h_matrix))]:
        if cand.exists():
            hd = torch.load(cand, weights_only=False)
            H = hd.get('H_matrix', hd.get('H')).to(device)
            break
    if H is None:
        console.print("[red]No H matrix found[/red]"); return

    # ---- calibrate centroids on the calibration split --------------------
    centroids = {}
    test_idx = {}
    for K in Ks:
        e = energy_feature(data[K]['y_recv']).numpy()
        n = len(e)
        n_cal = int(args.calib_frac * n)
        centroids[K] = float(np.mean(e[:n_cal]))
        test_idx[K] = np.arange(n_cal, n)
    console.print("\n[bold]Calibrated energy centroids[/bold]")
    for K in Ks:
        console.print(f"  K={K}: centroid={centroids[K]:8.2f}")

    cvals = np.array([centroids[K] for K in Ks])

    def classify(e):
        """Nearest-centroid K, optionally rounded up."""
        j = int(np.argmin(np.abs(cvals - e)))
        Khat = Ks[j]
        if args.round_up and e > cvals[j] and j + 1 < len(Ks):
            Khat = Ks[j + 1]
        return Khat

    checkpoints = load_all_checkpoints(args.checkpoint_dir, device,
                                       K_list=Ks)

    # ---- (1) confusion matrix + (2) composed vs oracle -------------------
    confusion = {kt: {kh: 0 for kh in Ks} for kt in Ks}
    # running PUPE counters
    comp = dict(hits=0, users=0, rows=0)   # composed pipeline (K_hat dispatch)
    orac = dict(hits=0, users=0)     # oracle-K dispatch
    per_k = {}

    t0 = time.perf_counter()
    for Kt in Ks:
        idx = test_idx[Kt]
        e_all = energy_feature(data[Kt]['y_recv']).numpy()
        Y_all = data[Kt]['Y']
        X0_all = data[Kt]['gt_codewords']

        khat_all = np.array([classify(e_all[i]) for i in idx])
        for kh in khat_all:
            confusion[Kt][kh] += 1

        c_hits = o_hits = c_rows = 0
        # Composed: group test frames by K_hat, decode each group with that
        # decoder, score against true users (rectangular).
        for Kh in np.unique(khat_all):
            sub = idx[khat_all == Kh]
            if Kh not in checkpoints:
                continue
            dm, qh = checkpoints[Kh]
            steps = INFERENCE_STEPS.get(int(Kh), 16)
            for s in range(0, len(sub), args.batch_size):
                b = sub[s:s + args.batch_size]
                Y = Y_all[b].to(device).float()
                X0 = X0_all[b].to(device)
                with torch.no_grad():
                    preds = decode(int(Kh), dm, qh, Y, H, steps)
                r = score_mismatched(preds, X0)
                c_hits += r['hits']
                c_rows += r['rows']

        # Oracle: decode all test frames with the true-K decoder.
        if Kt in checkpoints:
            dm, qh = checkpoints[Kt]
            steps = INFERENCE_STEPS.get(Kt, 16)
            for s in range(0, len(idx), args.batch_size):
                b = idx[s:s + args.batch_size]
                Y = Y_all[b].to(device).float()
                X0 = X0_all[b].to(device)
                with torch.no_grad():
                    preds = decode(Kt, dm, qh, Y, H, steps)
                o_hits += score_mismatched(preds, X0)['hits']

        n_users = len(idx) * Kt
        comp['hits'] += c_hits; comp['users'] += n_users
        comp['rows'] += c_rows
        orac['hits'] += o_hits; orac['users'] += n_users
        per_k[Kt] = dict(
            n=len(idx),
            acc=float(np.mean(khat_all == Kt)),
            pupe_composed=1.0 - c_hits / max(1, n_users),
            far_composed=1.0 - c_hits / max(1, c_rows),
            pupe_oracle=1.0 - o_hits / max(1, n_users),
        )

    dt = time.perf_counter() - t0

    # ---- report ----------------------------------------------------------
    console.print("\n[bold]Confusion matrix[/bold] (rows = true K, cols = K_hat)")
    ct = Table(show_header=True, header_style="bold")
    ct.add_column("true\\hat", justify="right")
    for kh in Ks:
        ct.add_column(str(kh), justify="right")
    ct.add_column("acc", justify="right")
    for kt in Ks:
        row = [str(kt)]
        tot = sum(confusion[kt].values())
        for kh in Ks:
            c = confusion[kt][kh]
            cell = str(c) if c == 0 else f"[bold]{c}[/bold]"
            row.append(cell)
        row.append(f"{confusion[kt][kt]/max(1,tot):.3f}")
        ct.add_row(*row)
    console.print(ct)

    console.print("\n[bold]Composed pipeline vs oracle-K[/bold] (PUPE, lower better)")
    pt = Table(show_header=True, header_style="bold")
    for col in ("K", "n", "K_hat acc", "PUPE composed", "FAR composed",
                "PUPE oracle", "gap"):
        pt.add_column(col, justify="right")
    for Kt in Ks:
        r = per_k[Kt]
        pt.add_row(str(Kt), str(r['n']), f"{r['acc']:.3f}",
                   f"{r['pupe_composed']:.4f}", f"{r['far_composed']:.4f}",
                   f"{r['pupe_oracle']:.4f}",
                   f"{r['pupe_composed'] - r['pupe_oracle']:+.4f}")
    comp_pupe = 1.0 - comp['hits'] / max(1, comp['users'])
    comp_far = 1.0 - comp['hits'] / max(1, comp['rows'])
    orac_pupe = 1.0 - orac['hits'] / max(1, orac['users'])
    overall_acc = np.mean([per_k[k]['acc'] for k in Ks])
    pt.add_row("[bold]all[/bold]", str(comp['users']), f"{overall_acc:.3f}",
               f"[bold]{comp_pupe:.4f}[/bold]", f"[bold]{comp_far:.4f}[/bold]",
               f"[bold]{orac_pupe:.4f}[/bold]",
               f"[bold]{comp_pupe - orac_pupe:+.4f}[/bold]")
    console.print(pt)
    console.print(f"\nround_up bias: {args.round_up}   "
                  f"eval time: {dt:.1f}s")

    if args.save_results:
        out = Path(os.path.expanduser(args.save_results))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(
            centroids=centroids, confusion=confusion, per_k=per_k,
            overall=dict(pupe_composed=comp_pupe, far_composed=comp_far,
                         pupe_oracle=orac_pupe,
                         khat_acc=float(overall_acc), round_up=args.round_up),
        ), indent=2))
        console.print(f"[bold green]Saved: {out}[/bold green]")


if __name__ == '__main__':
    main()

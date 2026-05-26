#!/usr/bin/env python3
"""
Per-K isolated evaluation, reusing eval_protocol.py's inference config.

For each K in --K_list, loads the
K-specific test set and decodes every sample with the exact same checkpoint,
inference steps, dispatch strategy, and quality-head thresholds that
eval_protocol.py would use for that K.

Usage:
    python inference/eval_per_k.py --K_list 6 7 8
    python inference/eval_per_k.py --K_list 6 7 8 --max_samples 2000
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           TaskProgressColumn, TextColumn, TimeRemainingColumn)
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.eval_protocol import (INFERENCE_STEPS, QUALITY_HEAD_PARAMS,
                                     evaluate_batch, load_all_checkpoints)




def load_data_for_ks(data_dir, K_list):
    """Load test data only for K's in K_list — avoids load_all_data's
    hardcoded K=1..8 loop."""
    data = {}
    H = None
    data_dir = Path(data_dir)
    for K in K_list:
        k_dir = data_dir / f"K{K}"
        data_path = k_dir / "test_data.pt"
        if not data_path.exists():
            print(f"Warning: No data for K={K} at {data_path}")
            continue
        print(f"Loading K={K} data from {data_path}")
        loaded = torch.load(data_path, weights_only=True)
        data[K] = (loaded['Y'], loaded['gt_codewords'])
        if H is None:
            H_path = k_dir / "H_matrix.pt"
            if H_path.exists():
                H_data = torch.load(H_path, weights_only=True)
                H = H_data.get('H_matrix', H_data.get('H', None))
    return data, H


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', default='checkpoints/protocol_scale')
    p.add_argument('--data_dir', default='data/gen_data/datasets/protocol_Eb10')
    p.add_argument('--K_list', type=int, nargs='+', default=[6, 7, 8])
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_samples', type=int, default=None,
                   help='Cap samples per K (default: use entire test set)')
    p.add_argument('--no_quality_head', action='store_true',
                   help='Force K>=6 to use first-reveal-alone (no remasking). '
                        'Achieved by dropping the quality head after load — '
                        'evaluate_batch then takes the qhead-is-None branch.')
    p.add_argument('--inference_steps', type=int, nargs='+', default=None,
                   help='Override INFERENCE_STEPS for the K values being run. '
                        'Pass a single value to broadcast to all K (e.g. '
                        '--inference_steps 100), or one value per --K_list '
                        'entry in the same order (e.g. --K_list 6 7 8 '
                        '--inference_steps 80 100 120).')
    p.add_argument('--save_results', type=str, default=None,
                   help='Path to save JSON results')
    args = p.parse_args()

    console = Console()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[bold]Device: {device}[/bold]")

    console.print(f"\n[bold]Loading checkpoints for K={args.K_list}...[/bold]")
    try:
        checkpoints = load_all_checkpoints(args.checkpoint_dir, device,
                                           K_list=args.K_list)
    except TypeError:
        # Older eval_protocol.py without K_list support — fall back to
        # K_range and filter the result dict.
        K_min, K_max = min(args.K_list), max(args.K_list) + 1
        checkpoints = load_all_checkpoints(args.checkpoint_dir, device,
                                           K_range=(K_min, K_max))
        checkpoints = {K: v for K, v in checkpoints.items() if K in args.K_list}

    if args.no_quality_head:
        console.print("[yellow]--no_quality_head: dropping loaded quality heads "
                      "→ K>=6 will use sample_first_alone[/yellow]")
        checkpoints = {K: (model, None) for K, (model, _) in checkpoints.items()}

    if args.inference_steps is not None:
        if len(args.inference_steps) == 1:
            steps_per_k = {K: args.inference_steps[0] for K in args.K_list}
        elif len(args.inference_steps) == len(args.K_list):
            steps_per_k = dict(zip(args.K_list, args.inference_steps))
        else:
            console.print(f"[red]--inference_steps length must be 1 or "
                          f"{len(args.K_list)} (got {len(args.inference_steps)})[/red]")
            return
        # Mutate the imported dict — evaluate_batch reads INFERENCE_STEPS.get(K,...)
        # at call time, so the override takes effect for this run only.
        for K, n in steps_per_k.items():
            old = INFERENCE_STEPS.get(K, '?')
            INFERENCE_STEPS[K] = n
            console.print(f"[yellow]--inference_steps: K={K} {old} -> {n}[/yellow]")

    console.print(f"\n[bold]Loading data...[/bold]")
    data, H = load_data_for_ks(args.data_dir, args.K_list)
    if H is None:
        console.print("[red]No H matrix found![/red]")
        return
    H = H.to(device)

    # Show config so the user can confirm it matches eval_protocol
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print("[bold cyan]ISOLATED PER-K EVALUATION[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"Checkpoints: {args.checkpoint_dir}")
    console.print(f"Data: {args.data_dir}")
    console.print(f"K list: {args.K_list}")
    console.print(f"Batch size: {args.batch_size}")
    if args.max_samples is None:
        console.print(f"Sampling: [bold]full test set, sequential[/bold] "
                      f"(every sample used exactly once, no random sampling)")
    else:
        console.print(f"Sampling: first {args.max_samples} samples per K (capped)")
    for K in args.K_list:
        steps = INFERENCE_STEPS.get(K, 16)
        qh = QUALITY_HEAD_PARAMS.get(K, {})
        has_qh = K in checkpoints and checkpoints[K][1] is not None
        strat = ('quality_head' if (K >= 6 and has_qh)
                 else 'first_alone' if K >= 3
                 else 'standard')
        n_avail = len(data[K][0]) if K in data else 0
        n_run = min(n_avail, args.max_samples) if args.max_samples else n_avail
        console.print(f"  K={K}: steps={steps}, strategy={strat}, "
                      f"qhead_params={qh}, samples={n_run}/{n_avail}")

    results = {}
    table = Table(show_header=True, header_style="bold")
    table.add_column("K", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Strategy")
    table.add_column("Steps", justify="right")
    table.add_column("SER", justify="right")
    table.add_column("CER (PUPE)", justify="right")
    table.add_column("ms/sample", justify="right")

    for K in args.K_list:
        if K not in data:
            console.print(f"[red]Skipping K={K}: data missing[/red]")
            continue
        if K not in checkpoints:
            console.print(f"[red]Skipping K={K}: checkpoint missing[/red]")
            continue

        Y_all, X0_all = data[K]
        N_total = len(Y_all)
        if args.max_samples is not None:
            N_total = min(N_total, args.max_samples)

        cs = ts = cw = tw = 0
        num_batches = (N_total + args.batch_size - 1) // args.batch_size
        t0 = time.perf_counter()

        with Progress(
            TextColumn(f"[bold blue]K={K}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("batches", total=num_batches)
            for s in range(0, N_total, args.batch_size):
                e = min(s + args.batch_size, N_total)
                res = evaluate_batch(K, Y_all[s:e], X0_all[s:e], H,
                                     checkpoints, device)
                if res is None:
                    progress.update(task, advance=1)
                    continue
                cs_b, ts_b, cw_b, tw_b = res
                cs += cs_b; ts += ts_b; cw += cw_b; tw += tw_b
                progress.update(task, advance=1)

        dt = time.perf_counter() - t0
        ser = 1.0 - cs / ts if ts else 0.0
        cer = 1.0 - cw / tw if tw else 0.0
        ms_per = dt * 1000 / max(1, N_total)
        steps = INFERENCE_STEPS.get(K, 16)
        has_qh = checkpoints[K][1] is not None
        strat = ('quality_head' if (K >= 6 and has_qh)
                 else 'first_alone' if K >= 3
                 else 'standard')

        table.add_row(str(K), str(N_total), strat, str(steps),
                      f"{ser:.6f}", f"{cer:.6f}", f"{ms_per:.2f}")
        results[K] = {
            'samples': N_total, 'strategy': strat, 'steps': steps,
            'ser': ser, 'cer': cer, 'ms_per_sample': ms_per,
        }

    console.print(f"\n[bold]Results:[/bold]")
    console.print(table)

    if args.save_results:
        import json
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        console.print(f"\n[bold green]Saved: {save_path}[/bold green]")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generalization / code-transfer eval with ON-THE-FLY data from any saved H.

Produces the L=72 and L=96 rows of the "Transfer CER" column of the main-text
transfer table (tab:response_transfer). Those lengths have no cached test split,
so their evidence is generated here; the L<=48 rows come from
inference/eval_code_transfer.py.

Unlike inference/eval_code_transfer.py (which reads a pre-baked test_data.pt per
target), this script GENERATES fresh test data from each target's H_matrix.pt via
QaryOnTheFlyDataset -- so it works for any parity-check matrix, with controllable
sample count, Eb/N0, and seed, and no dependence on shipped test splits.

For each target it: (1) loads H (-> L, M), (2) rebuilds the CIDER model at the
target geometry and loads the source checkpoint's weights (valid because no CIDER
parameter depends on N/M/H -- see eval_code_transfer.py), (3) draws num_samples
frames on the fly, (4) reports SER / PUPE (rectangular Hungarian). PUPE is the
paper's CER.

Test frames: sample idx is seeded with --seed + idx (per-sample CPU seeding in
QaryOnTheFlyDataset); the sensing matrix A is always the seed-42 one used for
training. The default --seed 199999 is the test stream of data_onthefly; the
tiny checkpoint was trained on the stored seed-42 tiny_LDPC set, so neither seed
replays its training frames.

Paper rows (tab:response_transfer, L=72/96, T=40/60, 256 frames each). The
targets are the Moebius-ladder (2,3)-regular codes from construct_H.py:
    mkdir -p ~/data/demix/mobius_L72 ~/data/demix/mobius_L96
    python data/gen_data/construct_H.py --q 64 --L 72 --M 48 --d_v 2 --d_c 3 \
        --seed 42 --output ~/data/demix/mobius_L72/H_matrix.pt
    python data/gen_data/construct_H.py --q 64 --L 96 --M 64 --d_v 2 --d_c 3 \
        --seed 42 --output ~/data/demix/mobius_L96/H_matrix.pt
    python inference/eval_transfer_onthefly.py \
        --checkpoint checkpoints/tiny_ldpc_tiny_cider/best_model.ckpt \
        --targets ~/data/demix/mobius_L72 ~/data/demix/mobius_L96 \
        --num_steps 40 60 --num_samples 256 --seed 42 --Eb_dB 10 --K 2
(--seed 42 selects the same 256 frames the paper rows were scored on. Those
rows were measured with the fast sampler at batch size 8, whose reveal-order
randomness differs from this script's, so a 256-frame CER here can differ from
the table by sampling noise.)

--num_steps: one value applied to all targets, or one per target (the reveal
schedule should scale with L; matching the trained per-L values is recommended).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from diffusion import Diffusion
from data.data_onthefly import QaryOnTheFlyDataset
from inference.eval_code_transfer import (
    load_source_config, build_config, build_transfer_model,
)
from inference.eval_protocol import sample_first_alone
from models.cider_iterative_v2 import _assign, _hamming_cost_matrix


@torch.no_grad()
def evaluate_perk(model, loader, H, device, num_steps, K):
    """Score SER/PUPE using the DEPLOYED decode strategy for this load:
       K<=2 -> plain masked-diffusion sampling; K>2 -> first-reveal-alone.
    (Quality-head remasking is intentionally not used -- first-reveal-alone is
    the strategy for all K>2 in this study.)"""
    model.H = H
    n_sym = n_sym_err = n_row = n_row_err = 0
    t0 = time.time()
    for batch in loader:
        Y, gt = batch[0].to(device).float(), batch[1].to(device)
        if K <= 2:
            pred = model._sample(Y, num_steps=num_steps, use_remasking=False,
                                 random_slot_first=True)
        else:
            pred = sample_first_alone(model.backbone, Y, H, num_steps=num_steps)
        if pred.dim() == 4:
            pred = pred.argmax(dim=-1)
        col = _assign(_hamming_cost_matrix(pred, gt.long()))
        B, Kk, N = pred.shape
        gt_perm = torch.gather(gt.long(), 1, col.unsqueeze(-1).expand(B, Kk, N))
        matches = (pred == gt_perm)
        n_sym += matches.numel(); n_sym_err += (~matches).sum().item()
        n_row += B * Kk; n_row_err += (~matches.all(dim=-1)).sum().item()
    return {'SER': n_sym_err / max(1, n_sym), 'PUPE': n_row_err / max(1, n_row),
            'rows': n_row, 'sec': time.time() - t0}


def resolve_H(path_str):
    """Accept a dir containing H_matrix.pt, or an H_matrix.pt file. -> (H, dir)."""
    p = Path(os.path.expanduser(path_str))
    hp = p / 'H_matrix.pt' if p.is_dir() else p
    if not hp.exists():
        raise FileNotFoundError(f"no H_matrix.pt at {hp}")
    d = torch.load(hp, map_location='cpu', weights_only=True)
    H = d.get('H_matrix', d.get('H'))
    return H.long(), hp.parent


def target_config(src_cfg, base_size, base_model, Q, N, K, M):
    """Full config with the source architecture and the TARGET geometry."""
    cfg = build_config('tiny_ldpc', base_size, base_model, [])   # any valid base
    OmegaConf.set_struct(cfg, False)
    cfg.model = src_cfg.model                    # source arch (D, layers, ...)
    cfg.data.Q, cfg.data.N, cfg.data.K_max, cfg.data.M = int(Q), int(N), int(K), int(M)
    OmegaConf.set_struct(cfg, True)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--targets', nargs='+', required=True,
                   help='dirs containing H_matrix.pt (or H_matrix.pt paths)')
    p.add_argument('--K', type=int, default=2)
    p.add_argument('--Eb_dB', type=float, default=10.0)
    p.add_argument('--n_s', type=int, default=24, help='inner sensing length')
    p.add_argument('--sigma2', type=float, default=1.0)
    p.add_argument('--num_samples', type=int, default=5000)
    p.add_argument('--num_steps', type=int, nargs='+', default=None,
                   help='one value for all targets, or one per target')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--seed', type=int, default=199999, help='fixed test seed')
    p.add_argument('--size', default='tiny', help='base config only (geometry '
                   'is overridden from H; arch comes from the checkpoint)')
    p.add_argument('--model', default='cider')
    p.add_argument('--Q', type=int, default=64)
    args = p.parse_args()

    console = Console()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[bold]Device: {device}[/bold]")

    ckpt = torch.load(os.path.expanduser(args.checkpoint), map_location='cpu',
                      weights_only=False)
    src_cfg = load_source_config(ckpt)
    console.print(f"source ckpt: {args.checkpoint}")
    console.print(f"  arch: D={src_cfg.model.get('D_model')} "
                  f"layers={src_cfg.model.get('num_layers')} "
                  f"(geometry taken from each target's H)\n")

    steps = args.num_steps
    if steps and len(steps) not in (1, len(args.targets)):
        raise SystemExit("--num_steps must be one value or one per target")

    table = Table(show_header=True, header_style="bold")
    for c in ("target", "L", "M", "steps", "n", "SER", "PUPE", "sec"):
        table.add_column(c, justify="right")

    for i, tgt in enumerate(args.targets):
        H, tgt_dir = resolve_H(tgt)
        M, N = H.shape
        T = (steps[0] if len(steps) == 1 else steps[i]) if steps \
            else int(src_cfg.model.get('inference_steps', 16))

        cfg = target_config(src_cfg, args.size, args.model, args.Q, N, args.K, M)
        model, _ = build_transfer_model(src_cfg, cfg, ckpt, device)

        ds = QaryOnTheFlyDataset(
            str(tgt_dir / 'H_matrix.pt'), K=args.K, Eb_dB=args.Eb_dB,
            n_s=args.n_s, sigma2=args.sigma2, num_samples=args.num_samples,
            fixed_seed=args.seed)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

        res = evaluate_perk(model, loader, H.to(device), device, T, args.K)
        name = tgt_dir.name
        table.add_row(name, str(N), str(M), str(T), str(res['rows'] // args.K),
                      f"{res['SER']:.5f}", f"{res['PUPE']:.5f}", f"{res['sec']:.1f}")
        console.print(f"  done {name}: L={N} M={M} T={T} "
                      f"PUPE={res['PUPE']:.5f}")

    console.print()
    console.print(table)


if __name__ == '__main__':
    main()

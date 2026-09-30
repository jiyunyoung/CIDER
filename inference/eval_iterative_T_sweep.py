#!/usr/bin/env python3
"""
Test-time iteration sweep for the non-diffusion iterative A+B controls
(Table 2a, tab:app_full_results_ablation_main, rows "Iter. A+B").

Two iterative controls share this script; the class is picked from the
checkpoint's model.backbone_type:
    cider_iterative     models/cider_iterative.py    (the "Iter. A+B" rows of
                        Table 2a; unconditioned A+B blocks, soft feedback)
    cider_iterative_v2  models/cider_iterative_v2.py (parameter-matched,
                        iteration-conditioned control wrapping models.cider.DiMP)

T is swept at eval time without retraining, so the comparison against CIDER
cannot be dismissed as "you ran the baseline for too few iterations". The
saturation curve is also reported directly, since where the iterative refiner
stops improving is itself the interesting part of the ablation.

Metrics (Hungarian-matched, identical definitions to the paper):
    SER  = symbol error rate
    PUPE = per-user probability of error = fraction of matched rows with any
           symbol error (the paper's CER)

Reproducing Table 2a (tiny_ldpc test split, 15k samples) -- BEST-EFFORT
RECONSTRUCTION. The rows "Iter. A+B, N_iter=1" (.1039/.4093) and
"N_iter=12" (.3863/.9989) come from models/cider_iterative.py, but no
checkpoint, log or job file of those runs survives, so the exact recipe is
unverified. The camera-ready appendix calls them "one-update" /
"twelve-update" iterative A+B; the pre-camera-ready draft labels them
"L=1 model, 12 iters" and "L=12 model", i.e. the N_iter=1 row may be a model
trained with one pass (model.num_iters=1, or possibly size.num_layers=1) but
EVALUATED at 12 iterations. Candidate commands:
    # N_iter=12 row: train and evaluate with 12 passes
    python main.py mode=train data=tiny_ldpc size=tiny model=cider_iterative \
        model.num_iters=12 experiment_name=tiny_ldpc_tiny_cider_iterative_T12
    python inference/eval_iterative_T_sweep.py \
        --checkpoint checkpoints/tiny_ldpc_tiny_cider_iterative_T12/best_model.ckpt \
        --T_list 12
    # N_iter=1 row, reading (a) "one-update": train and evaluate with 1 pass
    python main.py mode=train data=tiny_ldpc size=tiny model=cider_iterative \
        model.num_iters=1 experiment_name=tiny_ldpc_tiny_cider_iterative_T1
    python inference/eval_iterative_T_sweep.py \
        --checkpoint checkpoints/tiny_ldpc_tiny_cider_iterative_T1/best_model.ckpt \
        --T_list 1
    # N_iter=1 row, reading (b) "L=1 model, 12 iters": the same 1-pass model
    # (or one trained with size.num_layers=1) evaluated at 12 iterations
    python inference/eval_iterative_T_sweep.py \
        --checkpoint checkpoints/tiny_ldpc_tiny_cider_iterative_T1/best_model.ckpt \
        --T_list 1 12
Confirm which reading matches the paper before quoting reproduced numbers.

Full sweep of the v2 control (trained with main.py ... model=cider_iterative_v2):
    python inference/eval_iterative_T_sweep.py \
        --checkpoint checkpoints/tiny_ldpc_tiny_cider_iterative_v2_s0/best_model.ckpt \
        --data tiny_ldpc --size tiny --T_list 1 2 4 8 12 16 20 32

Note: Lightning appends -v1/-v2 to re-runs in the same directory, so
best_model.ckpt is the OLDEST run there; pass the file you mean explicitly.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloader import get_dataloaders
from main import load_H_matrix
from models.cider_iterative import DiMPIterative
from models.cider_iterative_v2 import (DiMPIterativeV2, _assign,
                                       _hamming_cost_matrix)

# Iterative controls whose forward takes num_iters=T.
ITERATIVE_MODELS = {
    'cider_iterative': DiMPIterative,
    'cider_iterative_v2': DiMPIterativeV2,
}


def build_config(data: str, size: str, model: str, overrides):
    from hydra import compose, initialize_config_dir
    cfg_dir = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(
            config_name="config",
            overrides=[f"data={data}", f"size={size}", f"model={model}", *overrides],
        )


@torch.no_grad()
def evaluate(model, loader, T, device, max_batches=None):
    n_sym = n_sym_err = n_row = n_row_err = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        Y, gt = batch[0].to(device), batch[1].to(device)
        pred = model(Y, num_iters=T).argmax(dim=-1)

        col = _assign(_hamming_cost_matrix(pred, gt.long()))
        B, K, N = pred.shape
        gt_perm = torch.gather(gt.long(), 1, col.unsqueeze(-1).expand(B, K, N))
        matches = (pred == gt_perm)

        n_sym += matches.numel()
        n_sym_err += (~matches).sum().item()
        n_row += B * K
        n_row_err += (~matches.all(dim=-1)).sum().item()

    return {
        'SER': n_sym_err / max(1, n_sym),
        'PUPE': n_row_err / max(1, n_row),
        'n_rows': n_row,
        'sec': time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='tiny_ldpc')
    p.add_argument('--size', default='tiny')
    p.add_argument('--model', default='cider_iterative_v2',
                   help='config used before the checkpoint config is applied; '
                        'the model class always follows the checkpoint')
    p.add_argument('--T_list', type=int, nargs='+',
                   default=[1, 2, 4, 8, 12, 16, 20, 32])
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--max_batches', type=int, default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--override', nargs='*', default=[],
                   help='extra hydra overrides, e.g. data.K_true=5')
    args = p.parse_args()
    args.checkpoint = os.path.expanduser(args.checkpoint)

    # mode=test + eval_split makes get_dataloaders load exactly the split asked for.
    config = build_config(args.data, args.size, args.model,
                          ["mode=test", f"eval_split={args.split}",
                           f"training.batch_size={args.batch_size}", *args.override])

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Architecture always comes from the checkpoint, so a sweep can never be run
    # against a mismatched config.
    if 'hyper_parameters' in ckpt and 'config' in ckpt['hyper_parameters']:
        ckpt_cfg = OmegaConf.create(ckpt['hyper_parameters']['config'])
        OmegaConf.set_struct(config, False)
        config.model = ckpt_cfg.model
        config.data = ckpt_cfg.data
        OmegaConf.set_struct(config, True)

    backbone = config.model.get('backbone_type', args.model)
    if backbone not in ITERATIVE_MODELS:
        raise SystemExit(f"backbone_type={backbone} is not an iterative control "
                         f"({', '.join(ITERATIVE_MODELS)})")
    model = ITERATIVE_MODELS[backbone](config)
    missing, unexpected = model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    H = load_H_matrix(config).to(device)
    model = model.to(device).eval()
    model.set_H_matrix(H)

    train_loader, val_loader, test_loader = get_dataloaders(config)
    loader = test_loader if args.split == 'test' else val_loader
    if loader is None:
        raise SystemExit(f"no usable {args.split} loader for data={args.data}")

    print("=" * 68)
    print(f"Iterative A+B test-time iteration sweep  |  {Path(args.checkpoint).parent.name}")
    print(f"model={backbone} data={args.data} size={args.size} split={args.split} "
          f"trained_T={model.num_iters} feedback={getattr(model, 'feedback', 'soft')}")
    print("=" * 68)
    print(f"{'T':>4}  {'SER':>10}  {'PUPE':>10}  {'rows':>8}  {'sec':>7}")
    print("-" * 68)

    best = None
    for T in args.T_list:
        r = evaluate(model, loader, T, device, args.max_batches)
        print(f"{T:>4}  {r['SER']:>10.5f}  {r['PUPE']:>10.5f}  "
              f"{r['n_rows']:>8}  {r['sec']:>7.1f}")
        if best is None or r['PUPE'] < best[1]['PUPE']:
            best = (T, r)

    print("-" * 68)
    print(f"best PUPE {best[1]['PUPE']:.5f} at T={best[0]} "
          f"(trained_T={model.num_iters})")


if __name__ == '__main__':
    main()

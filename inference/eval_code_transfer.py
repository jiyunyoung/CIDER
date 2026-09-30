#!/usr/bin/env python3
"""
Zero-shot code transfer: evaluate ONE trained CIDER checkpoint on code
instances, code lengths, and code families it was never trained on.

Produces the L=12/18/24/48 rows of the "Transfer CER" column of the main-text
transfer table (tab:response_transfer: training LDPC, PEG-LDPC, tree code, and
LDPC at L=18/24/48). The L=72/96 rows have no cached test split and come from
inference/eval_transfer_onthefly.py.

Why this is even possible: no parameter in models/cider.py depends on N, M, or
H. The parameter set is a function of (Q, K, D) alone --

    W_base       [(Q+1) x D]      slot_init  [K x D]
    output_proj  [D -> Q]         everything else is D-dimensional

-- while the parity-check matrix enters only at runtime, through
_build_tanner_cache(H), which rebuilds the Tanner neighbour/coefficient lists
from whatever H is handed to it. Module B is therefore a graph-conditioned
message-passing operator, not a code-specific function, exactly as in classical
BP. So the model is REBUILT here at the target's (N, M) and the source weights
load into it unchanged; there is no surgery and no retraining.

What that does NOT make free: Q cannot transfer (W_base, output_proj and
perm_table are all Q-sized), and K must match (slot_init). Both are checked and
refused loudly rather than silently producing nonsense. Degree profile,
GF-coefficient statistics and the reveal schedule's effective difficulty all
still shift with the target code -- that shift is precisely what this script
measures.

Decoding uses the K=2 random-row first reveal (_sample(...,
random_slot_first=True)), with only the refinement count T changed per target.
CER in the paper = PUPE here (per-row error after Hungarian matching).

A target is either a data config name (configs/data/<name>.yaml) or a data
directory holding H_matrix.pt and test_data.pt (e.g. ~/data/demix/tiny_LDPC_PEG);
for a directory the geometry (Q, N, M, K) is read from the files.

Datasets: tiny_LDPC / small_LDPC / moderate_LDPC / large_LDPC from
data/gen_data/ldpc_{tiny,small,moderate,large}.sh, tiny_LDPC_PEG from
data/gen_data/ldpc_tiny_peg.sh, tiny_tree from data/gen_data/construct_tree_code.py
+ generate_data_from_H.py. Only the test split is read.

Paper command (tab:response_transfer, T = 12/12/12/16/20/28):
    python inference/eval_code_transfer.py \
        --checkpoint checkpoints/tiny_ldpc_tiny_cider/best_model.ckpt \
        --targets tiny_ldpc ~/data/demix/tiny_LDPC_PEG tiny_tree \
                  small_ldpc moderate_ldpc large_ldpc \
        --num_steps 12 12 12 16 20 28

The row whose target == the checkpoint's own training set is the matched
reference; every other row is zero-shot.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion import Diffusion
from main import load_H_matrix
# Hungarian helpers live with the A2 ablation; identical definitions, reused
# here so the two scripts cannot drift apart on how PUPE is counted.
from models.cider_iterative_v2 import _assign, _hamming_cost_matrix


def load_test_split(cfg, batch_size):
    """Test tensors straight from disk.

    Deliberately NOT dataloader.get_dataloaders: several of the transfer targets
    (small/moderate/large_LDPC, tiny_tree) ship only test_data.pt, and the shared
    loader insists on a train split that does not exist for them. Transfer eval
    never needs train data anyway.
    """
    from torch.utils.data import DataLoader, TensorDataset
    path = Path(os.path.expanduser(cfg.data_dir)) / 'test_data.pt'
    if not path.exists():
        raise FileNotFoundError(f"no test split at {path}")
    d = torch.load(path, map_location='cpu', weights_only=True)
    Y, gt = d['Y'], d['gt_codewords']            # [S, N, Q], [S, K, N]
    return DataLoader(TensorDataset(Y, gt), batch_size=batch_size, shuffle=False)


def build_config(data: str, size: str, model: str, overrides):
    from hydra import compose, initialize_config_dir
    cfg_dir = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(
            config_name="config",
            overrides=[f"data={data}", f"size={size}", f"model={model}", *overrides],
        )


def target_label(target: str) -> str:
    """Row label: the config name, or the lower-cased name of a data directory."""
    tgt_dir = Path(os.path.expanduser(target))
    return tgt_dir.name.lower() if tgt_dir.is_dir() else target


def build_target_config(target: str, size: str, model: str, overrides):
    """Config for one --targets entry: a data config name, or a data directory.

    A directory needs no config file: it is composed on the tiny_ldpc base and
    its data_dir/name/geometry are replaced with what the files say. The name
    is the lower-cased directory name, so ~/data/demix/tiny_LDPC is recognised
    as the matched target of a tiny_ldpc checkpoint.
    """
    tgt_dir = Path(os.path.expanduser(target))
    if not tgt_dir.is_dir():
        return build_config(target, size, model, overrides)

    h = torch.load(tgt_dir / 'H_matrix.pt', map_location='cpu', weights_only=True)
    M, N = h['H_matrix'].shape
    meta = tgt_dir / 'dataset_metadata.json'
    if meta.exists():
        with open(meta) as f:
            K = int(json.load(f)['data_params']['K'])
    else:
        d = torch.load(tgt_dir / 'test_data.pt', map_location='cpu', weights_only=True)
        K = int(d['gt_codewords'].shape[1])

    cfg = build_config('tiny_ldpc', size, model, overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.data_dir = str(tgt_dir)
    cfg.data.name = target_label(target)
    cfg.data.Q, cfg.data.N, cfg.data.M = int(h['q']), int(N), int(M)
    cfg.data.K_true = cfg.data.K_max = K
    OmegaConf.set_struct(cfg, True)
    return cfg


def load_source_config(ckpt):
    """Architecture config as trained. Without it we cannot rebuild the backbone."""
    hp = ckpt.get('hyper_parameters', {})
    if 'config' not in hp:
        raise SystemExit(
            "checkpoint has no hyper_parameters.config; cannot recover the source "
            "architecture; this script needs a checkpoint saved by main.py.")
    return OmegaConf.create(hp['config'])


def build_transfer_model(src_cfg, tgt_cfg, ckpt, device):
    """Source architecture, target code geometry.

    config.model comes from the checkpoint (D_model, layers, heads,
    inference_steps, backbone_type); config.data stays the TARGET's (Q, N, K, M).
    Because no parameter is N- or M-shaped, the resulting state_dict is an exact
    match for the source weights.
    """
    cfg = tgt_cfg.copy()
    OmegaConf.set_struct(cfg, False)
    cfg.model = src_cfg.model
    OmegaConf.set_struct(cfg, True)

    model = Diffusion(cfg)
    state_dict = {k: v for k, v in ckpt.get('state_dict', ckpt).items() if k != 'H'}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # A parameter silently left at random init would look like a transfer
    # failure. Refuse instead of reporting a meaningless number.
    real_missing = [k for k in missing if not k.startswith('backbone._')]
    if real_missing:
        raise SystemExit(
            f"{len(real_missing)} parameters missing from the checkpoint, e.g. "
            f"{real_missing[:5]}. This is NOT a transfer result -- the shapes "
            f"disagree, so Q or K almost certainly differs.")

    # EMA weights, matching main.py's test path.
    if 'ema' in ckpt and 'shadow_params' in ckpt['ema']:
        shadow = ckpt['ema']['shadow_params']
        n = 0
        for name, param in model.backbone.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name])
                n += 1
        for name, buf in model.backbone.named_buffers():
            if name in shadow:
                buf.copy_(shadow[name])
        print(f"    loaded {n} EMA parameters")

    return model.to(device).eval(), unexpected


@torch.no_grad()
def evaluate(model, loader, H, device, num_steps, max_batches=None):
    model.H = H
    n_sym = n_sym_err = n_row = n_row_err = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        Y, gt = batch[0].to(device), batch[1].to(device)

        pred = model._sample(Y.float(), num_steps=num_steps,
                             use_remasking=False, random_slot_first=True)
        if pred.dim() == 4:                      # logits rather than tokens
            pred = pred.argmax(dim=-1)

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
        'rows': n_row,
        'sec': time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--targets', nargs='+', required=True,
                   help='data config names, or data directories holding '
                        'H_matrix.pt + test_data.pt, to transfer onto')
    p.add_argument('--size', default='tiny',
                   help='only used to compose a valid config; the checkpoint '
                        'architecture overrides it')
    p.add_argument('--model', default='cider')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--max_batches', type=int, default=None)
    p.add_argument('--num_steps', type=int, nargs='+', default=None,
                   help="override inference steps: one value for all targets, or "
                        "one per target. Because T already scales with N in the "
                        "trained models (12->10, 18->12, 24->20, 48->24), holding "
                        "the source's T fixed across lengths understates transfer "
                        "-- it decides ~4x more positions per step at N=48.")
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--override', nargs='*', default=[])
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(os.path.expanduser(args.checkpoint), map_location='cpu',
                      weights_only=False)
    src_cfg = load_source_config(ckpt)
    src_name = src_cfg.data.get('name', '?')
    src_Q = int(src_cfg.data.Q)
    src_K = int(src_cfg.data.get('K_max', src_cfg.data.get('K_true', 2)))
    src_steps = int(src_cfg.model.get('inference_steps', 16))
    if args.num_steps is None:
        steps_per_target = [src_steps] * len(args.targets)
    elif len(args.num_steps) == 1:
        steps_per_target = args.num_steps * len(args.targets)
    elif len(args.num_steps) == len(args.targets):
        steps_per_target = list(args.num_steps)
    else:
        raise SystemExit(
            f"--num_steps takes 1 value or one per target "
            f"({len(args.targets)}); got {len(args.num_steps)}")

    print("=" * 78)
    print(f"Zero-shot code transfer  |  source: {src_name} "
          f"(Q={src_Q}, N={src_cfg.data.N}, K={src_K}, "
          f"D={src_cfg.model.D_model}, trained steps={src_steps})")
    print(f"checkpoint: {args.checkpoint}")
    print("=" * 78)
    print(f"{'target':<18}{'N':>4}{'M':>5}{'T':>4}  {'kind':<12}{'SER':>10}{'PUPE':>10}"
          f"{'rows':>8}{'sec':>7}")
    print("-" * 78)

    rows, skipped = [], []
    for raw_tgt, steps in zip(args.targets, steps_per_target):
      tgt = target_label(raw_tgt)                 # row label
      # One unavailable dataset must not abort a sweep that may take hours.
      try:
        cfg = build_target_config(raw_tgt, args.size, args.model,
                           [f"training.batch_size={args.batch_size}", *args.override])
        tgt_Q = int(cfg.data.Q)
        tgt_K = int(cfg.data.get('K_max', cfg.data.get('K_true', 2)))

        # Q and K are baked into parameter shapes; N, M and H are not.
        if tgt_Q != src_Q or tgt_K != src_K:
            why = (f"Q {src_Q}->{tgt_Q}" if tgt_Q != src_Q else "") + \
                  (" " if tgt_Q != src_Q and tgt_K != src_K else "") + \
                  (f"K {src_K}->{tgt_K}" if tgt_K != src_K else "")
            print(f"{tgt:<18}{cfg.data.N:>4}{cfg.data.get('M', '?'):>5}  "
                  f"{'SKIP':<14}{'-':>10}{'-':>10}  ({why} needs retraining)")
            skipped.append((tgt, why))
            continue

        H = load_H_matrix(cfg)
        if H is None:
            print(f"{tgt:<18}{'':>4}{'':>5}  {'SKIP':<14}  (no H_matrix.pt)")
            skipped.append((tgt, 'no H'))
            continue
        H = H.to(device)

        model, unexpected = build_transfer_model(src_cfg, cfg, ckpt, device)
        loader = load_test_split(cfg, args.batch_size)

        r = evaluate(model, loader, H, device, steps, args.max_batches)
        kind = 'matched' if tgt == src_name else 'ZERO-SHOT'
        print(f"{tgt:<18}{cfg.data.N:>4}{cfg.data.M:>5}{steps:>4}  {kind:<12}"
              f"{r['SER']:>10.5f}{r['PUPE']:>10.5f}{r['rows']:>8}{r['sec']:>7.1f}")
        rows.append((tgt, kind, r))
      except SystemExit:
        raise
      except Exception as e:
        print(f"{tgt:<18}{'':>4}{'':>5}  {'ERROR':<14}  {type(e).__name__}: {e}")
        skipped.append((tgt, type(e).__name__))

    print("-" * 78)
    matched = next((r for t, k, r in rows if k == 'matched'), None)
    if matched:
        print(f"matched reference: SER {matched['SER']:.5f}  PUPE {matched['PUPE']:.5f}")
        for tgt, kind, r in rows:
            if kind == 'ZERO-SHOT':
                print(f"  {tgt:<18} PUPE {r['PUPE']:.5f}  "
                      f"({r['PUPE'] - matched['PUPE']:+.5f} vs matched)")
    else:
        print("note: no matched row -- include the checkpoint's own training set "
              "in --targets for a reference point.")
    if skipped:
        print(f"skipped: {', '.join(f'{t} ({w})' for t, w in skipped)}")


if __name__ == '__main__':
    main()

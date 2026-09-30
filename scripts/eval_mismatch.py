#!/usr/bin/env python3
"""
Evaluate CIDER robustness to inner-detector mismatch (Table tab:detector_mismatch).

Generates H matrix on-the-fly (same deterministic construction as training),
then generates test data under mismatched conditions (different Eb/N0, AMP iterations)
and evaluates a trained checkpoint without retraining.

Matched condition. Every dataset was generated with
BatchedModulationDecoder(max_iter=AMP_MAX_ITER=10) (generate_data_from_H.py),
so the matched condition inherits that value (Eb=10, I_AMP=10) and I_AMP=20
is a mismatch. The published tab:detector_mismatch labels I_AMP=20 as matched
and runs its SNR rows at I_AMP=20; that grid is available as the opt-in
conditions amp20 / amp10 / amp5 / eb8_amp20 / eb12_amp20 / eb6_amp20.

Test data. Codewords/noise are drawn from --data_seed (default 2026) after the
sensing matrix A is built from --seed 42, so A matches training while the frames
do not replay the seed-42 training stream. A is drawn with torch.randperm on the
generation device and the training sets were generated on CUDA, so run this on a
GPU to reproduce the training A (on CPU A differs).

Usage (defaults: tiny K=2 checkpoint, T=12, 2000 frames per condition):
    python scripts/eval_mismatch.py \
        --checkpoint checkpoints/tiny_ldpc_tiny_cider/best_model.ckpt \
        --conditions eb10_matched amp20 amp5 eb8 eb12 eb6

    # Condition grid of the published tab:detector_mismatch (I_AMP=20 reference),
    # on fresh (non-training) frames
    python scripts/eval_mismatch.py \
        --checkpoint checkpoints/tiny_ldpc_tiny_cider/best_model.ckpt \
        --conditions amp20 amp10 amp5 eb8_amp20 eb12_amp20 eb6_amp20 \
        --num_test 15000 --data_seed 2026

The published numbers were produced before --data_seed existed, i.e. on the
seed-42 stream (the first frames of the training set), so this script does not
reproduce them bit-for-bit.
"""

import argparse
import os
import sys
import math
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from omegaconf import OmegaConf
from rich.table import Table
from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data', 'gen_data'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data', 'gen_data', 'noisy_channel'))

from diffusion import Diffusion
from data.gen_data.gf_gpu import GF_GPU
from data.gen_data.construct_H import construct_H
from data.gen_data.noisy_channel.modulation_encoder import ModulationEncoder, create_sensing_matrix
from data.gen_data.noisy_channel.modulation_decoder_batch import BatchedModulationDecoder
from data.gen_data.generate_data_from_H import AMP_MAX_ITER


# =============================================================================
# Mismatch conditions
# =============================================================================

# Matched: Eb=10, I_AMP=AMP_MAX_ITER (=10). Every dataset was generated with
# BatchedModulationDecoder(max_iter=AMP_MAX_ITER) (generate_data_from_H.py), so
# that is the training condition and I_AMP=20 is a mismatch.
I_TRAIN = AMP_MAX_ITER


def _cond(Eb, max_iter, label):
    return {'Eb': Eb, 'max_iter': max_iter, 'label': label,
            'condition': f'Eb={Eb:g}, I_AMP={max_iter}'}


MATCHED = _cond(10.0, I_TRAIN, 'None (matched)')

CONDITIONS = {
    'eb10_matched': MATCHED,
    'amp20':  _cond(10.0, 20,      'More AMP iters'),
    'amp5':   _cond(10.0, 5,       'Reduced AMP iters'),
    'eb8':    _cond(8.0,  I_TRAIN, 'SNR mismatch'),
    'eb12':   _cond(12.0, I_TRAIN, 'SNR mismatch'),
    'eb6':    _cond(6.0,  I_TRAIN, 'SNR mismatch'),
    # Opt-in: the published table's grid, which takes I_AMP=20 as reference
    'amp10':      _cond(10.0, 10, 'Reduced AMP iters'),
    'eb8_amp20':  _cond(8.0,  20, 'SNR mismatch'),
    'eb12_amp20': _cond(12.0, 20, 'SNR mismatch'),
    'eb6_amp20':  _cond(6.0,  20, 'SNR mismatch'),
}

# H matrix parameters per scale (matching ldpc_*.sh scripts)
H_PARAMS = {
    'tiny':     {'q': 64, 'L': 12, 'M': 8,  'd_v': 2, 'd_c': 3},
    'small':    {'q': 64, 'L': 18, 'M': 12, 'd_v': 2, 'd_c': 3},
    'moderate': {'q': 64, 'L': 24, 'M': 16, 'd_v': 2, 'd_c': 3},
    'large':    {'q': 64, 'L': 48, 'M': 32, 'd_v': 2, 'd_c': 3},
}


# =============================================================================
# H matrix construction
# =============================================================================

def build_H_data(scale='tiny', seed=42):
    """Construct H matrix on-the-fly using the same deterministic construction as training."""
    params = H_PARAMS[scale]
    h_data = construct_H(
        q=params['q'], L=params['L'], M=params['M'],
        d_v=params['d_v'], d_c=params['d_c'], seed=seed,
    )
    return h_data


# =============================================================================
# Data generation (test only)
# =============================================================================

def create_encoder_from_H(h_data, device='cuda'):
    """Create GPU-accelerated encoding function from loaded H data."""
    q = h_data['q']
    k = h_data['k']
    L = h_data['L']

    gf_gpu = GF_GPU(q, device)

    H1 = h_data['H1']
    H2_inv = h_data['H2_inv']
    # Ensure tensors are on the right device
    if not isinstance(H1, torch.Tensor):
        H1 = torch.tensor(np.array(H1, dtype=np.int64), dtype=torch.long)
    if not isinstance(H2_inv, torch.Tensor):
        H2_inv = torch.tensor(np.array(H2_inv, dtype=np.int64), dtype=torch.long)
    H1 = H1.to(dtype=torch.long, device=device)
    H2_inv = H2_inv.to(dtype=torch.long, device=device)

    Pi = h_data['Pi']
    if isinstance(Pi, torch.Tensor):
        Pi = Pi.tolist()
    else:
        Pi = list(Pi)
    Pi_inv = torch.zeros(L, dtype=torch.long, device=device)
    for i, p in enumerate(Pi):
        Pi_inv[p] = i

    def encode_batch_fn(batch_size):
        codewords = gf_gpu.ldpc_encode_batch(H1, H2_inv, Pi_inv, k, batch_size)
        return codewords.cpu().numpy()

    return encode_batch_fn


def generate_test_data(h_data, K, Eb, max_iter, n_s=24, sigma2=1.0,
                       num_test=2000, batch_size=2048, seed=42, device='cuda',
                       data_seed=None):
    """Generate test data under specified conditions."""
    q = h_data['q']
    L = h_data['L']
    k_info = h_data['k']
    bits_per_symbol = int(np.log2(q))
    B = k_info * bits_per_symbol

    np.random.seed(seed)
    torch.manual_seed(seed)

    encode_batch_fn = create_encoder_from_H(h_data, device)

    A = create_sensing_matrix(n_s=n_s, Q=q, matrix_type='partial_dft', seed=seed, device=device)
    # create_sensing_matrix reseeds with `seed` (the training seed, so A matches
    # training). Without a separate data seed the codeword/noise stream below is
    # the training stream, i.e. the "test" frames would be train[:num_test].
    if data_seed is not None:
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
    encoder = ModulationEncoder(A=A, B=B, Eb=Eb, L=L)
    decoder = BatchedModulationDecoder(K=K, max_iter=max_iter, sigma2=sigma2)

    metadata = {'K': K, 'Q': q, 'gamma': encoder.Psym, 'sigma2': sigma2}

    Y_all = np.zeros((num_test, L, q), dtype=np.float32)
    X0_all = np.zeros((num_test, K, L), dtype=np.int64)

    num_batches = (num_test + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_test)
        curr_batch_size = end_idx - start_idx

        batch_codewords = encode_batch_fn(curr_batch_size * K)
        X0_batch = batch_codewords.reshape(curr_batch_size, K, L)
        X0_all[start_idx:end_idx] = X0_batch

        for pos in range(L):
            active_syms_batch = torch.tensor(X0_batch[:, :, pos], dtype=torch.long, device=device)
            batch_size_actual = active_syms_batch.shape[0]
            n_s_actual, Q_actual = A.shape
            sqrt_Psym = np.sqrt(encoder.Psym)

            x_batch = torch.zeros(batch_size_actual, n_s_actual, dtype=A.dtype, device=device)
            for kk in range(active_syms_batch.shape[1]):
                indices = active_syms_batch[:, kk]
                x_batch += sqrt_Psym * A[:, indices].T

            noise_real = torch.randn(batch_size_actual, n_s_actual, device=device) * np.sqrt(sigma2 / 2)
            noise_imag = torch.randn(batch_size_actual, n_s_actual, device=device) * np.sqrt(sigma2 / 2)
            noise = torch.complex(noise_real, noise_imag)
            Y_recv = x_batch + noise

            output = decoder.forward_batch(Y_recv, A, metadata, output_type='logits')
            Y_all[start_idx:end_idx, pos, :] = output.cpu().numpy()

    Y_out = torch.tensor(Y_all, dtype=torch.float32)
    X0_out = torch.tensor(X0_all, dtype=torch.long)
    return Y_out, X0_out


# =============================================================================
# Model loading
# =============================================================================

def load_model(checkpoint_path, h_matrix, device):
    """Load CIDER model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
        config = OmegaConf.create(checkpoint['hyper_parameters']['config'])
    else:
        raise ValueError("Cannot find config in checkpoint")

    # Remap legacy backbone names to current names
    BACKBONE_ALIASES = {'dimp_no_gru': 'cider', 'dimp_rev2': 'cider_gru'}
    OmegaConf.set_struct(config, False)
    backbone_type = config.model.get('backbone_type', 'cider')
    if backbone_type in BACKBONE_ALIASES:
        config.model.backbone_type = BACKBONE_ALIASES[backbone_type]
    OmegaConf.set_struct(config, True)

    model = Diffusion(config)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k: v for k, v in state_dict.items() if k != 'H'}
    model.load_state_dict(state_dict, strict=False)

    # Load EMA weights
    if 'ema' in checkpoint and 'shadow_params' in checkpoint['ema']:
        shadow_params = checkpoint['ema']['shadow_params']
        for name, param in model.backbone.named_parameters():
            if name in shadow_params:
                param.data.copy_(shadow_params[name])
        for name, buf in model.backbone.named_buffers():
            if name in shadow_params:
                buf.copy_(shadow_params[name])

    model.set_H_matrix(h_matrix)
    model = model.to(device)
    model.eval()
    return model, config


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, Y, X0, config, device, batch_size=128, T_steps=12):
    """Evaluate model on test data, return SER and CER."""
    K = config.data.K_max
    N = config.data.N

    num_samples = Y.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    correct_symbols = 0
    total_symbols = 0
    correct_codewords = 0
    total_codewords = 0

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        Y_batch = Y[start:end].to(device).float()
        X0_batch = X0[start:end].to(device)
        B = Y_batch.shape[0]

        preds = model._sample(Y_batch, num_steps=T_steps, use_remasking=False)

        for b in range(B):
            cost = torch.zeros(K, K, device=device)
            for i in range(K):
                for u in range(K):
                    cost[i, u] = (preds[b, i] != X0_batch[b, u]).float().sum()

            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

            for i, u in zip(row_ind, col_ind):
                matches = (preds[b, i] == X0_batch[b, u])
                correct_symbols += matches.sum().item()
                total_symbols += N
                if matches.all():
                    correct_codewords += 1
                total_codewords += 1

    symbol_acc = correct_symbols / total_symbols
    codeword_acc = correct_codewords / total_codewords
    ser = 1.0 - symbol_acc
    cer = 1.0 - codeword_acc
    return ser, cer


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate CIDER robustness to inner-detector mismatch")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint')
    parser.add_argument('--scale', type=str, default='tiny', choices=list(H_PARAMS.keys()),
                        help='Code scale (determines H matrix parameters)')
    parser.add_argument('--conditions', nargs='+',
                        default=['eb10_matched', 'amp20', 'amp5', 'eb8', 'eb12', 'eb6'],
                        choices=list(CONDITIONS.keys()),
                        help='Mismatch conditions to evaluate')
    parser.add_argument('--num_test', type=int, default=2000, help='Number of test samples')
    parser.add_argument('--n_s', type=int, default=24, help='Inner code length')
    parser.add_argument('--sigma2', type=float, default=1.0, help='Noise variance')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (same as training for identical H)')
    parser.add_argument('--data_seed', type=int, default=2026,
                        help='Seed for test codewords/noise, applied after A is built. Must differ '
                             'from the training seed, otherwise the frames replay the training set.')
    parser.add_argument('--num_steps', type=int, default=12,
                        help='Reveal steps T (12 = paper T for Tiny)')
    parser.add_argument('--batch_size', type=int, default=128, help='Evaluation batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--h_cache_dir', type=str, default=None,
                        help='Where the constructed H is saved and reloaded '
                             '(default: the checkpoint directory)')

    args = parser.parse_args()
    args.checkpoint = os.path.expanduser(args.checkpoint)
    if args.output:
        args.output = os.path.expanduser(args.output)
    h_cache_dir = os.path.expanduser(args.h_cache_dir or os.path.dirname(args.checkpoint))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    console = Console()

    # Construct H matrix once, save to disk, then reload
    # This ensures identical H across all conditions (no state drift)
    os.makedirs(h_cache_dir, exist_ok=True)
    h_matrix_path = os.path.join(h_cache_dir, f'H_matrix_{args.scale}_seed{args.seed}.pt')
    console.print(f"\n[bold]Constructing H matrix (scale={args.scale}, seed={args.seed})...[/bold]")
    h_data = build_H_data(scale=args.scale, seed=args.seed)
    torch.save(h_data, h_matrix_path)
    console.print(f"Saved H matrix to {h_matrix_path}")
    # Reload from disk to guarantee consistency
    h_data = torch.load(h_matrix_path, weights_only=False)
    H_matrix = h_data['H_matrix']
    q = h_data['q']
    L = h_data['L']

    console.print(f"\n[bold]CIDER Inner-Detector Mismatch Evaluation[/bold]")
    console.print(f"H matrix: q={q}, L={L}, M={h_data['M']}, k={h_data['k']}")
    console.print(f"Checkpoint: {args.checkpoint}")
    console.print(f"Conditions: {args.conditions}\n")

    # Load model once
    console.print("Loading model...")
    model, config = load_model(args.checkpoint, H_matrix, device)
    K_users = config.data.K_max
    console.print(f"Model loaded: {config.model.get('backbone_type', 'unknown')}\n")

    # Evaluate each condition
    results = []
    for cond_name in args.conditions:
        cond = CONDITIONS[cond_name]
        console.print(f"[bold cyan]Condition: {cond['condition']}[/bold cyan]")

        # Generate test data
        console.print(f"  Generating {args.num_test} test samples (Eb={cond['Eb']}, max_iter={cond['max_iter']})...")
        t0 = time.time()
        Y, X0 = generate_test_data(
            h_data=h_data, K=K_users, Eb=cond['Eb'], max_iter=cond['max_iter'],
            n_s=args.n_s, sigma2=args.sigma2, num_test=args.num_test,
            seed=args.seed, device=str(device), data_seed=args.data_seed,
        )
        gen_time = time.time() - t0
        console.print(f"  Generated in {gen_time:.1f}s")

        # Evaluate
        console.print(f"  Evaluating...")
        t0 = time.time()
        ser, cer = evaluate(model, Y, X0, config, device, batch_size=args.batch_size,
                            T_steps=args.num_steps)
        eval_time = time.time() - t0
        console.print(f"  SER={ser:.6f}, CER={cer:.6f} ({eval_time:.1f}s)\n")

        results.append({
            'name': cond_name,
            'label': cond['label'],
            'condition': cond['condition'],
            'Eb': cond['Eb'],
            'max_iter': cond['max_iter'],
            'data_seed': args.data_seed,
            'num_test': args.num_test,
            'num_steps': args.num_steps,
            'SER': ser,
            'CER': cer,
        })

    # Print results table
    table = Table(title="CIDER Robustness to Inner-Detector Mismatch (Tiny, K=2)")
    table.add_column("Mismatch type", style="cyan")
    table.add_column("Condition", style="white")
    table.add_column("SER", justify="right", style="green")
    table.add_column("CER", justify="right", style="green")

    matched_ser = None
    for r in results:
        if r['name'] == 'eb10_matched':
            matched_ser = r['SER']

    for r in results:
        ser_str = f"{r['SER']:.4f}"
        cer_str = f"{r['CER']:.4f}"
        table.add_row(r['label'], r['condition'], ser_str, cer_str)

    console.print(table)

    # Print relative degradation
    if matched_ser and matched_ser > 0:
        console.print("\n[bold]Relative degradation vs matched:[/bold]")
        for r in results:
            if r['name'] != 'eb10_matched':
                ratio = r['SER'] / matched_ser
                console.print(f"  {r['condition']:20s}: SER {ratio:.1f}x")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

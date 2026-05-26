#!/usr/bin/env python3
"""
Generate dataset from pre-constructed H matrix.

Step 2 of the two-step workflow:
1. construct_H.py → creates H_matrix.pt
2. generate_data_from_H.py → loads H, generates train/val/test data

Usage:
    python generate_data_from_H.py --h_matrix H_matrix.pt --K 2 --num_train 10000 \
        --n_s 24 --Eb 11.0 --output data/
"""

import argparse
import os
import sys
import json
import numpy as np
import torch

script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'noisy_channel'))

from gf_gpu import GF_GPU
from noisy_channel.modulation_encoder import ModulationEncoder, create_sensing_matrix
from noisy_channel.modulation_decoder_batch import BatchedModulationDecoder


def load_H(h_matrix_path):
    """Load H matrix and encoding components from file."""
    data = torch.load(h_matrix_path)

    print(f"Loaded H matrix from {h_matrix_path}")
    print(f"  q={data['q']}, L={data['L']}, M={data['M']}")
    print(f"  d_v={data['d_v']}, d_c={data['d_c']}, k={data['k']}, Z={data['Z']}")

    return data


def create_encoder_from_H(h_data, device='cuda'):
    """
    Create GPU-accelerated encoding function from loaded H data.

    Args:
        h_data: Loaded H matrix data
        device: 'cuda' or 'cpu'

    Returns:
        encode_batch_fn: Function that generates batch of random codewords on GPU
        gf_gpu: GF_GPU instance
    """
    q = h_data['q']
    k = h_data['k']
    L = h_data['L']

    # Initialize GPU GF operations
    gf_gpu = GF_GPU(q, device)

    # Move matrices to GPU
    H1 = torch.tensor(h_data['H1'].numpy(), dtype=torch.long, device=device)
    H2_inv = torch.tensor(h_data['H2_inv'].numpy(), dtype=torch.long, device=device)

    # Compute inverse permutation
    Pi = h_data['Pi'].tolist()
    Pi_inv = torch.zeros(L, dtype=torch.long, device=device)
    for i, p in enumerate(Pi):
        Pi_inv[p] = i

    def encode_batch_fn(batch_size):
        """Generate batch of random LDPC codewords on GPU."""
        codewords = gf_gpu.ldpc_encode_batch(H1, H2_inv, Pi_inv, k, batch_size)
        return codewords.cpu().numpy()

    return encode_batch_fn, gf_gpu


def run_inner_code_batch(active_symbols_batch, encoder, decoder, A, M_ant, sigma2, metadata, output_type='logits'):
    """Run inner code pipeline for a batch of samples at one position (fully batched).

    Args:
        output_type: 'logits' (log-posterior, default) or 'norm_mag' (legacy)
    """
    batch_size = active_symbols_batch.shape[0]
    n_s, Q = A.shape
    device = A.device
    dtype = A.dtype

    sqrt_Psym = np.sqrt(encoder.Psym)

    # Batch encode
    x_batch = torch.zeros(batch_size, n_s, dtype=dtype, device=device)
    for k in range(active_symbols_batch.shape[1]):
        indices = active_symbols_batch[:, k]
        x_batch += sqrt_Psym * A[:, indices].T

    # Batch AWGN channel
    noise_real = torch.randn(batch_size, n_s, device=device) * np.sqrt(sigma2 / 2)
    noise_imag = torch.randn(batch_size, n_s, device=device) * np.sqrt(sigma2 / 2)
    noise = torch.complex(noise_real, noise_imag)

    Y_recv_batch = x_batch + noise  # (batch, n_s)

    # Batched decode - output logits (log-posterior) by default
    output_batch = decoder.forward_batch(Y_recv_batch, A, metadata, output_type=output_type)

    return output_batch


def generate_dataset(num_samples, encode_batch_fn, K, L, Q, encoder, decoder, A, M_ant, sigma2, metadata, device, batch_size=2048):
    """Generate dataset with batched processing (codewords generated per batch)."""

    print(f"Generating {num_samples} samples (batch_size={batch_size})...")

    Y_all = np.zeros((num_samples, L, Q), dtype=np.float32)
    X0_all = np.zeros((num_samples, K, L), dtype=np.int64)

    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        curr_batch_size = end_idx - start_idx

        if (batch_idx + 1) % max(1, num_batches // 10) == 0 or batch_idx == 0:
            print(f"  Batch {batch_idx+1}/{num_batches}: generating {curr_batch_size} samples...")

        # Generate codewords for this batch only
        batch_codewords = encode_batch_fn(curr_batch_size * K)  # [batch * K, L]
        X0_batch = batch_codewords.reshape(curr_batch_size, K, L)
        X0_all[start_idx:end_idx] = X0_batch

        # Process through inner channel
        for pos in range(L):
            active_syms_batch = torch.tensor(
                X0_batch[:, :, pos], dtype=torch.long, device=device
            )
            norm_mag_batch = run_inner_code_batch(
                active_syms_batch, encoder, decoder, A, M_ant, sigma2, metadata
            )
            Y_all[start_idx:end_idx, pos, :] = norm_mag_batch.cpu().numpy()

    Y_out = torch.tensor(Y_all, dtype=torch.float32)
    X0_out = torch.tensor(X0_all, dtype=torch.long)

    return Y_out, X0_out


def main():
    parser = argparse.ArgumentParser(description="Generate dataset from pre-constructed H matrix")

    # H matrix input
    parser.add_argument('--h_matrix', type=str, required=True, help='Path to H_matrix.pt')

    # Data generation params
    parser.add_argument('--K', type=int, required=True, help='Number of active users')
    parser.add_argument('--num_train', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--num_val', type=int, default=2000, help='Number of validation samples')
    parser.add_argument('--num_test', type=int, default=2000, help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for generation')

    # Inner code params
    parser.add_argument('--matrix_type', type=str, default='partial_dft',
                       choices=['partial_dft', 'qr_gaussian', 'hadamard', 'complex_hadamard', 'qpsk_hadamard'],
                       help='Sensing matrix type')
    parser.add_argument('--n_s', type=int, default=8, help='Inner code length')
    parser.add_argument('--M_ant', type=int, default=1, help='Number of receive antennas')
    parser.add_argument('--Eb', type=float, default=0.0, help='Energy per bit (dB)')
    parser.add_argument('--sigma2', type=float, default=1.0, help='Noise variance')

    # Output
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data generation')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load H matrix
    h_data = load_H(args.h_matrix)
    q = h_data['q']
    L = h_data['L']
    k = h_data['k']

    # Create GPU encoder from H
    encode_batch_fn, gf_gpu = create_encoder_from_H(h_data, device)

    # Compute B (total info bits)
    bits_per_symbol = int(np.log2(q))
    B = k * bits_per_symbol

    print(f"\nInner code parameters:")
    print(f"  matrix_type={args.matrix_type}, n_s={args.n_s}, M_ant={args.M_ant}")
    print(f"  Eb={args.Eb}dB, sigma2={args.sigma2}")
    print(f"  B={B} (k={k} * {bits_per_symbol} bits/symbol)")

    # Create sensing matrix
    A = create_sensing_matrix(
        n_s=args.n_s, Q=q, matrix_type=args.matrix_type,
        seed=args.seed, device=device
    )
    print(f"  A shape: {A.shape}")

    # Create inner code encoder/decoder
    encoder = ModulationEncoder(A=A, B=B, Eb=args.Eb, L=L)
    decoder = BatchedModulationDecoder(K=args.K, max_iter=10, sigma2=args.sigma2)

    gamma = args.K / q
    metadata = {'K': args.K, 'Q': q, 'gamma': encoder.Psym, 'sigma2': args.sigma2}
    print(f"  Psym={encoder.Psym:.4f}, gamma(K/Q)={gamma:.4f}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Copy H matrix to output
    h_output_path = os.path.join(args.output, 'H_matrix.pt')
    torch.save(h_data, h_output_path)
    print(f"\nCopied H matrix to {h_output_path}")

    # Generate datasets
    for split, num_samples in [('train', args.num_train), ('val', args.num_val), ('test', args.num_test)]:
        if num_samples == 0:
            continue

        print(f"\n{'='*60}")
        print(f"Generating {split} set ({num_samples} samples)")
        print(f"{'='*60}")

        Y, X0 = generate_dataset(
            num_samples=num_samples,
            encode_batch_fn=encode_batch_fn,
            K=args.K, L=L, Q=q,
            encoder=encoder, decoder=decoder,
            A=A, M_ant=args.M_ant, sigma2=args.sigma2,
            metadata=metadata, device=device,
            batch_size=args.batch_size
        )

        data = {'Y': Y, 'gt_codewords': X0}
        output_path = os.path.join(args.output, f'{split}_data.pt')
        torch.save(data, output_path)
        print(f"Saved {output_path}: Y={Y.shape}, X0={X0.shape}")

    # Save metadata
    metadata_out = {
        'h_matrix_source': args.h_matrix,
        'code_params': {
            'q': q, 'L': L, 'M': h_data['M'],
            'd_v': h_data['d_v'], 'd_c': h_data['d_c'],
            'k': k, 'Z': h_data['Z'],
        },
        'inner_code_params': {
            'matrix_type': args.matrix_type,
            'n_s': args.n_s, 'M_ant': args.M_ant,
            'Eb': args.Eb, 'sigma2': args.sigma2,
            'B': B,
        },
        'data_params': {
            'K': args.K,
            'num_train': args.num_train,
            'num_val': args.num_val,
            'num_test': args.num_test,
            'seed': args.seed,
        }
    }
    with open(os.path.join(args.output, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata_out, f, indent=2)
    print(f"\nSaved dataset_metadata.json")

    print(f"\nDone! Dataset saved to {args.output}")


if __name__ == "__main__":
    main()

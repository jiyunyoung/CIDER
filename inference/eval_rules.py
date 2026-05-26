#!/usr/bin/env python
"""
Evaluate rule-based decoders (sic_bp, top_j_es) on test data.

Usage:
    python inference/eval_rules.py sic_bp tiny_ldpc
    python inference/eval_rules.py top_j_es tiny_ldpc
    python inference/eval_rules.py sic_bp moderate_ldpc --max_iters 100
"""

import argparse
import numpy as np
import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_data_path(data_name: str) -> str:
    """Get data path from config name."""
    # Map config names to directory names
    name_map = {
        'tiny_ldpc': 'tiny_LDPC',
        'small_ldpc': 'small_LDPC',
        'moderate_ldpc': 'moderate_LDPC',
        'large_ldpc': 'large_LDPC',
        'tiny_tree': 'tiny_tree',
        'tiny_ldpc_peg': 'tiny_LDPC_PEG',
    }
    dir_name = name_map.get(data_name, data_name)

    # Try multiple paths
    paths = [
        os.path.expanduser(f'~/data/demix/{dir_name}'),
        f'data/demix/{dir_name}',
        os.path.expanduser(f'~/data/demix/{dir_name}'),
    ]

    for path in paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Data not found for {data_name}. Tried: {paths}")


def load_data(data_path: str, split: str = 'test'):
    """Load data and H matrix."""
    data = torch.load(os.path.join(data_path, f'{split}_data.pt'))
    H_dict = torch.load(os.path.join(data_path, 'H_matrix.pt'))

    X = data['gt_codewords'].numpy()  # [B, K, N]
    Y = data['Y'].numpy()  # [B, N, Q]
    H = H_dict['H_matrix'].numpy()

    return X, Y, H


def hungarian_match(pred, gt):
    """
    Hungarian matching for K codewords.

    Returns:
        perm: assignment (pred[i] matches gt[perm[i]])
        total_symbol_errors: total symbol errors across all matched pairs
        per_codeword_errors: list of symbol errors for each matched codeword
    """
    from scipy.optimize import linear_sum_assignment
    K = pred.shape[0]
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = (pred[i] != gt[j]).sum()
    row_ind, col_ind = linear_sum_assignment(cost)

    per_codeword_errors = [cost[i, j] for i, j in zip(row_ind, col_ind)]
    total_symbol_errors = sum(per_codeword_errors)

    return col_ind, total_symbol_errors, per_codeword_errors


def evaluate_sic_bp(X, Y, H, max_iters=50, damping=0.1, explain_strength=1.0, K_override=None, num_orderings=1, batch_size=256):
    """Evaluate factorized BP decoder with batched processing."""
    from rules.sic_bp import FactorizedBPDecoder

    B, K_data, N = X.shape
    K = K_override if K_override is not None else K_data
    Q = Y.shape[-1]
    M = H.shape[0]

    print(f"  Using K={K} (data has K_data={K_data}), batch_size={batch_size}")

    decoder = FactorizedBPDecoder(
        Q=Q, N=N, K=K, M=M, H=H,
        max_iters=max_iters,
        damping=damping,
        explain_strength=explain_strength,
    )

    correct_codewords = 0
    total_codewords = 0
    total_symbols = 0
    correct_symbols = 0
    start = time.time()

    num_batches = (B + batch_size - 1) // batch_size
    for bi in range(num_batches):
        b_start = bi * batch_size
        b_end = min(b_start + batch_size, B)
        Y_batch = Y[b_start:b_end]
        X_batch = X[b_start:b_end]
        curr_B = b_end - b_start

        preds = decoder.decode_batch(Y_batch)  # [curr_B, K, N]

        for b in range(curr_B):
            K_eval = min(K, K_data)
            perm, symbol_errors, per_cw_errors = hungarian_match(preds[b, :K_eval], X_batch[b, :K_eval])

            correct_symbols += K_eval * N - int(symbol_errors)
            total_symbols += K_eval * N

            for err in per_cw_errors:
                if err == 0:
                    correct_codewords += 1
                total_codewords += 1

        processed = b_end
        if processed % 1000 < batch_size or bi == num_batches - 1:
            print(f"  Progress: {processed}/{B} ({100*correct_codewords/total_codewords:.1f}% cw_acc, {100*correct_symbols/total_symbols:.2f}% sym_acc)")

    elapsed = time.time() - start
    return correct_codewords, total_codewords, correct_symbols, total_symbols, elapsed


def evaluate_stitching(X, Y, H, proposal_width=2, beam_width=1000, K_override=None):
    """Evaluate stitching decoder (beam search with parity pruning)."""
    from rules.stitching_decoder import StitchingDecoder

    B, K_data, N = X.shape
    K = K_override if K_override is not None else K_data
    Q = Y.shape[-1]
    M = H.shape[0]

    print(f"  Using K={K} (data has K_data={K_data}), beam_width={beam_width}, proposal_width={proposal_width}")

    decoder = StitchingDecoder(
        Q=Q, N=N, K=K, M=M, H=H,
        beam_width=beam_width,
        proposal_width=proposal_width,
    )

    correct_codewords = 0
    total_codewords = 0
    total_symbols = 0
    correct_symbols = 0
    start = time.time()

    for b in range(B):
        pred, _ = decoder.decode(Y[b])
        gt = X[b]

        K_eval = min(K, K_data)
        perm, symbol_errors, per_cw_errors = hungarian_match(pred[:K_eval], gt[:K_eval])

        correct_symbols += K_eval * N - int(symbol_errors)
        total_symbols += K_eval * N

        for err in per_cw_errors:
            if err == 0:
                correct_codewords += 1
            total_codewords += 1

        if (b + 1) % 100 == 0:
            elapsed_so_far = time.time() - start
            ser_so_far = 1 - correct_symbols / total_symbols
            cer_so_far = 1 - correct_codewords / total_codewords
            print(f"  Progress: {b+1}/{B} SER={ser_so_far:.6f} CER={cer_so_far:.6f} ({1000*elapsed_so_far/(b+1):.1f} ms/sample)")

    elapsed = time.time() - start
    return correct_codewords, total_codewords, correct_symbols, total_symbols, elapsed


def evaluate_top_j_es(X, Y, H, proposal_width=2, K_override=None):
    """Evaluate beam search decoder."""
    from rules.top_j_exhaustive_search import BeamSearchDecoder

    B, K_data, N = X.shape
    K = K_override if K_override is not None else K_data
    Q = Y.shape[-1]
    M = H.shape[0]

    print(f"  Using K={K} (data has K_data={K_data})")

    decoder = BeamSearchDecoder(
        Q=Q, N=N, K=K, M=M, H=H,
        proposal_width=proposal_width,
    )

    correct_codewords = 0
    total_codewords = 0
    total_symbols = 0
    correct_symbols = 0
    start = time.time()

    for b in range(B):
        pred, _ = decoder.decode(Y[b])
        gt = X[b]

        # Hungarian matching for arbitrary K
        K_eval = min(K, K_data)
        perm, symbol_errors, per_cw_errors = hungarian_match(pred[:K_eval], gt[:K_eval])

        correct_symbols += K_eval * N - int(symbol_errors)
        total_symbols += K_eval * N

        # Per-codeword accuracy: count each codeword with 0 errors
        for err in per_cw_errors:
            if err == 0:
                correct_codewords += 1
            total_codewords += 1

        if (b + 1) % 1000 == 0:
            print(f"  Progress: {b+1}/{B} ({100*correct_codewords/total_codewords:.1f}% cw_acc, {100*correct_symbols/total_symbols:.2f}% sym_acc)")

    elapsed = time.time() - start
    return correct_codewords, total_codewords, correct_symbols, total_symbols, elapsed


def main():
    parser = argparse.ArgumentParser(description='Evaluate rule-based decoders')
    parser.add_argument('decoder', choices=['sic_bp', 'top_j_es', 'stitching'],
                        help='Decoder type')
    parser.add_argument('data', help='Data config name (e.g., tiny_ldpc)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Direct path to data directory (overrides data name lookup)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                        help='Data split to evaluate')
    parser.add_argument('--max_iters', type=int, default=50,
                        help='Max BP iterations (sic_bp only)')
    parser.add_argument('--damping', type=float, default=0.1,
                        help='Message damping (sic_bp only)')
    parser.add_argument('--explain_strength', type=float, default=1.0,
                        help='Explain-away strength (sic_bp only)')
    parser.add_argument('--proposal_width', type=int, default=2,
                        help='Top-L candidates per position (top_j_es/stitching)')
    parser.add_argument('--beam_width', type=int, default=1000,
                        help='Beam width (stitching only)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Limit number of samples (default: all)')
    parser.add_argument('--k', type=int, default=None,
                        help='Override K (number of codewords to decode)')
    parser.add_argument('--num_orderings', type=int, default=1,
                        help='Try N orderings for sic_bp (K! tries all, default=1)')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Evaluating {args.decoder} on {args.data} ({args.split})")
    print("=" * 60)

    # Load data
    if args.data_dir is not None:
        data_path = args.data_dir
    else:
        data_path = get_data_path(args.data)
    print(f"Data path: {data_path}")
    X, Y, H = load_data(data_path, args.split)

    # Limit samples if requested
    if args.num_samples is not None:
        X = X[:args.num_samples]
        Y = Y[:args.num_samples]
        print(f"Limited to {args.num_samples} samples")

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"H shape: {H.shape}")
    print()

    # Evaluate
    if args.decoder == 'sic_bp':
        print(f"Parameters: max_iters={args.max_iters}, damping={args.damping}, "
              f"explain_strength={args.explain_strength}, K={args.k or 'auto'}, num_orderings={args.num_orderings}")
        correct, total, correct_sym, total_sym, elapsed = evaluate_sic_bp(
            X, Y, H,
            max_iters=args.max_iters,
            damping=args.damping,
            explain_strength=args.explain_strength,
            K_override=args.k,
            num_orderings=args.num_orderings,
        )
    elif args.decoder == 'stitching':
        print(f"Parameters: proposal_width={args.proposal_width}, beam_width={args.beam_width}, K={args.k or 'auto'}")
        correct, total, correct_sym, total_sym, elapsed = evaluate_stitching(
            X, Y, H,
            proposal_width=args.proposal_width,
            beam_width=args.beam_width,
            K_override=args.k,
        )
    else:
        print(f"Parameters: proposal_width={args.proposal_width}, K={args.k or 'auto'}")
        correct, total, correct_sym, total_sym, elapsed = evaluate_top_j_es(
            X, Y, H,
            proposal_width=args.proposal_width,
            K_override=args.k,
        )

    cw_acc = correct / total
    sym_acc = correct_sym / total_sym
    ser = 1.0 - sym_acc  # Symbol Error Rate

    num_samples = X.shape[0]
    cer = 1.0 - cw_acc  # Codeword Error Rate

    print()
    print("=" * 60)
    print(f"Results:")
    print(f"  Codeword Accuracy: {correct}/{total} = {cw_acc:.10f} ({100*cw_acc:.6f}%)")
    print(f"  Codeword Error Rate (CER): {cer:.10f} ({100*cer:.6f}%)")
    print(f"  Symbol Accuracy:   {correct_sym}/{total_sym} = {sym_acc:.10f} ({100*sym_acc:.6f}%)")
    print(f"  Symbol Error Rate (SER): {ser:.10f} ({100*ser:.6f}%)")
    print(f"  Time: {elapsed:.1f}s ({1000*elapsed/num_samples:.2f} ms/sample)")
    print("=" * 60)


if __name__ == '__main__':
    main()

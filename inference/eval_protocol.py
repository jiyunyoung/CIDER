#!/usr/bin/env python3
"""
Protocol-Level Evaluation for Random Access with LDPC Demixing.

Simulates a two-step random access protocol:
1. Users randomly pick 1 of 12 ZC preambles
2. Users with same preamble collide in the same slot
3. Decoder uses K-specific checkpoint to decode each slot

Usage:
    python inference_scripts/eval_protocol.py \
        --checkpoint_dir checkpoints/scale \
        --data_dir ~/data/demixK \
        --user_counts 10 15 20 25 30 \
        --num_frames 1000
"""

import argparse
import math
import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from rich.table import Table
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion import Diffusion
from models.prism_head import TokenQualityHead
from utils.gf import syndrome_over_gfq_batch


# =============================================================================
# Per-K Configuration
# =============================================================================

# Inference steps per K
INFERENCE_STEPS = {
    1: 6,
    2: 12,
    3: 24,
    4: 42,
    5: 60,
    6: 50,
    7: 62,
    8: 74,
}

# Quality head parameters for K >= 6
# Uses multi-stage threshold remasking
# thresholds: list of quality thresholds for each remasking iteration
QUALITY_HEAD_PARAMS = {
    6: {'thresholds': [0.97]},
    7: {'thresholds': [0.96, 0.90]},
    8: {'thresholds': [0.96, 0.90, 0.85]},
}


# =============================================================================
# Model Loading
# =============================================================================

def load_backbone(checkpoint_path, device):
    """Load Diffusion model from checkpoint (same as main.py _test)."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
        config = OmegaConf.create(checkpoint['hyper_parameters']['config'])
    else:
        raise ValueError(f"Cannot find config in checkpoint: {checkpoint_path}")

    # Create full Diffusion model
    diffusion_model = Diffusion(config)

    # Load base weights
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k: v for k, v in state_dict.items() if k != 'H'}
    diffusion_model.load_state_dict(state_dict, strict=False)

    # Load EMA weights into backbone (same as main.py _test)
    if 'ema' in checkpoint and 'shadow_params' in checkpoint['ema']:
        shadow_params = checkpoint['ema']['shadow_params']
        for name, param in diffusion_model.backbone.named_parameters():
            if name in shadow_params:
                param.data.copy_(shadow_params[name])
        for name, buf in diffusion_model.backbone.named_buffers():
            if name in shadow_params:
                buf.copy_(shadow_params[name])

    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()

    return diffusion_model, config


def load_quality_head(checkpoint_path, D_backbone, device):
    """Load quality head from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'hyper_parameters' in checkpoint:
        hp = checkpoint['hyper_parameters']
        qh_cfg = hp.get('quality_head_config', {})
        hidden_dim = qh_cfg.get('hidden_dim', 256)
        dropout = qh_cfg.get('dropout', 0.1)
        use_slot_context = qh_cfg.get('use_slot_context', False)
    else:
        hidden_dim = 256
        dropout = 0.1
        use_slot_context = False

    quality_head = TokenQualityHead(
        D_backbone=D_backbone,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_slot_context=use_slot_context,
    )

    state_dict = checkpoint.get('state_dict', checkpoint)
    qh_state = {}
    for k, v in state_dict.items():
        if k.startswith('quality_head.'):
            qh_state[k[13:]] = v
        elif not k.startswith('backbone.') and not k.startswith('ema.'):
            qh_state[k] = v

    quality_head.load_state_dict(qh_state, strict=False)
    quality_head = quality_head.to(device)
    quality_head.eval()

    return quality_head


def load_all_checkpoints(checkpoint_dir, device, K_range=None, K_list=None):
    """
    Load all K-specific checkpoints.

    Supports multiple checkpoint directory structures:
    1. checkpoint_dir/K{K}/best_model.ckpt (original)
    2. checkpoint_dir/Kmodel{K}/best_model.ckpt
    3. checkpoint_dir/*_Kmodel{K}_*/best_model.ckpt (glob pattern)

    Args:
        checkpoint_dir: Base directory containing checkpoints
        device: torch device
        K_range: tuple (K_min, K_max) or None for default (2, 8). Ignored if
                 K_list is provided.
        K_list:  explicit iterable of K values to load (e.g., [6, 8] skips K=7).
                 Takes precedence over K_range when set.

    Returns:
        checkpoints: dict {K: (diffusion_model, qhead or None)}
    """
    checkpoints = {}
    checkpoint_dir = Path(checkpoint_dir)

    if K_list is not None:
        ks_to_load = list(K_list)
    else:
        if K_range is None:
            K_range = (2, 9)
        K_min, K_max = K_range
        ks_to_load = list(range(K_min, K_max))

    for K in ks_to_load:
        backbone_path = None
        qhead_path = None

        # Try different directory structures
        candidates = [
            checkpoint_dir / f"K{K}" / "best_model.ckpt",
            checkpoint_dir / f"Kmodel{K}" / "best_model.ckpt",
            checkpoint_dir / f"K{K}" / "latest.ckpt",
            checkpoint_dir / f"Kmodel{K}" / "latest.ckpt",
        ]

        # Also try glob pattern for Results_Interference style
        glob_patterns = [
            f"*_Kmodel{K}_*/best_model.ckpt",
            f"*_Kmodel{K}_*/latest.ckpt",
            f"*_K{K}_*/best_model.ckpt",
        ]

        for pattern in glob_patterns:
            matches = list(checkpoint_dir.glob(pattern))
            if matches:
                # Use most recent (sorted by name, last one)
                candidates.insert(0, sorted(matches)[-1])

        for candidate in candidates:
            if candidate.exists():
                backbone_path = candidate
                break

        if backbone_path is None:
            print(f"Warning: No checkpoint for K={K}")
            continue

        print(f"Loading K={K} model from {backbone_path}")
        diffusion_model, config = load_backbone(backbone_path, device)

        # Try to find quality head
        qhead = None
        qhead_candidates = [
            backbone_path.parent / "best_quality_head.ckpt",
            backbone_path.parent / "quality_head.ckpt",
        ]

        for qhead_candidate in qhead_candidates:
            if K >= 6 and qhead_candidate.exists():
                print(f"Loading K={K} quality head from {qhead_candidate}")
                qhead = load_quality_head(qhead_candidate, diffusion_model.backbone.D, device)
                break

        checkpoints[K] = (diffusion_model, qhead)

    return checkpoints


def load_all_data(data_dir):
    """
    Load test data for all K values.

    Returns:
        data: dict {K: (Y_tensor, X0_tensor)}
        H: parity check matrix
    """
    data = {}
    data_dir = Path(data_dir)
    H = None

    for K in range(1, 9):  # K=1 to K=8
        k_dir = data_dir / f"K{K}"
        data_path = k_dir / "test_data.pt"

        if not data_path.exists():
            print(f"Warning: No data for K={K} at {data_path}")
            continue

        print(f"Loading K={K} data from {data_path}")
        loaded = torch.load(data_path, weights_only=True)
        Y = loaded['Y']  # [num_samples, N, Q]
        X0 = loaded['gt_codewords']  # [num_samples, K, N]
        data[K] = (Y, X0)

        # Load H matrix (same for all K)
        if H is None:
            H_path = k_dir / "H_matrix.pt"
            if H_path.exists():
                H_data = torch.load(H_path, weights_only=True)
                H = H_data.get('H_matrix', H_data.get('H', None))

    return data, H


# =============================================================================
# Inference Functions
# =============================================================================

def cosine_anneal(start, end, t, T):
    if T <= 1:
        return end
    alpha = (t - 1) / (T - 1)
    weight = 0.5 * (1 + math.cos(math.pi * alpha))
    return start * weight + end * (1 - weight)


@torch.no_grad()
def sample_with_diffusion(diffusion_model, Y, H, num_steps=16):
    """
    Standard sampling using Diffusion model's _sample() (same as validation_step).
    Used for K=1,2.
    """
    # Set H on the diffusion model (it uses self.H internally)
    diffusion_model.H = H
    # Use _sample() like validation_step does, with random_slot_first=True
    preds = diffusion_model._sample(
        Y,
        num_steps=num_steps,
        use_remasking=False,
        random_slot_first=True,
    )
    return preds


@torch.no_grad()
def sample_first_alone(backbone, Y, H, num_steps=16, temp_start=1.5, temp_end=0.3):
    """
    Sample with first-reveal-alone constraint for K=3,4,5.
    When any slot makes its first reveal, ONLY that token is revealed.
    """
    B = Y.shape[0]
    K = backbone.K
    N = backbone.N
    Q = backbone.Q
    device = Y.device
    mask_token = Q
    total_positions = K * N

    X_t = torch.full((B, K, N), mask_token, device=device, dtype=torch.long)
    has_first_reveal = torch.zeros(B, K, dtype=torch.bool, device=device)

    def unmask_schedule(step, total_steps):
        if total_steps <= 1:
            return 1.0
        progress = step / total_steps
        return 0.5 * (1 - math.cos(math.pi * progress))

    for step in range(1, num_steps + 1):
        num_masked = (X_t == mask_token).sum().item()
        if num_masked == 0:
            break

        num_masked_per_sample = (X_t == mask_token).sum(dim=[1, 2])
        t = num_masked_per_sample.float() / total_positions

        syn = syndrome_over_gfq_batch(H, X_t, Q)
        logits = backbone(X_t, Y, syn, t, H, soft_input=False)

        tau = cosine_anneal(temp_start, temp_end, step, num_steps)
        probs = F.softmax(logits / tau, dim=-1)
        confidence, predictions = probs.max(dim=-1)

        is_masked = (X_t == mask_token)
        conf_masked = torch.where(is_masked, confidence, torch.tensor(-1.0, device=device))

        target_unmasked_frac = unmask_schedule(step, num_steps)
        target_unmasked = int(target_unmasked_frac * total_positions)

        for b in range(B):
            mask_b = is_masked[b]
            num_masked_b = mask_b.sum().item()
            if num_masked_b == 0:
                continue

            already_unmasked = total_positions - num_masked_b
            num_to_unmask = max(1, target_unmasked - already_unmasked)
            num_to_unmask = min(num_to_unmask, num_masked_b)

            conf_flat = conf_masked[b].view(-1)
            pred_flat = predictions[b].view(-1)
            num_candidates = min(num_to_unmask + K * 2, num_masked_b)
            _, top_indices = conf_flat.topk(num_candidates)

            candidates = []
            for idx in top_indices:
                idx = idx.item()
                if conf_flat[idx] > 0:
                    candidates.append((idx // N, idx % N, idx))

            first_reveal_candidate = None
            non_first_candidates = []

            for (k, n, flat_idx) in candidates:
                if not has_first_reveal[b, k]:
                    if first_reveal_candidate is None:
                        first_reveal_candidate = (k, n, flat_idx)
                else:
                    non_first_candidates.append((k, n, flat_idx))

            if first_reveal_candidate is not None:
                k, n, flat_idx = first_reveal_candidate
                X_t[b, k, n] = pred_flat[flat_idx]
                has_first_reveal[b, k] = True
            else:
                for k, n, flat_idx in non_first_candidates[:num_to_unmask]:
                    X_t[b, k, n] = pred_flat[flat_idx]

    # Final pass
    if (X_t == mask_token).any():
        t_final = torch.zeros(B, device=device)
        syn = syndrome_over_gfq_batch(H, X_t, Q)
        final_logits = backbone(X_t, Y, syn, t_final, H, soft_input=False)
        final_preds = final_logits.argmax(dim=-1)
        X_t = torch.where(X_t == mask_token, final_preds, X_t)

    return X_t


@torch.no_grad()
def get_hidden_states(model, y, Y, t, H):
    """Get hidden states from backbone for quality head."""
    B = y.shape[0]
    K, N, D = model.K, model.N, model.D
    Q = model.Q
    syn = torch.zeros(B, K, model.M, device=y.device, dtype=torch.long)

    logits, hidden_flat, _ = model.get_hidden_states(y, Y, syn, t, H, soft_input=False)
    h = hidden_flat.view(B, K, N, D)
    return logits, h


@torch.no_grad()
def run_inference_pass(model, quality_head, y, Y, H, num_steps, has_first_reveal, num_masked_slots=None):
    """
    Run one inference pass (initial or remasking).

    Args:
        num_masked_slots: Number of slots being remasked. If 1, skip first-reveal-alone rule.
    """
    device = Y.device
    B = Y.shape[0]
    K = model.K
    N = model.N
    Q = model.Q
    mask_token = Q
    total_positions = K * N

    # Skip first-reveal-alone if only 1 slot is being remasked
    skip_first_reveal_rule = (num_masked_slots == 1)

    for step in range(1, num_steps + 1):
        is_masked = (y == mask_token)
        num_masked_per_sample = is_masked.sum(dim=(1, 2))  # [B]
        if num_masked_per_sample.sum().item() == 0:
            break

        # Per-sample t
        t = num_masked_per_sample.float() / total_positions  # [B]

        logits, h = get_hidden_states(model, y, Y, t, H)
        probs = F.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

        confidence = torch.where(is_masked, confidence, torch.zeros_like(confidence))

        # Cosine schedule
        if num_masked_slots is not None:
            masked_positions = num_masked_slots * N
        else:
            masked_positions = total_positions
        progress = step / num_steps
        target_frac = 0.5 * (1 - math.cos(math.pi * progress))
        target_unmasked = int(target_frac * masked_positions)

        for b in range(B):
            conf_flat = confidence[b].view(-1)
            mask_flat = is_masked[b].view(-1)
            pred_flat = predictions[b].view(-1)

            masked_indices = mask_flat.nonzero(as_tuple=True)[0]
            if len(masked_indices) == 0:
                continue

            # Per-sample num_to_unmask
            num_masked_b = num_masked_per_sample[b].item()
            already_revealed_b = masked_positions - num_masked_b
            num_to_unmask = max(1, target_unmasked - already_revealed_b)

            masked_conf = conf_flat[masked_indices]
            num_candidates = min(num_to_unmask + K * 2, len(masked_indices))
            _, topk_idx = masked_conf.topk(num_candidates)
            top_indices = masked_indices[topk_idx]

            candidates = []
            for idx in top_indices:
                flat_idx = idx.item()
                slot_k = flat_idx // N
                pos_n = flat_idx % N
                conf = conf_flat[flat_idx].item()
                candidates.append((slot_k, pos_n, conf, flat_idx))

            # First-reveal-alone logic (skip if only 1 slot is being remasked)
            if skip_first_reveal_rule:
                # No first-reveal-alone rule needed for single slot
                unmask_list = candidates[:num_to_unmask]
                for (slot_k, pos_n, conf, flat_idx) in unmask_list:
                    y[b, slot_k, pos_n] = pred_flat[flat_idx]
                    if not has_first_reveal[b][slot_k]:
                        has_first_reveal[b][slot_k] = True
            else:
                first_reveal_candidate = None
                non_first_candidates = []

                for (slot_k, pos_n, conf, flat_idx) in candidates:
                    if not has_first_reveal[b][slot_k]:
                        if first_reveal_candidate is None:
                            first_reveal_candidate = (slot_k, pos_n, flat_idx)
                    else:
                        non_first_candidates.append((slot_k, pos_n, flat_idx))

                if first_reveal_candidate is not None:
                    slot_k, pos_n, flat_idx = first_reveal_candidate
                    y[b, slot_k, pos_n] = pred_flat[flat_idx]
                    has_first_reveal[b][slot_k] = True
                else:
                    unmask_list = non_first_candidates[:num_to_unmask]
                    for (slot_k, pos_n, flat_idx) in unmask_list:
                        y[b, slot_k, pos_n] = pred_flat[flat_idx]

    # Get final quality scores
    t_final = (y == mask_token).sum(dim=(1, 2)).float() / total_positions  # [B]
    _, h_final = get_hidden_states(model, y, Y, t_final, H)
    q_hat = quality_head(h_final)

    return y, q_hat


@torch.no_grad()
def sample_with_quality_head(
    model,
    quality_head,
    Y,
    H,
    num_steps=16,
    thresholds=None,
):
    """
    Sample with multi-stage threshold-based remasking for K=6,7,8.

    For each threshold in thresholds list, remask slots with avg quality < threshold.
    Default: [0.99, 0.97, 0.94, 0.90] (4 iterations)

    Remask steps are dynamic based on number of slots: num_steps * num_slots_remasked / K
    Slot_init is scaled by num_slots_remasked / K during remasking.
    First-reveal-alone rule is disabled when only 1 slot is being remasked.
    """
    if thresholds is None:
        thresholds = [0.99, 0.97, 0.94, 0.90]

    device = Y.device
    B = Y.shape[0]
    K = model.K
    N = model.N
    Q = model.Q
    mask_token = Q

    # Start fully masked
    y = torch.full((B, K, N), mask_token, device=device, dtype=torch.long)
    has_first_reveal = [[False] * K for _ in range(B)]

    # Store original slot_init for scaling
    original_slot_init = model.slot_init.data.clone()

    # Initial inference (all K slots masked, scale = 1.0)
    y, q_hat = run_inference_pass(model, quality_head, y, Y, H, num_steps, has_first_reveal, num_masked_slots=K)

    # Multi-stage remasking with different thresholds
    for threshold in thresholds:
        slots_to_remask_per_batch = []

        for b in range(B):
            slot_qualities = q_hat[b].mean(dim=-1)  # [K]

            # Find slots below threshold
            below_threshold = (slot_qualities < threshold).nonzero(as_tuple=True)[0].tolist()
            slots_to_remask_per_batch.append(below_threshold)

        # Check if any remasking needed
        total_to_remask = sum(len(slots) for slots in slots_to_remask_per_batch)
        if total_to_remask == 0:
            continue  # Skip this iteration but try next threshold

        # For batch processing, use max slots to remask in this batch
        max_slots_to_remask = max(len(slots) for slots in slots_to_remask_per_batch)
        remask_steps = max(1, int(num_steps * max_slots_to_remask / K))

        # Scale slot_init by remask_k / K
        scale = max_slots_to_remask / K
        model.slot_init.data = original_slot_init * scale

        # Remask the identified slots
        for b in range(B):
            for slot_k in slots_to_remask_per_batch[b]:
                y[b, slot_k, :] = mask_token
                has_first_reveal[b][slot_k] = False

        # Run remasking inference with dynamic steps
        y, q_hat = run_inference_pass(
            model, quality_head, y, Y, H, remask_steps, has_first_reveal,
            num_masked_slots=max_slots_to_remask
        )

    # Final pass - fill any remaining masked tokens
    if (y == mask_token).any():
        t_final = torch.zeros(B, device=device)
        logits, _ = get_hidden_states(model, y, Y, t_final, H)
        final_preds = logits.argmax(dim=-1)
        y = torch.where(y == mask_token, final_preds, y)

    # Restore original slot_init
    model.slot_init.data = original_slot_init

    return y


# =============================================================================
# Protocol Simulation
# =============================================================================

def simulate_collisions(num_users, num_preambles=12, seed=None):
    """
    Simulate random preamble selection.

    Returns:
        collision_pattern: dict {slot_idx: num_users_in_slot}
    """
    if seed is not None:
        np.random.seed(seed)

    preamble_choices = np.random.randint(0, num_preambles, size=num_users)
    collision_pattern = defaultdict(int)

    for slot in preamble_choices:
        collision_pattern[slot] += 1

    return dict(collision_pattern)


def evaluate_batch(K, Y_batch, X0_batch, H, checkpoints, device):
    """
    Evaluate a batch of samples with same K.

    Args:
        K: number of users
        Y_batch: [B, N, Q] tensor
        X0_batch: [B, K, N] tensor
        H: parity check matrix
        checkpoints: dict of loaded models
        device: torch device

    Returns:
        correct_symbols: int
        total_symbols: int
        correct_codewords: int
        total_codewords: int
    """
    B = Y_batch.shape[0]

    if K == 0 or B == 0:
        return 0, 0, 0, 0

    if K > 8:
        return None

    if K not in checkpoints:
        return None

    diffusion_model, qhead = checkpoints[K]
    backbone = diffusion_model.backbone
    N = backbone.N

    Y_batch = Y_batch.to(device).float()
    X0_batch = X0_batch.to(device)

    # Get per-K inference steps
    num_steps = INFERENCE_STEPS.get(K, 16)

    # Inference strategy based on K
    if K <= 2:
        preds = sample_with_diffusion(diffusion_model, Y_batch, H, num_steps=num_steps)
    elif K <= 5 or qhead is None:
        preds = sample_first_alone(backbone, Y_batch, H, num_steps=num_steps)
    else:
        qh_params = QUALITY_HEAD_PARAMS.get(K, {})
        preds = sample_with_quality_head(
            backbone, qhead, Y_batch, H,
            num_steps=num_steps,
            thresholds=qh_params.get('thresholds', [0.99, 0.97, 0.94, 0.90]),
        )

    # Hungarian matching for each sample in batch
    correct_symbols = 0
    correct_codewords = 0
    total_symbols = B * K * N
    total_codewords = B * K

    for b in range(B):
        cost = torch.zeros(K, K, device=device)
        for i in range(K):
            for j in range(K):
                cost[i, j] = (preds[b, i] != X0_batch[b, j]).float().sum()

        row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

        for i, j in zip(row_ind, col_ind):
            matches = (preds[b, i] == X0_batch[b, j])
            correct_symbols += matches.sum().item()
            if matches.all():
                correct_codewords += 1

    return correct_symbols, total_symbols, correct_codewords, total_codewords


def evaluate_protocol(
    checkpoints,
    data,
    H,
    device,
    user_counts,
    num_preambles=12,
    num_frames=1000,
    batch_size=32,
    preambles_map=None,
):
    """
    Evaluate protocol over multiple user counts with batching.

    Args:
        preambles_map: optional dict {num_users: num_preambles} overriding the
            scalar `num_preambles` per user count. Useful for keeping mean
            per-slot load constant across K_a.

    Returns:
        results: dict {num_users: {metrics}}
    """
    console = Console()
    results = {}

    for num_users in user_counts:
        slots = preambles_map[num_users] if preambles_map is not None else num_preambles
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Evaluating with {num_users} users across {slots} slots (mean load {num_users/slots:.2f})[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")

        # First pass: simulate all frames and collect samples by K
        samples_by_k = defaultdict(list)  # K -> list of (Y, X0)
        collision_counts = defaultdict(int)
        total_k_overflow = 0          # slot count (K>8)
        total_overflow_users = 0      # user count in overflow slots
        total_slots_processed = 0

        with Progress(
            TextColumn("[bold blue]Collecting"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Frames", total=num_frames)

            for frame_idx in range(num_frames):
                # Deterministic collision pattern per frame (different pattern for each frame)
                collision_pattern = simulate_collisions(num_users, slots, seed=frame_idx)

                for slot_idx, K in collision_pattern.items():
                    collision_counts[K] += 1

                    if K == 0:
                        continue

                    if K > 8:
                        total_k_overflow += 1
                        total_overflow_users += K
                        total_slots_processed += 1
                        continue

                    if K not in data:
                        continue

                    # Random sample from test set
                    Y_all, X0_all = data[K]
                    idx = np.random.randint(0, len(Y_all))
                    samples_by_k[K].append((Y_all[idx], X0_all[idx]))
                    total_slots_processed += 1

                progress.update(task, advance=1)

        # Accumulators
        total_correct_symbols = 0
        total_symbols = 0
        total_correct_codewords = 0
        total_codewords = 0
        per_k_stats = defaultdict(lambda: {'correct_sym': 0, 'total_sym': 0, 'correct_cw': 0, 'total_cw': 0})

        # Print sample distribution
        console.print(f"Samples by K: " + ", ".join(f"K{k}={len(samples_by_k[k])}" for k in sorted(samples_by_k.keys())))
        console.print(f"K>8 overflow: {total_k_overflow}")

        # Second pass: process each K in batches
        total_batches = sum((len(samples) + batch_size - 1) // batch_size for samples in samples_by_k.values())
        console.print(f"Total batches: {total_batches}")

        with Progress(
            TextColumn("[bold blue]Batches"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("[cyan]SER:[/cyan] {task.fields[ser]:.4f}"),
            TextColumn("[cyan]CER:[/cyan] {task.fields[cer]:.4f}"),
        ) as progress:
            task = progress.add_task("Evaluating", total=total_batches, ser=0.0, cer=0.0)

            for K in sorted(samples_by_k.keys()):
                samples = samples_by_k[K]
                num_samples = len(samples)

                for batch_start in range(0, num_samples, batch_size):
                    batch_end = min(batch_start + batch_size, num_samples)
                    batch_samples = samples[batch_start:batch_end]

                    # Stack into tensors
                    Y_batch = torch.stack([s[0] for s in batch_samples])
                    X0_batch = torch.stack([s[1] for s in batch_samples])

                    result = evaluate_batch(K, Y_batch, X0_batch, H, checkpoints, device)

                    if result is not None:
                        correct_sym, total_sym, correct_cw, total_cw = result

                        total_correct_symbols += correct_sym
                        total_symbols += total_sym
                        total_correct_codewords += correct_cw
                        total_codewords += total_cw

                        per_k_stats[K]['correct_sym'] += correct_sym
                        per_k_stats[K]['total_sym'] += total_sym
                        per_k_stats[K]['correct_cw'] += correct_cw
                        per_k_stats[K]['total_cw'] += total_cw

                    # Update progress
                    ser = 1.0 - (total_correct_symbols / total_symbols) if total_symbols > 0 else 0.0
                    cer = 1.0 - (total_correct_codewords / total_codewords) if total_codewords > 0 else 0.0
                    progress.update(task, advance=1, ser=ser, cer=cer)

        # Compute final metrics
        ser = 1.0 - (total_correct_symbols / total_symbols) if total_symbols > 0 else 0.0
        cer = 1.0 - (total_correct_codewords / total_codewords) if total_codewords > 0 else 0.0
        k_overflow_rate = total_k_overflow / total_slots_processed if total_slots_processed > 0 else 0.0

        # System-level metrics: USER-WEIGHTED (every colliding user is one trial,
        # overflow users count as 100% failure — matches PUPE convention).
        N_code = int(H.shape[1])
        overflow_symbols = total_overflow_users * N_code
        sys_total_codewords = total_codewords + total_overflow_users
        sys_total_symbols = total_symbols + overflow_symbols
        failed_codewords = (total_codewords - total_correct_codewords) + total_overflow_users
        failed_symbols = (total_symbols - total_correct_symbols) + overflow_symbols
        system_cer = failed_codewords / sys_total_codewords if sys_total_codewords > 0 else 0.0
        system_ser = failed_symbols / sys_total_symbols if sys_total_symbols > 0 else 0.0
        user_overflow_rate = total_overflow_users / sys_total_codewords if sys_total_codewords > 0 else 0.0

        results[num_users] = {
            'num_preambles': slots,
            'ser': ser,
            'cer': cer,
            'system_ser': system_ser,
            'system_cer': system_cer,
            'k_overflow_rate': k_overflow_rate,           # slot-weighted
            'user_overflow_rate': user_overflow_rate,     # user-weighted
            'total_symbols': total_symbols,
            'total_codewords': total_codewords,
            'total_overflow_users': total_overflow_users,
            'per_k_stats': dict(per_k_stats),
            'collision_counts': dict(collision_counts),
        }

        # Print per-K breakdown
        console.print("\n[bold]Per-K Breakdown:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("K", justify="right")
        table.add_column("Slots", justify="right")
        table.add_column("SER", justify="right")
        table.add_column("CER", justify="right")

        for K in sorted(per_k_stats.keys()):
            stats = per_k_stats[K]
            k_ser = 1.0 - (stats['correct_sym'] / stats['total_sym']) if stats['total_sym'] > 0 else 0.0
            k_cer = 1.0 - (stats['correct_cw'] / stats['total_cw']) if stats['total_cw'] > 0 else 0.0
            table.add_row(
                str(K),
                str(collision_counts[K]),
                f"{k_ser:.6f}",
                f"{k_cer:.6f}",
            )

        if total_k_overflow > 0:
            table.add_row(">8", str(total_k_overflow), "FAIL", "FAIL")

        console.print(table)

        console.print(f"\n[bold]Overall (K<=8 only):[/bold]")
        console.print(f"  SER: {ser:.6f} ({ser*100:.4f}%)")
        console.print(f"  CER: {cer:.6f} ({cer*100:.4f}%)")
        console.print(f"\n[bold]System-Level (user-weighted; K>8 users = 100% failure):[/bold]")
        console.print(f"  K>8 Slot Rate:    {k_overflow_rate:.4f} ({k_overflow_rate*100:.2f}%)")
        console.print(f"  K>8 User Rate:    {user_overflow_rate:.4f} ({user_overflow_rate*100:.2f}%)")
        console.print(f"  System SER:       {system_ser:.6f} ({system_ser*100:.4f}%)")
        console.print(f"  System CER (PUPE):{system_cer:.6f} ({system_cer*100:.4f}%)")

    return results


def print_summary(results, console):
    """Print final summary table."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print("[bold cyan]SUMMARY[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Users", justify="right")
    table.add_column("SER", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("K>8 Fail", justify="right")
    table.add_column("Sys SER", justify="right")
    table.add_column("Sys CER", justify="right")

    for num_users in sorted(results.keys()):
        r = results[num_users]
        table.add_row(
            str(num_users),
            f"{r['ser']:.6f}",
            f"{r['cer']:.6f}",
            f"{r['k_overflow_rate']*100:.2f}%",
            f"{r['system_ser']:.6f}",
            f"{r['system_cer']:.6f}",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Protocol-level evaluation for random access")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/protocol_scale',
                        help='Directory containing K{1..8}/best_model.ckpt')
    parser.add_argument('--data_dir', type=str, default='data/gen_data/datasets/protocol_Eb10',
                        help='Directory containing K{1..8}/test_data.pt and K{#}/H_matrix.pt')
    parser.add_argument('--user_counts', type=int, nargs='+', default=[10, 15, 20, 25, 30],
                        help='Number of users to evaluate')
    parser.add_argument('--num_preambles', type=int, default=12,
                        help='Number of ZC preambles/slots (ignored if --target_load or --preambles_per_count is set)')
    parser.add_argument('--target_load', type=float, default=None,
                        help='If set, num_preambles per K_a is ceil(K_a / target_load) — keeps mean per-slot load constant')
    parser.add_argument('--preambles_per_count', type=int, nargs='+', default=None,
                        help='Explicit per-K_a slot counts (must match length of --user_counts)')
    parser.add_argument('--num_frames', type=int, default=1000,
                        help='Number of frames per user count')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--K_range', type=int, nargs=2, default=[1, 9],
                        help='K range [min, max) for loading checkpoints (default: 1 9)')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save results (JSON)')
    args = parser.parse_args()

    console = Console()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[bold]Device: {device}[/bold]")

    # Load checkpoints
    console.print("\n[bold]Loading checkpoints...[/bold]")
    checkpoints = load_all_checkpoints(args.checkpoint_dir, device, K_range=tuple(args.K_range))

    if not checkpoints:
        console.print("[red]No checkpoints found![/red]")
        return

    # Load data
    console.print("\n[bold]Loading data...[/bold]")
    data, H = load_all_data(args.data_dir)

    if not data:
        console.print("[red]No data found![/red]")
        return

    H = H.to(device)

    # Print config
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print("[bold cyan]PROTOCOL EVALUATION[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"Checkpoints: {args.checkpoint_dir}")
    console.print(f"Data: {args.data_dir}")
    # Build per-K_a preambles map if adaptive mode requested
    preambles_map = None
    if args.preambles_per_count is not None:
        if len(args.preambles_per_count) != len(args.user_counts):
            console.print("[red]--preambles_per_count length must match --user_counts[/red]")
            return
        preambles_map = dict(zip(args.user_counts, args.preambles_per_count))
    elif args.target_load is not None:
        preambles_map = {K_a: max(1, math.ceil(K_a / args.target_load)) for K_a in args.user_counts}

    console.print(f"User counts: {args.user_counts}")
    if preambles_map is not None:
        console.print(f"Preambles per K_a: {preambles_map}")
    else:
        console.print(f"Preambles: {args.num_preambles}")
    console.print(f"Frames per count: {args.num_frames}")
    console.print(f"Batch size: {args.batch_size}")
    console.print(f"Available K: {sorted(checkpoints.keys())}")
    console.print(f"Inference steps per K: {INFERENCE_STEPS}")

    # Run evaluation
    results = evaluate_protocol(
        checkpoints=checkpoints,
        data=data,
        H=H,
        device=device,
        user_counts=args.user_counts,
        num_preambles=args.num_preambles,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        preambles_map=preambles_map,
    )

    # Print summary
    print_summary(results, console)

    # Save results
    if args.save_results:
        import json
        # Create parent directory if needed
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert to serializable format
        save_data = {}
        for num_users, r in results.items():
            save_data[num_users] = {
                'num_preambles': r['num_preambles'],
                'ser': r['ser'],
                'cer': r['cer'],
                'system_ser': r['system_ser'],
                'system_cer': r['system_cer'],
                'k_overflow_rate': r['k_overflow_rate'],
                'user_overflow_rate': r['user_overflow_rate'],
                'total_symbols': r['total_symbols'],
                'total_codewords': r['total_codewords'],
                'total_overflow_users': r['total_overflow_users'],
            }
        with open(args.save_results, 'w') as f:
            json.dump(save_data, f, indent=2)
        console.print(f"\n[bold green]Results saved to {args.save_results}[/bold green]")


if __name__ == '__main__':
    main()

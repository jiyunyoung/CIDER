"""
Diffusion Module.

PyTorch Lightning wrapper with:
- MaskGIT-style training with curriculum
- Hungarian loss for permutation-invariant matching
- EMA for stable inference
- PRISM adapter for self-correction (2-step training)

Training Workflow:
  Step 1: Train backbone (mode=train)
  Step 2: Train PRISM adapter on frozen backbone (mode=prism_adapter)
"""

import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from torch.optim import AdamW
from scipy.optimize import linear_sum_assignment

import models
BACKBONE_CLASSES={'cider': models.CIDER, 'dimp_no_gru': models.CIDER, 'cider_noA': models.CIDER_NoSlot, 'dimp_no_gru_no_slot': models.CIDER_NoSlot, 'cider_noB': models.CIDER_NoMP, 'dimp_no_gru_no_mp': models.CIDER_NoMP, 'mdd': models.DiT, 'cider_gru': models.CIDER_GRU, 'dimp_rev2': models.CIDER_GRU}
from models.ema import ExponentialMovingAverage
from utils.gf import syndrome_over_gfq_batch


# =============================================================================
# Masking Utilities
# =============================================================================

def _sample_categorical(categorical_probs):
    """Sample from categorical distribution using Gumbel-max trick."""
    categorical_probs = categorical_probs.to(torch.float64)
    gumbel_norm = (
        1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    )
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def hard_to_soft(X_t: torch.Tensor, Q: int, mask_token: int) -> torch.Tensor:
    """
    Convert hard symbols (with MASK) to soft beliefs.

    Args:
        X_t: [B, K, N] hard symbols (0 to Q-1, or mask_token)
        Q: vocabulary size
        mask_token: MASK token index (usually Q)

    Returns:
        soft: [B, K, N, Q] soft beliefs
            - Non-MASK positions: one-hot
            - MASK positions: uniform 1/Q
    """
    B, K, N = X_t.shape
    device = X_t.device

    # Initialize with zeros
    soft = torch.zeros(B, K, N, Q, device=device)

    # MASK positions get uniform distribution
    is_mask = (X_t == mask_token)
    soft[is_mask] = 1.0 / Q

    # Non-MASK positions get one-hot
    not_mask = ~is_mask
    X_clamped = X_t.clamp(0, Q - 1)  # Clamp for safety
    soft.scatter_(-1, X_clamped.unsqueeze(-1), 1.0)
    # But masked positions should stay uniform, so re-apply
    soft[is_mask] = 1.0 / Q

    return soft


def mask_corrupt_with_gamma(X0, gamma_t, mask_token):
    """
    Mask corruption with gamma (masking ratio) provided directly.

    Args:
        X0: [B, K, N] ground truth symbols
        gamma_t: [B] masking ratio per sample (0=no mask, 1=all masked)
        mask_token: token to use for masking

    Returns:
        X_t: [B, K, N] masked symbols
        mask: [B, K, N] boolean mask (True = masked)
    """
    B, K, N = X0.shape
    device = X0.device

    gamma_t = gamma_t.view(B, 1, 1)
    mask = torch.rand(B, K, N, device=device) < gamma_t

    X_t = X0.clone()
    X_t[mask] = mask_token

    return X_t, mask


def complementary_mask_corrupt_with_gamma(X0, gamma_t, mask_token):
    """
    MAE-style complementary pair masking.

    For each sample, generates TWO views with complementary masks:
      - View 1: mask1 ~ Bernoulli(gamma) per (k, n)
      - View 2: mask2 = NOT mask1 (complement)

    Every (b, k, n) is masked in EXACTLY one view → summing the loss across
    both views gives gradient signal at every position. Doubles per-step
    compute (2 forward passes worth of work) but maximizes data efficiency
    and gives the model both halves of every (slot, position) for the same
    underlying X0.

    Output is concatenated along batch dim: [view1 ; view2]. The caller
    must also double Y, gamma, and X0 to match the [2B, ...] shape.

    Args:
        X0: [B, K, N] ground truth symbols
        gamma_t: [B] mask probability per (k, n) for view 1
        mask_token: token used for masking

    Returns:
        X_t: [2B, K, N] — concatenated [view1 ; view2]
        mask: [2B, K, N] — concatenated [mask1 ; mask2 = ~mask1]
    """
    B, K, N = X0.shape
    device = X0.device

    gamma_t_view = gamma_t.view(B, 1, 1)
    mask1 = torch.rand(B, K, N, device=device) < gamma_t_view
    mask2 = ~mask1

    X_t1 = X0.clone()
    X_t1[mask1] = mask_token
    X_t2 = X0.clone()
    X_t2[mask2] = mask_token

    X_t = torch.cat([X_t1, X_t2], dim=0)   # [2B, K, N]
    mask = torch.cat([mask1, mask2], dim=0)  # [2B, K, N]
    return X_t, mask


def cosine_anneal(start, end, t, T):
    """Cosine annealing schedule for inference."""
    if T <= 1:
        return end
    alpha = (t - 1) / (T - 1)
    weight = 0.5 * (1 + math.cos(math.pi * (1 - alpha)))
    return start * weight + end * (1 - weight)


# =============================================================================
# Hungarian Loss
# =============================================================================

def hungarian_loss(logits, X0, mask0):
    """
    Permutation-invariant loss using Hungarian algorithm.

    Matching: Use GT-REVEALED positions (visible in ANY row) for slot-user alignment
    - Same positions used for all row-GT comparisons (no scale mismatch)
    - Only GT anchors, no model-generated positions

    Loss: Apply only to MASKED positions (mask0)
    - Where the model must predict the original tokens

    Args:
        logits: [B, K, N, Q] predicted logits
        X0: [B, K, N] ground truth symbols
        mask0: [B, K, N] mask (True = corrupted/masked, loss target)

    Returns:
        scalar loss
    """
    B, K, N, Q = logits.shape
    device = logits.device
    total_loss = 0.0

    for b in range(B):
        logits_b = logits[b]
        X0_b = X0[b]

        # Positions visible in ANY row (union of GT-revealed)
        visible = ~mask0[b]  # [K, N]
        any_visible = visible.any(dim=0)  # [N]

        # Fallback: if no positions visible anywhere, use all
        if any_visible.sum() == 0:
            any_visible = torch.ones(N, dtype=torch.bool, device=device)

        # Build cost matrix using SAME positions for all comparisons
        cost = torch.zeros(K, K, device=device)
        for i in range(K):
            for u in range(K):
                cost[i, u] = F.cross_entropy(logits_b[i, any_visible], X0_b[u, any_visible])

        # Safety check for NaN/Inf
        if torch.isnan(cost).any() or torch.isinf(cost).any():
            cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=100.0)

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        # Compute loss ONLY on masked positions
        loss_b = 0.0
        for i, u in zip(row_ind, col_ind):
            pos = mask0[b, i]
            if pos.sum() > 0:
                loss_b = loss_b + F.cross_entropy(logits_b[i, pos], X0_b[u, pos])

        total_loss = total_loss + loss_b / K

    return total_loss / B


def direct_slot_loss(logits, X0, mask0):
    """
    Direct per-slot CE — no Hungarian matching.

    Used when slot↔target binding is fixed at the data side (e.g., per-slot
    fixed permutations: slot k always predicts the codeword permuted by π_k).
    Loss is computed only on masked positions, summed across slots, averaged
    by per-slot active-mask count then by batch.
    """
    B, K, N, Q = logits.shape
    total_loss = 0.0
    for b in range(B):
        loss_b = 0.0
        for k in range(K):
            pos = mask0[b, k]
            if pos.sum() > 0:
                loss_b = loss_b + F.cross_entropy(logits[b, k, pos], X0[b, k, pos])
        total_loss = total_loss + loss_b / K
    return total_loss / B


def demixing_loss(logits, mask0, mode='cosine'):
    """
    Auxiliary loss to encourage slot diversity (demixing).

    Penalizes when different slots have similar predictions at the same position.
    Only computed on masked positions where model is actually predicting.

    Args:
        logits: [B, K, N, Q] predicted logits
        mask0: [B, K, N] mask (True = masked positions)
        mode: 'cosine' (cosine similarity) or 'kl' (symmetric KL divergence)

    Returns:
        scalar loss (higher = more similar slots = bad)
    """
    B, K, N, Q = logits.shape
    device = logits.device

    if K < 2:
        return torch.tensor(0.0, device=device)

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)  # [B, K, N, Q]

    total_loss = 0.0
    num_pairs = 0

    for b in range(B):
        # Positions masked in ALL slots (where all slots are predicting)
        all_masked = mask0[b].all(dim=0)  # [N]
        if all_masked.sum() == 0:
            continue

        probs_b = probs[b, :, all_masked, :]  # [K, N_masked, Q]

        # Compute pairwise similarity between slots
        for i in range(K):
            for j in range(i + 1, K):
                if mode == 'cosine':
                    # Cosine similarity (flatten across positions)
                    p_i = probs_b[i].reshape(-1)  # [N_masked * Q]
                    p_j = probs_b[j].reshape(-1)
                    sim = F.cosine_similarity(p_i.unsqueeze(0), p_j.unsqueeze(0))
                    total_loss = total_loss + sim.squeeze()
                elif mode == 'kl':
                    # Symmetric KL divergence (per position, then mean)
                    p_i = probs_b[i]  # [N_masked, Q]
                    p_j = probs_b[j]
                    kl_ij = F.kl_div(p_i.log(), p_j, reduction='batchmean')
                    kl_ji = F.kl_div(p_j.log(), p_i, reduction='batchmean')
                    # Negative KL (we want to maximize divergence = minimize negative KL)
                    total_loss = total_loss - 0.5 * (kl_ij + kl_ji)
                num_pairs += 1

    if num_pairs == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / num_pairs


def parity_loss(logits, H, Q):
    """
    Auxiliary loss to encourage parity constraint satisfaction.

    Soft parity: compute expected syndrome from logits and penalize non-zero.
    For GF(Q), parity check is: sum_j (H[m,j] * x[j]) = 0 mod Q for each check m.

    In soft form: for each check m and each possible syndrome value s,
    compute the probability that the syndrome equals s, and penalize s != 0.

    Simplified version: use argmax predictions and compute hard syndrome violation.

    Args:
        logits: [B, K, N, Q] predicted logits
        H: [M, N] parity check matrix in GF(Q)
        Q: field size

    Returns:
        scalar loss (fraction of violated parity checks)
    """
    B, K, N, Q_logits = logits.shape
    device = logits.device

    # Get hard predictions
    preds = logits.argmax(dim=-1)  # [B, K, N]

    # Compute syndrome for each slot
    # H: [M, N], preds: [B, K, N]
    M = H.shape[0]
    H_expanded = H.unsqueeze(0).unsqueeze(0).expand(B, K, M, N)  # [B, K, M, N]
    preds_expanded = preds.unsqueeze(2).expand(B, K, M, N)  # [B, K, M, N]

    # Syndrome: sum of (H * x) mod Q for each check
    syndrome = (H_expanded * preds_expanded).sum(dim=-1) % Q  # [B, K, M]

    # Fraction of non-zero syndrome entries (parity violations)
    violations = (syndrome != 0).float().mean()

    return violations


def soft_parity_loss(logits, H, Q):
    """
    Differentiable parity loss using soft expectations.

    For each check m: sum_j H[m,j] * E[x_j] should be close to 0 (mod Q).
    We compute expected symbol value and penalize deviation from valid codewords.

    Args:
        logits: [B, K, N, Q] predicted logits
        H: [M, N] parity check matrix in GF(Q)
        Q: field size

    Returns:
        scalar loss
    """
    B, K, N, Q_logits = logits.shape
    device = logits.device
    M = H.shape[0]

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)  # [B, K, N, Q]

    # Expected symbol value: sum_a (a * p(a))
    symbol_values = torch.arange(Q, device=device, dtype=torch.float)  # [Q]
    expected_x = (probs * symbol_values).sum(dim=-1)  # [B, K, N]

    # Compute expected syndrome
    # H: [M, N], expected_x: [B, K, N]
    H_float = H.float()  # [M, N]

    # syndrome = H @ expected_x^T for each (b, k)
    # expected_x: [B, K, N] -> [B*K, N]
    expected_x_flat = expected_x.view(B * K, N)  # [B*K, N]
    syndrome = torch.matmul(expected_x_flat, H_float.T)  # [B*K, M]
    syndrome = syndrome.view(B, K, M)  # [B, K, M]

    # Penalize deviation from 0 (mod Q)
    # Use soft modulo: penalize distance to nearest multiple of Q
    # syndrome_mod = syndrome - Q * round(syndrome / Q)
    syndrome_mod = syndrome - Q * torch.round(syndrome / Q)

    # L2 penalty on syndrome residual
    loss = (syndrome_mod ** 2).mean()

    return loss


# =============================================================================
# Diffusion Lightning Module
# =============================================================================

class Diffusion(L.LightningModule):
    """
    PyTorch Lightning module for masked diffusion ECC demixing.

    Backbone: DiT (simple additive fusion, like original MDM)

    Features:
    - MaskGIT-style training with curriculum
    - EMA for stable inference

    Training Modes:
    - Normal: Train backbone with Hungarian loss
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Data parameters
        self.Q = config.data.Q
        self.N = config.data.N
        self.K = config.data.K_max
        self.M = config.data.get('M', self.N)
        self.mask_index = self.Q  # MASK token = Q

        # Backbone selection
        backbone_type = config.model.get('backbone_type', 'CIDER')
        print(f"[Diffusion] Loading backbone: {backbone_type}")

        if backbone_type not in BACKBONE_CLASSES:
            raise ValueError(f"Unknown backbone_type: {backbone_type}. Available: {list(BACKBONE_CLASSES.keys())}")

        BackboneClass = BACKBONE_CLASSES[backbone_type]

        self.backbone = BackboneClass(
            Q=self.Q,
            N=self.N,
            K=self.K,
            M=self.M,
            D_model=config.model.get('D_model', 256),
            num_layers=config.model.get('num_layers', 8),
            heads=config.model.get('heads', 8),
            mlp_ratio=config.model.get('mlp_ratio', 4),
            dropout=config.model.get('dropout', 0.1),
            tau_min=config.model.get('tau_min', 0.2),
            slot_init_scale=config.model.get('slot_init_scale', 0.5),
            perm_pool_size=config.model.get('perm_pool_size', None),
        )

        # H matrix (parity-check matrix, set externally)
        self.register_buffer('H', None)

        # EMA
        self.ema = None
        self.use_ema = config.training.get('ema', 0) > 0
        self.ema_decay = config.training.get('ema', 0.9999)

        # Masking schedule parameters
        self.gamma_min = config.model.get('gamma_min', 0.1)
        self.gamma_max = config.model.get('gamma_max', 1.0)

        # Sampling config
        self.sampling_eps = config.training.get('sampling_eps', 1e-5)

    def set_H_matrix(self, H):
        """Set the parity-check matrix."""
        if isinstance(H, np.ndarray):
            H = torch.from_numpy(H).long()
        self.register_buffer('H', H.long())

    def on_load_checkpoint(self, checkpoint):
        """Load EMA state from checkpoint."""
        if self.ema and 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])

    def on_save_checkpoint(self, checkpoint):
        """Save EMA state to checkpoint."""
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()

    def on_train_start(self):
        """Initialize EMA when training starts."""
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                self.backbone,
                decay=self.ema_decay,
            )
            self.ema.to(self.device)

    def optimizer_step(self, *args, **kwargs):
        """Update EMA after optimizer step."""
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self.backbone)

    def forward(self, X_t, Y_mag, t, soft_input=False, perm_indices=None):
        """
        Forward pass through backbone.

        Args:
            X_t: [B, K, N] hard symbols or [B, K, N, Q] soft beliefs
            Y_mag: [B, N, Q] soft scores from inner decoder
            t: [B] timestep (masking ratio)
            soft_input: If True, X_t is soft beliefs [B, K, N, Q]
            perm_indices: [B, K] long, optional — per-sample slot↔perm-pool
                          assignment (for cider_binary). Backbones that don't
                          consume this argument simply ignore it.

        Returns:
            logits: [B, K, N, Q]
        """
        assert self.H is not None, "H matrix not set. Call set_H_matrix first."

        # Compute syndrome (only for hard input; soft input uses learned CN init)
        if soft_input:
            B, K, N, Q = X_t.shape
            syn = torch.zeros(B, K, self.M, device=X_t.device, dtype=torch.long)
        else:
            syn = syndrome_over_gfq_batch(self.H, X_t, self.Q)  # [B, K, M]

        # Forward to backbone — only pass perm_indices if backbone accepts it.
        if perm_indices is not None:
            try:
                return self.backbone(X_t, Y_mag, syn, t, self.H,
                                     soft_input=soft_input,
                                     perm_indices=perm_indices)
            except TypeError:
                # Backbone doesn't accept perm_indices — fall through
                pass
        return self.backbone(X_t, Y_mag, syn, t, self.H, soft_input=soft_input)

    def _sample_mask_ratio(self, B, epoch):
        """
        Canonical diffusion curriculum with warmup.

        Timestep t ∈ {0, 1, ..., T}:
          - t=0: fully masked (gamma=1.0)
          - t=T: minimal masking (gamma=gamma_min)

        Warmup (epochs 0 to warmup_epochs-1):
          - t ~ Uniform({1, ..., T})  # exclude fully masked

        After warmup:
          - t ~ Uniform({0, ..., T})  # include fully masked
          - p(t=0) ≈ mask_p_full (10-25% typical)
        """
        device = self.device

        # Curriculum parameters
        warmup_epochs = self.config.model.get('mask_warmup_epochs', 4)
        gamma_min = self.config.model.get('mask_gamma_min', 0.1)
        gamma_max = self.config.model.get('mask_gamma_max', 1.0)
        mask_p_full = self.config.model.get('mask_p_full', 0.15)
        T = self.config.model.get('inference_steps', 16)

        if epoch < warmup_epochs:
            # Warmup: exclude t=0 (fully masked)
            t = torch.randint(1, T + 1, (B,), device=device).float()
        else:
            # After warmup: include t=0 with probability mask_p_full
            t = torch.randint(0, T + 1, (B,), device=device).float()
            # Optionally bias toward t=0 less often (already ~1/(T+1) ≈ 6%)
            # If mask_p_full > 1/(T+1), we could add more t=0 samples
            # For now, uniform is fine since 1/17 ≈ 6% < 15%

        # Convert timestep to gamma: t=0 → gamma=1.0, t=T → gamma=gamma_min
        # gamma(t) = gamma_max - (gamma_max - gamma_min) * (t / T)
        gamma_t = gamma_max - (gamma_max - gamma_min) * (t / T)
        gamma_t = torch.clamp(gamma_t, gamma_min, gamma_max)

        return gamma_t

    # =========================================================================
    # Backbone Training (Step 1)
    # =========================================================================

    def _backbone_loss(self, batch):
        """
        Masked prediction training with configurable corruption.

        Corruption types:
        - 'mask': 100% MASK tokens (default)
        - 'bert': 80% MASK, 10% random, 10% unchanged

        1. Corrupt X0 -> X_t
        2. Convert to soft beliefs (MASK -> uniform, non-MASK -> one-hot)
        3. Forward pass with soft input
        4. Hungarian loss: match on GT-revealed, loss on corrupted
        """
        # Batch may be 2-tuple (Y, X0) or 3-tuple (Y, X0, perm_indices)
        if len(batch) == 3:
            Y, X0, perm_indices = batch
        else:
            Y, X0 = batch
            perm_indices = None
        B = X0.shape[0]

        # Sample mask ratio with curriculum
        gamma = self._sample_mask_ratio(B, self.current_epoch)

        # Corrupt X0 -> X_t (configurable corruption type)
        corruption_type = self.config.model.get('corruption_type', 'mask')
        complementary = self.config.model.get('complementary_masking', False)
        if corruption_type == 'bert':
            X_t, mask = bert_corrupt_with_gamma(X0, gamma, self.mask_index, self.Q)
        elif complementary:
            # MAE-style pair: X_t and mask come back doubled to [2B, ...].
            # View 1 has mask density gamma; view 2 (the complement) has
            # density 1 - gamma. The model's adaLN time conditioning uses
            # gamma as the diffusion timestep, so view 2 must receive 1 - gamma
            # (NOT gamma) — otherwise the model is told the wrong noise level.
            X_t, mask = complementary_mask_corrupt_with_gamma(X0, gamma, self.mask_index)
            Y = torch.cat([Y, Y], dim=0)
            gamma = torch.cat([gamma, 1.0 - gamma], dim=0)
            X0 = torch.cat([X0, X0], dim=0)
            if perm_indices is not None:
                perm_indices = torch.cat([perm_indices, perm_indices], dim=0)
        else:
            X_t, mask = mask_corrupt_with_gamma(X0, gamma, self.mask_index)

        # Convert to soft beliefs: MASK -> uniform, non-MASK -> one-hot
        use_soft_input = self.config.model.get('use_soft_input', True)
        if use_soft_input:
            X_soft = hard_to_soft(X_t, self.Q, self.mask_index)
            logits = self(X_soft, Y.float(), gamma, soft_input=True, perm_indices=perm_indices)
        else:
            logits = self(X_t, Y.float(), gamma, soft_input=False, perm_indices=perm_indices)

        # Loss: Hungarian matching by default (permutation-invariant) OR direct
        # per-slot CE when use_hungarian=False (e.g., binary path with per-slot
        # fixed permutations — slot↔target binding is fixed at the data side).
        use_hungarian = self.config.model.get('use_hungarian', True)
        if use_hungarian:
            loss = hungarian_loss(logits, X0, mask0=mask)
        else:
            loss = direct_slot_loss(logits, X0, mask0=mask)
        self.log('train/ce_loss', loss, prog_bar=False)

        # Auxiliary losses (optional regularization)
        demix_weight = self.config.model.get('demixing_loss_weight', 0.0)
        parity_weight = self.config.model.get('parity_loss_weight', 0.0)

        if demix_weight > 0:
            demix_mode = self.config.model.get('demixing_loss_mode', 'cosine')
            loss_demix = demixing_loss(logits, mask, mode=demix_mode)
            loss = loss + demix_weight * loss_demix
            self.log('train/demix_loss', loss_demix, prog_bar=False)

        if parity_weight > 0:
            parity_mode = self.config.model.get('parity_loss_mode', 'soft')
            if parity_mode == 'soft':
                loss_parity = soft_parity_loss(logits, self.H, self.Q)
            else:
                loss_parity = parity_loss(logits, self.H, self.Q)
            loss = loss + parity_weight * loss_parity
            self.log('train/parity_loss', loss_parity, prog_bar=False)

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/gamma_mean', gamma.mean(), prog_bar=False)

        # Log debug embedding ratios if available (dit_new)
        if hasattr(self.backbone, '_debug_emb_ratios'):
            ratios = self.backbone._debug_emb_ratios
            self.log('train/emb_obs_sym', ratios.get('obs/sym', 0), prog_bar=False)
            self.log('train/emb_par_sym', ratios.get('par/sym', 0), prog_bar=False)

        return loss

    # =========================================================================

    def training_step(self, batch, batch_idx):
        return self._backbone_loss(batch)

    # =========================================================================
    # Validation
    # =========================================================================

    def on_validation_epoch_start(self):
        """Switch to EMA weights for validation and reset accumulators."""
        if self.ema:
            self.ema.store(self.backbone)
            self.ema.copy_to(self.backbone)
        self.backbone.eval()

        # Reset epoch-level accumulators for true micro averaging
        self._val_correct_symbols = 0
        self._val_total_symbols = 0
        self._val_correct_codewords = 0
        self._val_total_codewords = 0

    def on_validation_epoch_end(self):
        """Compute true micro-averaged metrics and restore weights."""
        # Symbol-level accuracy
        if self._val_total_symbols > 0:
            accuracy = self._val_correct_symbols / self._val_total_symbols
        else:
            accuracy = 0.0

        # Codeword-level micro recall: sum(TP) / sum(TP + FN)
        if self._val_total_codewords > 0:
            micro_recall = self._val_correct_codewords / self._val_total_codewords
        else:
            micro_recall = 0.0

        # Log epoch-level metrics
        self.log('val/accuracy', accuracy, prog_bar=True)
        self.log('val/micro_recall', micro_recall, prog_bar=True)

        # Restore training weights
        if self.ema:
            self.ema.restore(self.backbone)

    def validation_step(self, batch, batch_idx):
        """Validation step with iterative denoising."""
        # Batch may be 2-tuple (Y, X0) or 3-tuple (Y, X0, perm_indices)
        if len(batch) == 3:
            Y, X0, perm_indices = batch
        else:
            Y, X0 = batch
            perm_indices = None
        B = X0.shape[0]

        # Inference steps: respect config value by default. If
        # adaptive_inference_steps=true, bump up by max(base, K*N//10).
        base_steps = self.config.model.get('inference_steps', 16)
        if self.config.model.get('adaptive_inference_steps', False):
            T_steps = max(base_steps, self.K * self.N // 10)
        else:
            T_steps = base_steps

        preds = self._sample(Y.float(), num_steps=T_steps, use_remasking=False,
                             perm_indices=perm_indices)

        # Compute metrics: Hungarian by default; direct per-slot match when off
        use_hungarian = self.config.model.get('use_hungarian', True)
        for b in range(B):
            if use_hungarian:
                cost = torch.zeros(self.K, self.K, device=self.device)
                for i in range(self.K):
                    for u in range(self.K):
                        cost[i, u] = (preds[b, i] != X0[b, u]).float().sum()
                row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
                pairs = list(zip(row_ind, col_ind))
            else:
                # Direct slot↔target binding (slot k ↔ target k)
                pairs = [(i, i) for i in range(self.K)]

            for i, u in pairs:
                pred_cw = preds[b, i]
                gt_cw = X0[b, u]
                matches = (pred_cw == gt_cw)

                self._val_correct_symbols += matches.sum().item()
                self._val_total_symbols += self.N

                if matches.all():
                    self._val_correct_codewords += 1
                self._val_total_codewords += 1

    # =========================================================================
    # Sampling / Inference
    # =========================================================================

    @torch.no_grad()
    def _sample(self, Y_mag, num_steps=None, use_remasking=False,
                random_slot_first=None, perm_indices=None):
        """
        Generate samples via iterative denoising.

        Routes to discrete or soft sampling based on config.

        Args:
            random_slot_first: If True, break symmetry by revealing random slot first.
                              If None, reads from config.random_slot_first (default True)
            perm_indices: [B, K] long, optional — per-sample slot↔perm-pool
                          assignment passed through to backbone.
        """
        # Get random_slot_first from config if not explicitly passed
        if random_slot_first is None:
            random_slot_first = self.config.get('random_slot_first', True)

        use_soft_input = self.config.model.get('use_soft_input', False)

        if use_soft_input:
            return self._sample_soft(Y_mag, num_steps, use_remasking,
                                     perm_indices=perm_indices)
        else:
            return self._sample_discrete(Y_mag, num_steps,
                                         random_slot_first=random_slot_first,
                                         perm_indices=perm_indices)

    @torch.no_grad()
    def _sample_discrete(self, Y_mag, num_steps=None, random_slot_first=True,
                         perm_indices=None):
        """
        Discrete MASK-based sampling for dit_new with temperature annealing.

        1. Start with all MASK tokens
        2. Predict logits with temperature annealing (high→low)
        3. Unmask most confident positions (fewer early, more later)
        4. Repeat until all unmasked

        Temperature annealing: Early steps are exploratory (high temp),
        later steps are confident (low temp). This prevents early lock-in.

        Args:
            random_slot_first: If True, break symmetry by revealing random slot first (default True)
            perm_indices: [B, K] long, optional — passed through to backbone.
        """
        if num_steps is None:
            num_steps = self.config.model.get('inference_steps', 16)

        B = Y_mag.shape[0]
        K = self.K
        N = self.N
        device = Y_mag.device

        # Initialize with all MASK tokens
        X_t = torch.full((B, K, N), self.mask_index, device=device, dtype=torch.long)

        # Total positions to unmask
        total_positions = K * N

        # Temperature schedule: start high (exploration), end low (exploitation)
        temp_start = self.config.model.get('sample_temp_start', 1.5)
        temp_end = self.config.model.get('sample_temp_end', 0.3)

        # Unmasking schedule: cosine (unmask fewer early, more later)
        # This gives the model more context before making hard decisions
        def unmask_schedule(step, total_steps):
            """Cosine schedule: slow start, fast end."""
            if total_steps <= 1:
                return 1.0
            # Cumulative fraction to unmask by step t
            progress = step / total_steps
            # Cosine: slow at start, fast at end
            return 0.5 * (1 - math.cos(math.pi * progress))

        for step in range(1, num_steps + 1):
            # Count current masked positions
            num_masked = (X_t == self.mask_index).sum().item()
            if num_masked == 0:
                break

            # Timestep: fraction of positions still masked
            t_val = num_masked / (B * total_positions)
            t = torch.full((B,), t_val, device=device)

            # Get predictions (soft_input=False for discrete)
            logits = self(X_t, Y_mag, t, soft_input=False, perm_indices=perm_indices)  # [B, K, N, Q]

            # Temperature annealing: high early (exploratory), low late (confident)
            tau = cosine_anneal(temp_start, temp_end, step, num_steps)

            # Compute confidence with temperature
            probs = F.softmax(logits / tau, dim=-1)
            confidence, predictions = probs.max(dim=-1)  # [B, K, N]

            # Only consider masked positions
            is_masked = (X_t == self.mask_index)  # [B, K, N]
            confidence = torch.where(is_masked, confidence, torch.zeros_like(confidence))

            # Cosine unmasking schedule: unmask fewer early, more later
            # Target cumulative fraction unmasked by this step
            target_unmasked_frac = unmask_schedule(step, num_steps)
            target_unmasked = int(target_unmasked_frac * total_positions)

            # Flatten for top-k selection
            conf_flat = confidence.view(B, -1)  # [B, K*N]
            pred_flat = predictions.view(B, -1)  # [B, K*N]
            mask_flat = is_masked.view(B, -1)  # [B, K*N]

            # Select top-k most confident masked positions per batch
            for b in range(B):
                masked_indices = mask_flat[b].nonzero(as_tuple=True)[0]
                if len(masked_indices) == 0:
                    continue

                # How many already unmasked?
                already_unmasked = total_positions - len(masked_indices)
                # How many should be unmasked total by this step?
                should_unmask_total = target_unmasked
                # How many to unmask this step?
                num_to_unmask = max(1, should_unmask_total - already_unmasked)
                num_to_unmask = min(num_to_unmask, len(masked_indices))

                # First step: break symmetry by picking random slot's top-1
                if random_slot_first and step == 1 and K > 1:
                    slot_idx = torch.randint(0, K, (1,), device=device).item()
                    slot_mask = torch.zeros(K, N, dtype=torch.bool, device=device)
                    slot_mask[slot_idx] = True
                    slot_mask_flat = slot_mask.view(-1)
                    # Only consider this slot's masked positions
                    slot_masked = mask_flat[b] & slot_mask_flat
                    slot_masked_indices = slot_masked.nonzero(as_tuple=True)[0]
                    if len(slot_masked_indices) > 0:
                        slot_conf = conf_flat[b, slot_masked_indices]
                        _, top_idx = slot_conf.topk(1)  # just top-1
                        unmask_indices = slot_masked_indices[top_idx]
                    else:
                        continue
                else:
                    masked_conf = conf_flat[b, masked_indices]
                    _, top_indices = masked_conf.topk(num_to_unmask)
                    unmask_indices = masked_indices[top_indices]

                # Unmask: set X_t to predicted symbol
                X_t_flat = X_t[b].view(-1)
                X_t_flat[unmask_indices] = pred_flat[b, unmask_indices]
                X_t[b] = X_t_flat.view(K, N)

        # Final pass: any remaining MASK tokens get argmax prediction (low temp)
        if (X_t == self.mask_index).any():
            t_final = torch.zeros(B, device=device)
            final_logits = self(X_t, Y_mag, t_final, soft_input=False, perm_indices=perm_indices)
            final_preds = final_logits.argmax(dim=-1)
            X_t = torch.where(X_t == self.mask_index, final_preds, X_t)

        return X_t

    @torch.no_grad()
    def _sample_soft(self, Y_mag, num_steps=None, use_remasking=False, perm_indices=None):
        """
        Soft belief-based sampling (original method for dit, dimp, dibp).

        perm_indices: [B, K] long, optional — passed through to backbone.
        """
        if num_steps is None:
            num_steps = self.config.model.get('inference_steps', 16)

        B = Y_mag.shape[0]
        K = self.K
        N = self.N
        Q = self.Q
        device = Y_mag.device

        # Initialize with uniform soft beliefs
        X_soft = torch.ones(B, K, N, Q, device=device) / Q

        for step in range(1, num_steps + 1):
            # Timestep based on entropy/confidence
            entropy = -(X_soft * (X_soft + 1e-10).log()).sum(dim=-1).mean()
            max_entropy = math.log(Q)
            t_val = (entropy / max_entropy).clamp(0, 1).item()
            t = torch.full((B,), t_val, device=device)

            # Run backbone with soft input
            logits = self(X_soft, Y_mag, t, soft_input=True, perm_indices=perm_indices)

            # Temperature annealing
            tau = cosine_anneal(1.2, 0.3, step, num_steps)
            probs = F.softmax(logits / tau, dim=-1)

            # Mix old and new beliefs (momentum for stability)
            alpha = step / num_steps  # 0 -> 1 as we progress
            X_soft = (1 - alpha) * X_soft + alpha * probs

        # Final pass at t=0 with current soft beliefs
        t_final = torch.zeros(B, device=device)
        final_logits = self(X_soft, Y_mag, t_final, soft_input=True, perm_indices=perm_indices)
        X_hat = final_logits.argmax(dim=-1)

        return X_hat

    @torch.no_grad()
    def _backbone_inference_discrete(self, X_t, Y_mag, num_steps):
        """
        Run backbone diffusion inference (discrete sampling).

        Used internally by PRISM for each revision round.
        """
        B = X_t.shape[0]
        K = self.K
        N = self.N
        device = X_t.device
        total_positions = K * N

        def unmask_schedule(step, total_steps):
            if total_steps <= 1:
                return 1.0
            progress = step / total_steps
            return 0.5 * (1 - math.cos(math.pi * progress))

        for step in range(1, num_steps + 1):
            num_masked = (X_t == self.mask_index).sum().item()
            if num_masked == 0:
                break

            mask_ratio = num_masked / (B * total_positions)
            t = torch.full((B,), mask_ratio, device=device)

            logits = self(X_t, Y_mag, t, soft_input=False)
            predictions = logits.argmax(dim=-1)

            target_unmasked_frac = unmask_schedule(step, num_steps)
            target_unmasked = int(target_unmasked_frac * total_positions)

            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values
            is_masked = (X_t == self.mask_index)
            confidence[~is_masked] = -1

            for b in range(B):
                mask_b = is_masked[b]
                masked_indices = mask_b.view(-1).nonzero(as_tuple=True)[0]
                if len(masked_indices) == 0:
                    continue

                already_unmasked = total_positions - len(masked_indices)
                num_to_unmask = max(1, target_unmasked - already_unmasked)
                num_to_unmask = min(num_to_unmask, len(masked_indices))

                conf_flat = confidence[b].view(-1)
                _, top_indices = conf_flat.topk(num_to_unmask)

                for idx in top_indices:
                    k = idx // N
                    n = idx % N
                    X_t[b, k, n] = predictions[b, k, n]

        # Final pass for any remaining MASK tokens
        mask = (X_t == self.mask_index)
        if mask.any():
            t = torch.zeros(B, device=device)
            logits = self(X_t, Y_mag, t, soft_input=False)
            predictions = logits.argmax(dim=-1)
            X_t = torch.where(mask, predictions, X_t)

        return X_t

    def restore_model_and_sample(self, Y_mag, num_steps=None, use_remasking=False):
        """Generate samples with EMA weights."""
        if self.ema:
            self.ema.store(self.backbone)
            self.ema.copy_to(self.backbone)

        self.backbone.eval()
        samples = self._sample(Y_mag, num_steps=num_steps, use_remasking=use_remasking)

        if self.ema:
            self.ema.restore(self.backbone)
        self.backbone.train()

        return samples

    # =========================================================================
    # Optimizer Configuration
    # =========================================================================

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        lr = self.config.optim.get('lr', 1e-4)
        weight_decay = self.config.optim.get('weight_decay', 0.01)
        beta1 = self.config.optim.get('beta1', 0.9)
        beta2 = self.config.optim.get('beta2', 0.99)
        eps = self.config.optim.get('eps', 1e-8)

        optimizer = AdamW(
            self.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )

        # Scheduler
        warmup_epochs = self.config.training.get('warmup_epochs', 10)
        num_epochs = self.config.training.get('num_epochs', 100)
        min_lr_ratio = self.config.training.get('min_lr_ratio', 0.01)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs - warmup_epochs),
            eta_min=lr * min_lr_ratio,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "trainer/lr",
            }
        }

    @torch.no_grad()
    def infer(self, Y_mag, num_steps=None, use_remasking=False):
        """Single-sample inference."""
        if num_steps is None:
            base_steps = self.config.model.get('inference_steps', 16)
            if self.config.model.get('adaptive_inference_steps', False):
                num_steps = max(base_steps, self.K * self.N // 10)
            else:
                num_steps = base_steps

        Y_batch = Y_mag.unsqueeze(0).to(self.device)
        X_hat = self.restore_model_and_sample(Y_batch, num_steps=num_steps, use_remasking=use_remasking)
        return X_hat[0]

"""
Token Quality Head for PRISM-style training.

Predicts P(sampled token is correct) for each position in a partially-revealed sequence.

Training:
    1. Create masked input z from GT x
    2. Select subset S of masked positions
    3. Sample symbols at S from backbone logits to form y
    4. Train head to predict correctness: BCE(q_hat[S], (y[S] == x[S]))

Usage at inference:
    1. Get hidden states h from backbone for filled sequence y
    2. Compute token quality: q_hat = head(h)
    3. Aggregate to slot scores: slot_score = q_hat.mean(dim=-1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import lightning as L
from scipy.optimize import linear_sum_assignment


class TokenQualityHead(nn.Module):
    """
    Lightweight MLP head for token-level quality prediction.

    Input: h [B, K, N, D] - hidden states from backbone
    Output: q_hat [B, K, N] - quality scores (probability of correctness)

    Options:
        use_slot_context: If True, concatenate slot-level summary to each token.
            This helps detect slot-level failures (e.g., whole slot drifting).
            Token features become: u[b,k,n] = concat(h[b,k,n], slot_summary[b,k])
    """

    def __init__(
        self,
        D_backbone: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        use_slot_context: bool = False,
    ):
        super().__init__()

        self.D_backbone = D_backbone
        self.hidden_dim = hidden_dim
        self.use_slot_context = use_slot_context

        # Input dimension depends on whether we use slot context
        if use_slot_context:
            input_dim = D_backbone * 2  # token + slot summary
        else:
            input_dim = D_backbone

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize output near zero for stable start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _add_slot_context(self, h: torch.Tensor) -> torch.Tensor:
        """
        Add slot-level context to each token.

        Args:
            h: [B, K, N, D] token hidden states

        Returns:
            u: [B, K, N, 2D] token features with slot context
        """
        # Compute slot summary via mean pooling
        slot_summary = h.mean(dim=2)  # [B, K, D]

        # Broadcast back to token level
        slot_summary_expanded = slot_summary.unsqueeze(2).expand_as(h)  # [B, K, N, D]

        # Concatenate
        u = torch.cat([h, slot_summary_expanded], dim=-1)  # [B, K, N, 2D]

        return u

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict quality for all positions.

        Args:
            h: [B, K, N, D] hidden states

        Returns:
            q_hat: [B, K, N] quality scores in (0, 1)
        """
        if self.use_slot_context:
            u = self._add_slot_context(h)  # [B, K, N, 2D]
        else:
            u = h  # [B, K, N, D]

        logit = self.mlp(u).squeeze(-1)  # [B, K, N]
        return torch.sigmoid(logit)

    def forward_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Return raw logits (before sigmoid) for loss computation."""
        if self.use_slot_context:
            u = self._add_slot_context(h)
        else:
            u = h

        return self.mlp(u).squeeze(-1)  # [B, K, N]


class PRISMSampler:
    """
    Sampler to build PRISM training pairs (x, z, S, y, t).

    Given GT x:
        1. Create masked input z
        2. Select subset S of masked positions
        3. Sample symbols at S to form y
        4. Compute labels t = (y == x) on S
    """

    def __init__(
        self,
        mask_token: int,
        k_per_slot: int = 4,
        n_y: int = 8,
        temperature: float = 1.0,
        selection_mode: str = 'random',  # 'random' or 'topk'
    ):
        self.mask_token = mask_token
        self.k_per_slot = k_per_slot
        self.n_y = n_y
        self.temperature = temperature
        self.selection_mode = selection_mode

    def create_masked_input(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masked input z from GT x.

        Args:
            x: [B, K, N] ground truth symbols
            gamma: [B] masking ratio per sample

        Returns:
            z: [B, K, N] masked input (some positions are mask_token)
            mask: [B, K, N] boolean mask (True = masked position)
        """
        B, K, N = x.shape
        device = x.device

        # Sample mask positions
        gamma_expanded = gamma.view(B, 1, 1).expand(B, K, N)
        mask = torch.rand(B, K, N, device=device) < gamma_expanded

        # Ensure at least 1 masked position per slot
        for b in range(B):
            for k in range(K):
                if not mask[b, k].any():
                    # Force mask one random position
                    idx = torch.randint(0, N, (1,), device=device)
                    mask[b, k, idx] = True

        # Create masked input
        z = x.clone()
        z[mask] = self.mask_token

        return z, mask

    def select_positions(
        self,
        mask: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Select subset S of masked positions for supervision.

        Args:
            mask: [B, K, N] boolean mask (True = masked position)
            confidence: [B, K, N] optional confidence scores for topk selection

        Returns:
            S: [B, K, N] boolean mask of selected positions (subset of mask)
        """
        B, K, N = mask.shape
        device = mask.device

        S = torch.zeros_like(mask)

        if self.selection_mode == 'random':
            # Random selection per slot
            for b in range(B):
                for k in range(K):
                    masked_indices = mask[b, k].nonzero(as_tuple=True)[0]
                    num_masked = len(masked_indices)
                    if num_masked == 0:
                        continue

                    # Select up to k_per_slot positions
                    num_select = min(self.k_per_slot, num_masked)
                    perm = torch.randperm(num_masked, device=device)[:num_select]
                    selected_indices = masked_indices[perm]
                    S[b, k, selected_indices] = True

        elif self.selection_mode == 'topk':
            # Top-k by confidence per slot
            if confidence is None:
                raise ValueError("confidence required for topk selection")

            # Set confidence of non-masked positions to -inf
            conf = confidence.clone()
            conf[~mask] = float('-inf')

            for b in range(B):
                for k in range(K):
                    masked_indices = mask[b, k].nonzero(as_tuple=True)[0]
                    num_masked = len(masked_indices)
                    if num_masked == 0:
                        continue

                    # Select top-k by confidence
                    num_select = min(self.k_per_slot, num_masked)
                    _, topk_local = conf[b, k, masked_indices].topk(num_select)
                    selected_indices = masked_indices[topk_local]
                    S[b, k, selected_indices] = True
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")

        return S

    def sample_fills(
        self,
        z: torch.Tensor,
        S: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample symbols at positions S to form y.

        Args:
            z: [B, K, N] masked input
            S: [B, K, N] boolean mask of positions to fill
            logits: [B, K, N, Q] unmasking logits from backbone

        Returns:
            y: [B, K, N] filled sequence (S positions sampled, rest copied from z)
        """
        B, K, N, Q = logits.shape
        device = logits.device

        # Compute probabilities with temperature
        probs = F.softmax(logits / self.temperature, dim=-1)  # [B, K, N, Q]

        # Sample from categorical
        probs_flat = probs.view(-1, Q)  # [B*K*N, Q]
        samples_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # [B*K*N]
        samples = samples_flat.view(B, K, N)  # [B, K, N]

        # Fill only at S positions
        y = z.clone()
        y[S] = samples[S]

        return y


class QualityHeadTrainer(L.LightningModule):
    """
    Lightning module for training TokenQualityHead.

    Freezes backbone, trains head to predict token correctness.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        backbone: nn.Module,
        H: torch.Tensor,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'H'])

        self.config = config

        # Frozen backbone
        self.backbone = backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # H matrix for backbone
        self.register_buffer('H', H)

        # Get dimensions from backbone
        self.Q = backbone.Q
        self.N = backbone.N
        self.K = backbone.K
        self.D = backbone.D
        self.mask_token = self.Q  # mask_id = Q

        # Quality head (trainable)
        head_config = config.get('quality_head', {})
        self.quality_head = TokenQualityHead(
            D_backbone=self.D,
            hidden_dim=head_config.get('hidden_dim', 256),
            dropout=head_config.get('dropout', 0.0),
            use_slot_context=head_config.get('use_slot_context', False),
        )

        # Sampler
        sampler_config = config.get('sampler', {})
        self.sampler = PRISMSampler(
            mask_token=self.mask_token,
            k_per_slot=sampler_config.get('k_per_slot', 4),
            n_y=sampler_config.get('n_y', 8),
            temperature=sampler_config.get('temperature', 1.0),
            selection_mode=sampler_config.get('selection_mode', 'random'),
        )

        # Masking config (match backbone training)
        mask_config = config.get('masking', {})
        self.gamma_min = mask_config.get('gamma_min', 0.1)
        self.gamma_max = mask_config.get('gamma_max', 1.0)

        # Training config
        self.lr = config.get('lr', 1e-3)
        self.weight_decay = config.get('weight_decay', 0.01)

    def _sample_gamma(self, B: int) -> torch.Tensor:
        """Sample masking ratio matching backbone training distribution."""
        # Uniform over [gamma_min, gamma_max]
        gamma = torch.rand(B, device=self.device) * (self.gamma_max - self.gamma_min) + self.gamma_min
        return gamma

    @torch.no_grad()
    def _get_backbone_outputs(
        self,
        z: torch.Tensor,
        Y: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get logits and hidden states from frozen backbone.

        Args:
            z: [B, K, N] input tokens (possibly masked)
            Y: [B, N, Q] channel observation
            t: [B] timestep

        Returns:
            logits: [B, K, N, Q] unmasking logits
            h: [B, K, N, D] hidden states
        """
        B = z.shape[0]
        syn = torch.zeros(B, self.K, self.backbone.M, device=self.device, dtype=torch.long)

        logits, hidden_flat, _ = self.backbone.get_hidden_states(
            z, Y, syn, t, self.H, soft_input=False
        )

        # Reshape hidden states: [B, K*N, D] -> [B, K, N, D]
        h = hidden_flat.view(B, self.K, self.N, self.D)

        return logits, h

    def _hungarian_match(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reorder ground truth x to match predictions y using Hungarian algorithm.

        Only compares non-masked positions (y may have MASK tokens).

        Args:
            y: [B, K, N] predicted/sampled sequences (may contain mask_token)
            x: [B, K, N] ground truth sequences

        Returns:
            matched_x: [B, K, N] ground truth reordered to align with y
        """
        B, K, N = y.shape
        device = y.device

        matched_x = torch.zeros_like(x)

        for b in range(B):
            # Compute cost matrix (Hamming distance on revealed positions only)
            cost = torch.zeros(K, K, device=device)
            for i in range(K):
                # Mask for revealed positions in slot i
                revealed = (y[b, i] != self.mask_token)  # [N]
                if revealed.sum() == 0:
                    # All masked - use uniform cost
                    cost[i, :] = 1.0
                    continue

                for j in range(K):
                    # Only count mismatches on revealed positions
                    mismatches = (y[b, i, revealed] != x[b, j, revealed]).float().sum()
                    cost[i, j] = mismatches

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

            # Reorder GT to match predictions
            for i, j in zip(row_ind, col_ind):
                matched_x[b, i] = x[b, j]

        return matched_x

    def training_step(self, batch, batch_idx):
        """
        Training step with PRISM sampling.

        For each batch:
            1. Create masked input z
            2. Select positions S
            3. For j=1..n_y: sample y, get h, compute loss
        """
        Y, x = batch  # Y: [B, N, Q], x: [B, K, N]
        B = x.shape[0]

        # Sample masking ratio
        gamma = self._sample_gamma(B)

        # Create masked input
        z, mask = self.sampler.create_masked_input(x, gamma)

        # Get backbone logits for sampling (use z)
        t = gamma  # timestep = mask ratio
        logits_z, _ = self._get_backbone_outputs(z, Y, t)

        # Compute confidence for potential topk selection
        probs_z = F.softmax(logits_z, dim=-1)
        confidence = probs_z.max(dim=-1).values  # [B, K, N]

        # Select positions S
        S = self.sampler.select_positions(mask, confidence)

        # Accumulate loss over n_y samples
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for _ in range(self.sampler.n_y):
            # Sample fills at S
            y = self.sampler.sample_fills(z, S, logits_z)

            # Get hidden states for y (filled sequence)
            # Use t=0 since y is (partially) revealed
            t_filled = torch.zeros(B, device=self.device)
            _, h = self._get_backbone_outputs(y, Y, t_filled)

            # Compute quality predictions
            q_logits = self.quality_head.forward_logits(h)  # [B, K, N]

            # Hungarian matching: align GT x to match y's slot ordering
            with torch.no_grad():
                matched_x = self._hungarian_match(y, x)

            # Target: 1 if y == matched_x at position, else 0
            target = (y == matched_x).float()  # [B, K, N]

            # BCE loss only on S positions
            loss = F.binary_cross_entropy_with_logits(
                q_logits[S],
                target[S],
                reduction='mean'
            )
            total_loss += loss

            # Track accuracy
            with torch.no_grad():
                preds = (torch.sigmoid(q_logits[S]) > 0.5).float()
                total_correct += (preds == target[S]).sum().item()
                total_count += S.sum().item()

        # Average loss
        avg_loss = total_loss / self.sampler.n_y
        accuracy = total_correct / max(total_count, 1)

        # Logging
        self.log('train/loss', avg_loss, prog_bar=True)
        self.log('train/accuracy', accuracy, prog_bar=True)
        self.log('train/num_positions', S.sum().float().mean())

        return avg_loss

    def validation_step(self, batch, batch_idx):
        """Validation step (same as training but no gradient)."""
        Y, x = batch
        B = x.shape[0]

        gamma = self._sample_gamma(B)
        z, mask = self.sampler.create_masked_input(x, gamma)

        t = gamma
        logits_z, _ = self._get_backbone_outputs(z, Y, t)
        probs_z = F.softmax(logits_z, dim=-1)
        confidence = probs_z.max(dim=-1).values

        S = self.sampler.select_positions(mask, confidence)

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for _ in range(self.sampler.n_y):
            y = self.sampler.sample_fills(z, S, logits_z)
            t_filled = torch.zeros(B, device=self.device)
            _, h = self._get_backbone_outputs(y, Y, t_filled)

            q_logits = self.quality_head.forward_logits(h)

            # Hungarian matching: align GT x to match y's slot ordering
            matched_x = self._hungarian_match(y, x)
            target = (y == matched_x).float()

            loss = F.binary_cross_entropy_with_logits(
                q_logits[S],
                target[S],
                reduction='mean'
            )
            total_loss += loss

            preds = (torch.sigmoid(q_logits[S]) > 0.5).float()
            total_correct += (preds == target[S]).sum().item()
            total_count += S.sum().item()

        avg_loss = total_loss / self.sampler.n_y
        accuracy = total_correct / max(total_count, 1)

        self.log('val/loss', avg_loss, prog_bar=True, sync_dist=True)
        self.log('val/accuracy', accuracy, prog_bar=True, sync_dist=True)

        return avg_loss

    def configure_optimizers(self):
        """Configure optimizer for quality head only."""
        optimizer = torch.optim.AdamW(
            self.quality_head.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.lr * 0.01,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

    @torch.no_grad()
    def predict_quality(self, y: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Predict token quality for a filled sequence.

        Args:
            y: [B, K, N] filled sequence (no masks)
            Y: [B, N, Q] channel observation

        Returns:
            q_hat: [B, K, N] quality scores
        """
        B = y.shape[0]
        t = torch.zeros(B, device=self.device)
        _, h = self._get_backbone_outputs(y, Y, t)
        return self.quality_head(h)

    @torch.no_grad()
    def predict_slot_scores(
        self,
        y: torch.Tensor,
        Y: torch.Tensor,
        aggregation: str = 'mean',
    ) -> torch.Tensor:
        """
        Predict slot-level scores by aggregating token qualities.

        Args:
            y: [B, K, N] filled sequence
            Y: [B, N, Q] channel observation
            aggregation: 'mean' or 'logit_mean'

        Returns:
            slot_scores: [B, K] slot-level scores
        """
        q_hat = self.predict_quality(y, Y)  # [B, K, N]

        if aggregation == 'mean':
            return q_hat.mean(dim=-1)  # [B, K]
        elif aggregation == 'logit_mean':
            # More "AND-like" aggregation
            eps = 1e-7
            logits = torch.log(q_hat + eps) - torch.log(1 - q_hat + eps)
            return torch.sigmoid(logits.mean(dim=-1))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

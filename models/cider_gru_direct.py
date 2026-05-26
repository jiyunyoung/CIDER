"""
Direct prediction without diffusion framework.

Uses the same cider_gru backbone but as a one-shot predictor:
- No diffusion timesteps or masking schedule
- Single forward pass from Y to predictions
- Hungarian matching loss (like baselines)

Input:  Y [batch, N, Q] - soft scores from inner decoder
Output: logits [batch, K, N, Q] - K codewords over GF(Q)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from scipy.optimize import linear_sum_assignment

from models.cider_gru import DiMP


class DiMPRev2OneShot(L.LightningModule):
    """
    cider-gru as a one-shot predictor (no diffusion).

    Architecture:
        Y [N, Q] -> DiMP backbone (single pass) -> [K, N, Q]

    Uses Hungarian matching loss for permutation-invariant training.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # Data config
        self.N = config.data.N
        self.Q = config.data.Q
        self.K = config.data.get('K', config.data.get('K_max', 2))
        self.M = config.data.M

        # Model config
        model_cfg = config.model
        self.D_model = model_cfg.get('D_model', 128)
        self.num_layers = model_cfg.get('num_layers', 4)
        self.heads = model_cfg.get('heads', 4)
        self.mlp_ratio = model_cfg.get('mlp_ratio', 4)
        self.dropout = model_cfg.get('dropout', 0.1)
        self.tau_min = model_cfg.get('tau_min', 0.2)
        self.slot_init_scale = model_cfg.get('slot_init_scale', 0.5)

        # Training config
        self.lr = config.training.learning_rate
        self.weight_decay = config.training.optimizer.weight_decay

        # Build DiMP backbone
        self.backbone = DiMP(
            Q=self.Q,
            N=self.N,
            K=self.K,
            M=self.M,
            D_model=self.D_model,
            num_layers=self.num_layers,
            heads=self.heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            tau_min=self.tau_min,
            slot_init_scale=self.slot_init_scale,
        )

        # H matrix placeholder
        self.register_buffer('H_matrix', None)

    def set_H_matrix(self, H):
        """Set the parity check matrix H."""
        if H is not None:
            if isinstance(H, torch.Tensor):
                self.H_matrix = H
            else:
                self.H_matrix = torch.tensor(H)

    def forward(self, Y):
        """
        Forward pass.

        Args:
            Y: [batch, N, Q] soft scores from inner decoder

        Returns:
            logits: [batch, K, N, Q] predicted codeword logits
        """
        B = Y.shape[0]
        device = Y.device

        # Create uniform initial belief for X_t (soft input)
        # All symbols equally likely at start
        X_t = torch.ones(B, self.K, self.N, self.Q, device=device) / self.Q

        # Fixed timestep t=1.0 (beginning of "denoising" - most uncertain)
        t = torch.ones(B, device=device)

        # Dummy syndrome (not used in one-shot mode)
        syn = torch.zeros(B, self.M, device=device)

        # Forward through backbone
        logits = self.backbone(
            X_t=X_t,
            Y_mag=Y,
            syn=syn,
            t=t,
            H=self.H_matrix,
            soft_input=True,
        )

        return logits

    def hungarian_loss(self, pred_logits, gt_codewords):
        """Permutation-invariant loss using Hungarian algorithm."""
        batch_size = pred_logits.shape[0]
        K = self.K
        total_loss = 0.0

        for b in range(batch_size):
            cost_matrix = torch.zeros(K, K, device=pred_logits.device)
            for i in range(K):
                for j in range(K):
                    cost_matrix[i, j] = F.cross_entropy(
                        pred_logits[b, i], gt_codewords[b, j], reduction='sum'
                    )

            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            for i, j in zip(row_ind, col_ind):
                total_loss += F.cross_entropy(
                    pred_logits[b, i], gt_codewords[b, j], reduction='mean'
                )

        return total_loss / batch_size

    def compute_accuracy(self, pred_logits, gt_codewords):
        """Compute symbol and codeword accuracy with Hungarian matching."""
        batch_size = pred_logits.shape[0]
        K, N = self.K, self.N
        pred_codewords = pred_logits.argmax(dim=-1)

        total_symbols = correct_symbols = 0
        total_codewords = correct_codewords = 0

        for b in range(batch_size):
            cost_matrix = torch.zeros(K, K, device=pred_logits.device)
            for i in range(K):
                for j in range(K):
                    cost_matrix[i, j] = (pred_codewords[b, i] != gt_codewords[b, j]).sum().float()

            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            for i, j in zip(row_ind, col_ind):
                matches = (pred_codewords[b, i] == gt_codewords[b, j])
                correct_symbols += matches.sum().item()
                total_symbols += N
                if matches.all():
                    correct_codewords += 1
                total_codewords += 1

        return {
            'symbol_acc': correct_symbols / total_symbols if total_symbols > 0 else 0.0,
            'codeword_acc': correct_codewords / total_codewords if total_codewords > 0 else 0.0,
        }

    def training_step(self, batch, batch_idx):
        Y, gt_codewords = batch
        logits = self.forward(Y)
        loss = self.hungarian_loss(logits, gt_codewords)

        with torch.no_grad():
            metrics = self.compute_accuracy(logits, gt_codewords)

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/symbol_acc', metrics['symbol_acc'], prog_bar=True)
        self.log('train/codeword_acc', metrics['codeword_acc'])

        return loss

    def validation_step(self, batch, batch_idx):
        Y, gt_codewords = batch
        logits = self.forward(Y)
        loss = self.hungarian_loss(logits, gt_codewords)
        metrics = self.compute_accuracy(logits, gt_codewords)

        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/symbol_acc', metrics['symbol_acc'], prog_bar=True, sync_dist=True)
        self.log('val/codeword_acc', metrics['codeword_acc'], sync_dist=True)
        self.log('val/accuracy', metrics['symbol_acc'], sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        Y, gt_codewords = batch
        logits = self.forward(Y)
        loss = self.hungarian_loss(logits, gt_codewords)
        metrics = self.compute_accuracy(logits, gt_codewords)

        self.log('test/loss', loss, sync_dist=True)
        self.log('test/symbol_acc', metrics['symbol_acc'], sync_dist=True)
        self.log('test/codeword_acc', metrics['codeword_acc'], sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if self.trainer else 100, eta_min=1e-6
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

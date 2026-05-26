"""
CNN Baseline for Codeword Demixing.

CNN that directly predicts K codewords from soft scores Y.
No diffusion - one-shot prediction.

Input:  Y [batch, N, Q] - soft scores from inner decoder
Output: logits [batch, K, N, Q] - K codewords over GF(Q)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from scipy.optimize import linear_sum_assignment
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block with 1D convolutions."""

    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        """x: [batch, channels, seq_len]"""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.activation(out)

        return out


class CNNDemixer(L.LightningModule):
    """
    CNN baseline for codeword demixing.

    Architecture:
        Y [N, Q] -> Conv1d blocks -> Output heads -> [K, N, Q]

    Each slot (1..K) has a separate output head.
    Uses Hungarian matching loss for permutation-invariant training.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # Extract config
        self.N = config.data.N
        self.Q = config.data.Q
        self.K = config.data.get('K', config.data.get('K_max', 2))

        # Model config
        model_cfg = config.model
        self.channels = model_cfg.get('channels', 256)
        self.num_blocks = model_cfg.get('num_blocks', 4)
        self.kernel_size = model_cfg.get('kernel_size', 3)
        self.dropout = model_cfg.get('dropout', 0.1)

        # Training config
        self.lr = config.training.learning_rate
        self.weight_decay = config.training.optimizer.weight_decay

        # Input projection: [N, Q] -> [N, channels]
        self.input_proj = nn.Sequential(
            nn.Linear(self.Q, self.channels),
            nn.LayerNorm(self.channels),
            nn.GELU(),
        )

        # Residual CNN blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(self.channels, self.kernel_size, self.dropout)
            for _ in range(self.num_blocks)
        ])

        # Slot-specific output heads
        # Each head produces [N, Q] logits for one slot
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.channels, self.channels),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.channels, self.Q),
            )
            for _ in range(self.K)
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, Y):
        """
        Forward pass.

        Args:
            Y: [batch, N, Q] soft scores from inner decoder

        Returns:
            logits: [batch, K, N, Q] predicted codeword logits
        """
        batch_size = Y.shape[0]

        # Input projection: [batch, N, Q] -> [batch, N, channels]
        x = self.input_proj(Y)

        # Transpose for Conv1d: [batch, N, channels] -> [batch, channels, N]
        x = x.transpose(1, 2)

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Transpose back: [batch, channels, N] -> [batch, N, channels]
        x = x.transpose(1, 2)

        # Apply slot-specific heads
        slot_outputs = []
        for head in self.output_heads:
            slot_logits = head(x)  # [batch, N, Q]
            slot_outputs.append(slot_logits)

        # Stack: [batch, K, N, Q]
        logits = torch.stack(slot_outputs, dim=1)

        return logits

    def hungarian_loss(self, pred_logits, gt_codewords):
        """
        Permutation-invariant loss using Hungarian algorithm.

        Args:
            pred_logits: [batch, K, N, Q] predicted logits
            gt_codewords: [batch, K, N] ground truth codewords

        Returns:
            loss: scalar loss value
        """
        batch_size = pred_logits.shape[0]
        K = self.K

        total_loss = 0.0

        for b in range(batch_size):
            # Compute cost matrix: cost[i,j] = CE(pred[i], gt[j])
            cost_matrix = torch.zeros(K, K, device=pred_logits.device)

            for i in range(K):
                for j in range(K):
                    # Cross-entropy loss for slot i predicting codeword j
                    ce = F.cross_entropy(
                        pred_logits[b, i],  # [N, Q]
                        gt_codewords[b, j],  # [N]
                        reduction='sum'
                    )
                    cost_matrix[i, j] = ce

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            # Compute loss with optimal assignment
            for i, j in zip(row_ind, col_ind):
                total_loss += F.cross_entropy(
                    pred_logits[b, i],
                    gt_codewords[b, j],
                    reduction='mean'
                )

        return total_loss / batch_size

    def compute_accuracy(self, pred_logits, gt_codewords):
        """
        Compute symbol and codeword accuracy with Hungarian matching.

        Args:
            pred_logits: [batch, K, N, Q]
            gt_codewords: [batch, K, N]

        Returns:
            dict with symbol_acc, codeword_acc
        """
        batch_size = pred_logits.shape[0]
        K = self.K
        N = self.N

        # Get predictions
        pred_codewords = pred_logits.argmax(dim=-1)  # [batch, K, N]

        total_symbols = 0
        correct_symbols = 0
        total_codewords = 0
        correct_codewords = 0

        for b in range(batch_size):
            # Compute cost matrix for matching
            cost_matrix = torch.zeros(K, K, device=pred_logits.device)

            for i in range(K):
                for j in range(K):
                    # Number of mismatches
                    cost_matrix[i, j] = (pred_codewords[b, i] != gt_codewords[b, j]).sum().float()

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            # Compute accuracy with optimal assignment
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

        # Forward
        logits = self.forward(Y)

        # Loss
        loss = self.hungarian_loss(logits, gt_codewords)

        # Metrics
        with torch.no_grad():
            metrics = self.compute_accuracy(logits, gt_codewords)

        # Log
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/symbol_acc', metrics['symbol_acc'], prog_bar=True)
        self.log('train/codeword_acc', metrics['codeword_acc'])

        return loss

    def validation_step(self, batch, batch_idx):
        Y, gt_codewords = batch

        # Forward
        logits = self.forward(Y)

        # Loss
        loss = self.hungarian_loss(logits, gt_codewords)

        # Metrics
        metrics = self.compute_accuracy(logits, gt_codewords)

        # Log
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/symbol_acc', metrics['symbol_acc'], prog_bar=True, sync_dist=True)
        self.log('val/codeword_acc', metrics['codeword_acc'], sync_dist=True)
        self.log('val/accuracy', metrics['symbol_acc'], sync_dist=True)  # alias for checkpoint

        return loss

    def test_step(self, batch, batch_idx):
        Y, gt_codewords = batch

        # Forward
        logits = self.forward(Y)

        # Loss
        loss = self.hungarian_loss(logits, gt_codewords)

        # Metrics
        metrics = self.compute_accuracy(logits, gt_codewords)

        # Log
        self.log('test/loss', loss, sync_dist=True)
        self.log('test/symbol_acc', metrics['symbol_acc'], sync_dist=True)
        self.log('test/codeword_acc', metrics['codeword_acc'], sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

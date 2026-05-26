"""
Transformer Baselines for Codeword Demixing.

Two variants:
1. TransformerDemixer: Simple encoder + K output heads
2. TransformerDemixerH: Cross-attention with parity matrix H

Both output codewords [K, N, Q] directly (scalable to high B).

Input:  Y [batch, N, Q] - soft scores from inner decoder
Output: logits [batch, K, N, Q] - K codewords over GF(Q)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from scipy.optimize import linear_sum_assignment
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# Variant 1: Simple Transformer Encoder
# ============================================================

class TransformerDemixer(L.LightningModule):
    """
    Simple Transformer for codeword demixing.

    Architecture:
        Y [N, Q] -> Linear -> Transformer Encoder -> K Output Heads -> [K, N, Q]

    No codebook, no H matrix - pure self-attention.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # Data config
        self.N = config.data.N
        self.Q = config.data.Q
        self.K = config.data.get('K', config.data.get('K_max', 2))

        # Model config
        model_cfg = config.model
        self.d_model = model_cfg.get('d_model', 256)
        self.num_layers = model_cfg.get('num_layers', 4)
        self.num_heads = model_cfg.get('num_heads', 8)
        self.d_ff = model_cfg.get('d_ff', 1024)
        self.dropout = model_cfg.get('dropout', 0.1)

        # Training config
        self.lr = config.training.learning_rate
        self.weight_decay = config.training.optimizer.weight_decay

        # Input projection
        self.input_proj = nn.Linear(self.Q, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=self.N, dropout=self.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # K output heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.Q),
            )
            for _ in range(self.K)
        ])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, Y):
        """
        Args:
            Y: [batch, N, Q] soft scores
        Returns:
            logits: [batch, K, N, Q]
        """
        x = self.input_proj(Y)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        slot_outputs = [head(x) for head in self.output_heads]
        logits = torch.stack(slot_outputs, dim=1)

        return logits

    def hungarian_loss(self, pred_logits, gt_codewords):
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
        self.log('val/accuracy', metrics['symbol_acc'], sync_dist=True)  # alias for checkpoint
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


# ============================================================
# Variant 2: Transformer with H-matrix Cross-Attention
# ============================================================

class TransformerDemixerH(L.LightningModule):
    """
    Transformer with parity matrix H cross-attention.

    Architecture:
        Y [N, Q] -> Encoder -> Cross-Attention(H) -> K Output Heads -> [K, N, Q]

    Cross-attention uses H matrix rows as keys/values, allowing the model
    to leverage code structure without needing the full codebook.

    H matrix: [M_parity, N] where M_parity << 2^B
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # Data config
        self.N = config.data.N
        self.Q = config.data.Q
        self.K = config.data.get('K', config.data.get('K_max', 2))
        self.M_parity = config.data.M  # Number of parity checks

        # Model config
        model_cfg = config.model
        self.d_model = model_cfg.get('d_model', 256)
        self.num_layers = model_cfg.get('num_layers', 4)
        self.num_heads = model_cfg.get('num_heads', 8)
        self.d_ff = model_cfg.get('d_ff', 1024)
        self.dropout = model_cfg.get('dropout', 0.1)

        # Training config
        self.lr = config.training.learning_rate
        self.weight_decay = config.training.optimizer.weight_decay

        # Input projection
        self.input_proj = nn.Linear(self.Q, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=self.N, dropout=self.dropout)

        # H matrix embedding: each row of H -> embedding
        # H[m, n] indicates connection between parity m and position n
        self.h_row_embedding = nn.Embedding(self.M_parity, self.d_model)
        self.h_pos_embedding = nn.Linear(self.N, self.d_model)

        # Self-attention encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dim_feedforward=self.d_ff,
                dropout=self.dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(self.num_layers // 2)
        ])

        # Cross-attention with H
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.d_model)
        self.cross_ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model),
            nn.Dropout(self.dropout),
        )
        self.cross_ff_norm = nn.LayerNorm(self.d_model)

        # Post cross-attention encoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dim_feedforward=self.d_ff,
                dropout=self.dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(self.num_layers // 2)
        ])

        # K output heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.Q),
            )
            for _ in range(self.K)
        ])

        # H matrix placeholder (set via set_H_matrix)
        self.register_buffer('H_matrix', None)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_H_matrix(self, H):
        """Set the parity check matrix H [M_parity, N]."""
        if isinstance(H, torch.Tensor):
            self.H_matrix = H.float()
        else:
            self.H_matrix = torch.tensor(H, dtype=torch.float32)

    def _get_h_embeddings(self, batch_size, device):
        """
        Create H matrix embeddings for cross-attention.

        Returns:
            h_emb: [batch, M_parity, d_model]
        """
        # Row indices embedding
        row_idx = torch.arange(self.M_parity, device=device)
        row_emb = self.h_row_embedding(row_idx)  # [M_parity, d_model]

        # H matrix position features
        H = self.H_matrix.to(device)  # [M_parity, N]
        pos_emb = self.h_pos_embedding(H)  # [M_parity, d_model]

        # Combine
        h_emb = row_emb + pos_emb  # [M_parity, d_model]

        # Expand for batch
        h_emb = h_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, M_parity, d_model]

        return h_emb

    def forward(self, Y):
        """
        Args:
            Y: [batch, N, Q] soft scores
        Returns:
            logits: [batch, K, N, Q]
        """
        batch_size = Y.shape[0]
        device = Y.device

        # Input projection
        x = self.input_proj(Y)  # [batch, N, d_model]
        x = self.pos_encoding(x)

        # Self-attention encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Cross-attention with H matrix
        h_emb = self._get_h_embeddings(batch_size, device)  # [batch, M_parity, d_model]

        # Query: x [batch, N, d_model], Key/Value: h_emb [batch, M_parity, d_model]
        attn_out, _ = self.cross_attn(query=x, key=h_emb, value=h_emb)
        x = self.cross_norm(x + attn_out)
        x = self.cross_ff_norm(x + self.cross_ff(x))

        # Post cross-attention layers
        for layer in self.decoder_layers:
            x = layer(x)

        # K output heads
        slot_outputs = [head(x) for head in self.output_heads]
        logits = torch.stack(slot_outputs, dim=1)

        return logits

    def hungarian_loss(self, pred_logits, gt_codewords):
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
        self.log('val/accuracy', metrics['symbol_acc'], sync_dist=True)  # alias for checkpoint
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

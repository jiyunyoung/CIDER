"""
GNN Baseline for Codeword Demixing.

Graph Neural Network operating on the Tanner graph structure.
Message passing between variable nodes (VN) and check nodes (CN).
No diffusion - one-shot prediction.

Input:  Y [batch, N, Q] - soft scores from inner decoder
        H [M, N] - parity check matrix (defines Tanner graph)
Output: logits [batch, K, N, Q] - K codewords over GF(Q)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from scipy.optimize import linear_sum_assignment
import numpy as np


class GNNLayer(nn.Module):
    """
    One layer of GNN message passing on Tanner graph.

    VN -> CN: Variable nodes send messages to check nodes
    CN -> VN: Check nodes send messages back to variable nodes
    """

    def __init__(self, D_model: int, dropout: float = 0.1):
        super().__init__()
        self.D = D_model

        # VN -> CN message (pre-norm)
        self.vn_to_cn = nn.Sequential(
            nn.LayerNorm(D_model),
            nn.Linear(D_model, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
        )

        # CN -> VN message (pre-norm)
        self.cn_to_vn = nn.Sequential(
            nn.LayerNorm(D_model),
            nn.Linear(D_model, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
        )

        # CN self-update (pre-norm)
        self.cn_update = nn.Sequential(
            nn.LayerNorm(2 * D_model),
            nn.Linear(2 * D_model, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
        )

        # VN self-update (pre-norm)
        self.vn_update = nn.Sequential(
            nn.LayerNorm(2 * D_model),
            nn.Linear(2 * D_model, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
        )

    def forward(
        self,
        vn: torch.Tensor,      # [B, N, D]
        cn: torch.Tensor,      # [B, M, D]
        H: torch.Tensor,       # [M, N]
    ) -> tuple:
        """
        One round of message passing.

        Returns:
            vn_new: [B, N, D]
            cn_new: [B, M, D]
        """
        B, N, D = vn.shape
        M = cn.shape[1]

        # Create adjacency mask from H (ensure same device)
        H = H.to(vn.device)
        adj = (H != 0).float()  # [M, N]

        # VN -> CN: aggregate variable messages to checks
        vn_msg = self.vn_to_cn(vn)  # [B, N, D]
        # Sum over connected variables for each check
        # cn_agg[b, m] = sum_{n: H[m,n]!=0} vn_msg[b, n]
        cn_agg = torch.einsum('mn,bnd->bmd', adj, vn_msg)  # [B, M, D]
        # Normalize by degree
        degree_cn = adj.sum(dim=1, keepdim=True).clamp(min=1)  # [M, 1]
        cn_agg = cn_agg / degree_cn.unsqueeze(0)

        # Update CN
        cn_new = cn + self.cn_update(torch.cat([cn, cn_agg], dim=-1))

        # CN -> VN: aggregate check messages to variables
        cn_msg = self.cn_to_vn(cn_new)  # [B, M, D]
        # Sum over connected checks for each variable
        # vn_agg[b, n] = sum_{m: H[m,n]!=0} cn_msg[b, m]
        vn_agg = torch.einsum('mn,bmd->bnd', adj, cn_msg)  # [B, N, D]
        # Normalize by degree
        degree_vn = adj.sum(dim=0, keepdim=True).clamp(min=1)  # [1, N]
        vn_agg = vn_agg / degree_vn.unsqueeze(-1)  # [1, N, 1] for broadcasting with [B, N, D]

        # Update VN
        vn_new = vn + self.vn_update(torch.cat([vn, vn_agg], dim=-1))

        return vn_new, cn_new


class GNNDemixer(L.LightningModule):
    """
    GNN baseline for codeword demixing.

    Architecture:
        Y [N, Q] -> Embed -> K copies -> GNN layers on Tanner graph -> Output heads -> [K, N, Q]

    Uses the Tanner graph structure defined by parity check matrix H.
    Each slot has separate VN embeddings but shares GNN parameters.
    Uses Hungarian matching loss for permutation-invariant training.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # Extract config
        self.N = config.data.N
        self.Q = config.data.Q
        self.K = config.data.get('K', config.data.get('K_max', 2))
        self.M = config.data.M

        # Model config
        model_cfg = config.model
        self.D = model_cfg.get('D_model', 128)
        self.num_layers = model_cfg.get('num_layers', 4)
        self.dropout = model_cfg.get('dropout', 0.1)

        # Training config
        self.lr = config.training.learning_rate
        self.weight_decay = config.training.optimizer.weight_decay

        # Evidence encoder: Y -> VN initial embedding
        self.evidence_encoder = nn.Sequential(
            nn.LayerNorm(self.Q),
            nn.Linear(self.Q, self.D),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.D, self.D),
        )

        # Slot embeddings for symmetry breaking
        self.slot_embed = nn.Parameter(torch.randn(self.K, self.D) * 0.02)

        # Check node initialization
        self.cn_init = nn.Parameter(torch.randn(self.M, self.D) * 0.02)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(self.D, self.dropout)
            for _ in range(self.num_layers)
        ])

        # Output heads (K independent)
        self.out_norm = nn.LayerNorm(self.D)
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.D, self.D),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.D, self.Q),
            )
            for _ in range(self.K)
        ])

        # Store H matrix
        self.register_buffer('H_matrix', None)

    def set_H_matrix(self, H: torch.Tensor):
        """Set the parity check matrix."""
        self.H_matrix = H

    def forward(self, Y: torch.Tensor, H: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            Y: [B, N, Q] soft scores
            H: [M, N] parity check matrix (optional if set via set_H_matrix)

        Returns:
            logits: [B, K, N, Q]
        """
        if H is None:
            H = self.H_matrix
        assert H is not None, "H matrix not provided"

        B, N, Q = Y.shape

        # Encode evidence
        y_emb = self.evidence_encoder(Y)  # [B, N, D]

        # Create K copies with slot embeddings
        vn = y_emb.unsqueeze(1) + self.slot_embed.view(1, self.K, 1, self.D)  # [B, K, N, D]

        # Initialize check nodes (same for all slots)
        cn = self.cn_init.unsqueeze(0).unsqueeze(0).expand(B, self.K, -1, -1).contiguous()  # [B, K, M, D]

        # Flatten K into batch for GNN processing
        vn_flat = vn.reshape(B * self.K, N, self.D).contiguous()
        cn_flat = cn.reshape(B * self.K, self.M, self.D).contiguous()

        # GNN layers
        for layer in self.gnn_layers:
            vn_flat, cn_flat = layer(vn_flat, cn_flat, H)

        # Reshape back
        vn = vn_flat.reshape(B, self.K, N, self.D)

        # Output projection
        vn = self.out_norm(vn)

        # Apply K independent output heads
        slot_logits = []
        for k in range(self.K):
            slot_logits.append(self.output_heads[k](vn[:, k]))  # [B, N, Q]
        logits = torch.stack(slot_logits, dim=1)  # [B, K, N, Q]

        return logits

    def hungarian_loss(self, pred_logits, gt_codewords):
        """
        Compute cross-entropy loss with Hungarian matching.

        Args:
            pred_logits: [B, K, N, Q] predicted logits
            gt_codewords: [B, K, N] ground truth codewords (indices)

        Returns:
            loss: scalar
        """
        B, K, N, Q = pred_logits.shape
        device = pred_logits.device

        total_loss = 0.0

        for b in range(B):
            # Cost matrix: [K_pred, K_gt]
            cost = torch.zeros(K, K, device=device)

            for i in range(K):
                for j in range(K):
                    # Cross-entropy loss for matching pred[i] to gt[j]
                    ce = F.cross_entropy(
                        pred_logits[b, i],  # [N, Q]
                        gt_codewords[b, j],  # [N]
                        reduction='sum'
                    )
                    cost[i, j] = ce

            # Hungarian matching
            cost_np = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            # Accumulate matched loss
            for i, j in zip(row_ind, col_ind):
                total_loss += F.cross_entropy(
                    pred_logits[b, i],
                    gt_codewords[b, j],
                    reduction='mean'
                )

        return total_loss / B

    def training_step(self, batch, batch_idx):
        Y, X = batch

        # Forward pass (H_matrix should be set during setup)
        logits = self(Y)

        # Loss
        loss = self.hungarian_loss(logits, X)

        # Accuracy
        preds = logits.argmax(dim=-1)  # [B, K, N]

        # Match predictions to ground truth
        B, K, N = X.shape
        correct = 0
        total = 0
        for b in range(B):
            cost = torch.zeros(K, K, device=X.device)
            for i in range(K):
                for j in range(K):
                    cost[i, j] = (preds[b, i] != X[b, j]).sum().float()
            cost_np = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            for i, j in zip(row_ind, col_ind):
                correct += (preds[b, i] == X[b, j]).sum().item()
            total += K * N

        acc = correct / total

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/accuracy', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        Y, X = batch

        logits = self(Y)
        loss = self.hungarian_loss(logits, X)

        metrics = self.compute_accuracy(logits, X)

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/accuracy', metrics['symbol_acc'], prog_bar=True)
        self.log('val/symbol_acc', metrics['symbol_acc'])
        self.log('val/codeword_acc', metrics['codeword_acc'])

        return loss

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

        pred_codewords = pred_logits.argmax(dim=-1)  # [batch, K, N]

        total_symbols = 0
        correct_symbols = 0
        total_codewords = 0
        correct_codewords = 0

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

    def test_step(self, batch, batch_idx):
        Y, gt_codewords = batch

        logits = self(Y)
        loss = self.hungarian_loss(logits, gt_codewords)

        metrics = self.compute_accuracy(logits, gt_codewords)

        self.log('test/loss', loss, sync_dist=True)
        self.log('test/symbol_acc', metrics['symbol_acc'], sync_dist=True)
        self.log('test/codeword_acc', metrics['codeword_acc'], sync_dist=True)
        self.log('test/CER', 1.0 - metrics['codeword_acc'], sync_dist=True)

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

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """Clip gradients to prevent explosion."""
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")

"""
Unfolded BP Baseline for Codeword Demixing.

Neural-enhanced Belief Propagation unfolded into a neural network.
Each layer corresponds to one BP iteration with learnable components.
Uses the Tanner graph structure defined by H for message passing.

Standard BP:
  VN→CN: Variable nodes send messages to connected check nodes
  CN→VN: Check nodes send messages back to connected variable nodes

Neural BP enhancements:
  - Learnable message transformations (MLPs)
  - Learnable edge weights
  - Learnable damping factors

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


class UnfoldedBPLayer(nn.Module):
    """
    One iteration of neural-enhanced BP on Tanner graph.

    Message passing:
      1. VN→CN: Each VN sends transformed belief to connected CNs
      2. CN aggregation: Each CN aggregates messages from connected VNs
      3. CN→VN: Each CN sends transformed message back to connected VNs
      4. VN update: Each VN updates belief using channel evidence + CN messages

    Neural enhancements:
      - MLPs for message transformation
      - Learnable damping for stable training
    """

    def __init__(self, D_model: int, dropout: float = 0.1):
        super().__init__()
        self.D = D_model

        # VN→CN message transformation
        self.vn_to_cn = nn.Sequential(
            nn.LayerNorm(D_model),
            nn.Linear(D_model, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
        )

        # CN message aggregation and transformation
        self.cn_aggregate = nn.Sequential(
            nn.LayerNorm(D_model),
            nn.Linear(D_model, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
        )

        # CN→VN message transformation
        self.cn_to_vn = nn.Sequential(
            nn.LayerNorm(D_model),
            nn.Linear(D_model, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
        )

        # VN belief update (combines channel evidence with incoming messages)
        self.vn_update = nn.Sequential(
            nn.LayerNorm(D_model * 2),
            nn.Linear(D_model * 2, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
        )

        # Learnable damping factor
        self.damping = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        vn_belief: torch.Tensor,  # [B, N, D] current VN beliefs
        channel_emb: torch.Tensor,  # [B, N, D] channel evidence embedding (fixed)
        H: torch.Tensor,  # [M, N] parity check matrix
    ) -> torch.Tensor:
        """
        One BP iteration.

        Returns:
            vn_belief_new: [B, N, D] updated VN beliefs
        """
        B, N, D = vn_belief.shape
        M = H.shape[0]
        device = vn_belief.device

        # Create adjacency from H (ensure same device)
        H = H.to(device)
        adj = (H != 0).float()  # [M, N]

        # Degree normalization
        deg_cn = adj.sum(dim=1, keepdim=True).clamp(min=1)  # [M, 1]
        deg_vn = adj.sum(dim=0, keepdim=True).clamp(min=1)  # [1, N]

        # === VN → CN ===
        # Transform VN beliefs for sending to CNs
        vn_msg = self.vn_to_cn(vn_belief)  # [B, N, D]

        # Aggregate at each CN: sum of connected VN messages
        # cn_input[b, m] = sum_{n: H[m,n]!=0} vn_msg[b, n]
        cn_input = torch.einsum('mn,bnd->bmd', adj, vn_msg)  # [B, M, D]
        cn_input = cn_input / deg_cn.unsqueeze(0)  # normalize by degree

        # CN processes aggregated messages
        cn_state = self.cn_aggregate(cn_input)  # [B, M, D]

        # === CN → VN ===
        # Transform CN state for sending to VNs
        cn_msg = self.cn_to_vn(cn_state)  # [B, M, D]

        # Aggregate at each VN: sum of connected CN messages
        # vn_input[b, n] = sum_{m: H[m,n]!=0} cn_msg[b, m]
        vn_input = torch.einsum('mn,bmd->bnd', adj, cn_msg)  # [B, N, D]
        vn_input = vn_input / deg_vn.unsqueeze(-1)  # [1, N, 1] for broadcasting with [B, N, D]

        # === VN Update ===
        # Combine channel evidence with incoming CN messages
        vn_combined = torch.cat([channel_emb, vn_input], dim=-1)  # [B, N, 2D]
        vn_update = self.vn_update(vn_combined)  # [B, N, D]

        # Damped update for stability
        damp = torch.sigmoid(self.damping)
        vn_belief_new = damp * vn_belief + (1 - damp) * vn_update

        return vn_belief_new


class UnfoldedBPDemixer(L.LightningModule):
    """
    Unfolded BP baseline for codeword demixing.

    Architecture:
        Y [N, Q] -> Embed -> K copies -> Unfolded BP iterations -> Output heads -> [K, N, Q]

    Each iteration is a neural-enhanced BP layer operating on the Tanner graph.
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
        self.D = model_cfg.get('D_model', 256)
        self.num_iterations = model_cfg.get('num_layers', 6)
        self.dropout = model_cfg.get('dropout', 0.1)

        # Training config
        self.lr = config.training.learning_rate
        self.weight_decay = config.training.optimizer.weight_decay

        # Evidence encoder: Y -> channel embedding
        self.evidence_encoder = nn.Sequential(
            nn.LayerNorm(self.Q),
            nn.Linear(self.Q, self.D),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.D, self.D),
        )

        # Slot embeddings for symmetry breaking
        self.slot_embed = nn.Parameter(torch.randn(self.K, self.D) * 0.02)

        # Unfolded BP layers (shared or per-iteration)
        self.bp_layers = nn.ModuleList([
            UnfoldedBPLayer(self.D, self.dropout)
            for _ in range(self.num_iterations)
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
            H: [M, N] parity check matrix

        Returns:
            logits: [B, K, N, Q]
        """
        if H is None:
            H = self.H_matrix
        assert H is not None, "H matrix not provided"

        B, N, Q = Y.shape

        # Encode channel evidence (fixed throughout iterations)
        channel_emb = self.evidence_encoder(Y)  # [B, N, D]

        # Create K copies with slot embeddings
        vn_belief = channel_emb.unsqueeze(1) + self.slot_embed.view(1, self.K, 1, self.D)  # [B, K, N, D]
        channel_emb_k = channel_emb.unsqueeze(1).expand(-1, self.K, -1, -1).contiguous()  # [B, K, N, D]

        # Flatten K into batch for BP processing
        vn_belief_flat = vn_belief.reshape(B * self.K, N, self.D).contiguous()
        channel_emb_flat = channel_emb_k.reshape(B * self.K, N, self.D).contiguous()

        # Unfolded BP iterations
        for layer in self.bp_layers:
            vn_belief_flat = layer(vn_belief_flat, channel_emb_flat, H)

        # Reshape back
        vn_belief = vn_belief_flat.reshape(B, self.K, N, self.D)

        # Output projection
        vn_belief = self.out_norm(vn_belief)

        # Apply K independent output heads
        slot_logits = []
        for k in range(self.K):
            slot_logits.append(self.output_heads[k](vn_belief[:, k]))  # [B, N, Q]
        logits = torch.stack(slot_logits, dim=1)  # [B, K, N, Q]

        return logits

    def hungarian_loss(self, pred_logits, gt_codewords):
        """
        Compute cross-entropy loss with Hungarian matching.
        """
        B, K, N, Q = pred_logits.shape
        device = pred_logits.device

        total_loss = 0.0

        for b in range(B):
            cost = torch.zeros(K, K, device=device)

            for i in range(K):
                for j in range(K):
                    ce = F.cross_entropy(
                        pred_logits[b, i],
                        gt_codewords[b, j],
                        reduction='sum'
                    )
                    cost[i, j] = ce

            cost_np = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            for i, j in zip(row_ind, col_ind):
                total_loss += F.cross_entropy(
                    pred_logits[b, i],
                    gt_codewords[b, j],
                    reduction='mean'
                )

        return total_loss / B

    def training_step(self, batch, batch_idx):
        Y, X = batch

        logits = self(Y)
        loss = self.hungarian_loss(logits, X)

        preds = logits.argmax(dim=-1)

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

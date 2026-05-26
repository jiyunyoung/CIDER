"""
DiMP-Iterative: Iterative A+B without masking.

Same Module A (slot responsibility) and Module B (neural MP) as CIDER,
but replaces masked diffusion with simple iterative refinement:
- Iter 1: VN = slot_init (same as one-shot)
- Iter t>1: VN = softmax(logits_{t-1}) @ W_base + slot_init

No masking at any point. Trained end-to-end with loss on final iteration.
This isolates the contribution of masked diffusion from the URA-specific structure.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Optional

from utils.gf import build_gf_perm_table

# Reuse building blocks from cider_direct
from models.cider_direct import (
    SinusoidalPositionEmbedding,
    FeedForward,
    EdgeSelfAttention,
    SlotResponsibilityBlock,
    NeuralMPBlock,
    ResidualUpdateBlock,
)


class DiMPIterative(L.LightningModule):
    """
    Iterative A+B prediction without masking.

    Runs num_iters forward passes through the same A+B layers,
    feeding softmax(logits) back as VN initialization each iteration.
    Trained with Hungarian loss on the final iteration's output.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # Data config
        self.Q = int(config.data.Q)
        self.N = int(config.data.N)
        self.K = int(config.data.get('K', config.data.get('K_max', 2)))
        self.M = int(config.data.M)

        # Model config
        model_cfg = config.model
        self.D = int(model_cfg.D_model)
        self.num_layers = int(model_cfg.num_layers)
        self.heads = int(model_cfg.heads)
        self.mlp_ratio = int(model_cfg.get('mlp_ratio', 4))
        self.dropout = float(model_cfg.get('dropout', 0.1))
        self.tau_min = float(model_cfg.get('tau_min', 0.2))
        slot_init_scale = float(model_cfg.get('slot_init_scale', 0.5))
        self.num_iters = int(model_cfg.get('num_iters', 8))

        # Training config
        self.lr = config.training.learning_rate
        self.weight_decay = config.training.optimizer.weight_decay

        # Shared symbol embedding
        W_base_init = torch.empty(self.Q + 1, self.D)
        nn.init.orthogonal_(W_base_init)
        self.W_base = nn.Parameter(W_base_init.contiguous())

        # GF permutation table
        perm_table = build_gf_perm_table(self.Q)
        self.register_buffer('perm_table', perm_table)

        # Slot initialization
        rng_state = torch.get_rng_state()
        torch.manual_seed(0)
        slot_init = torch.zeros(self.K, self.D)
        nn.init.orthogonal_(slot_init)
        torch.set_rng_state(rng_state)
        self.slot_init = nn.Parameter(slot_init * slot_init_scale)

        # Slot responsibility blocks (shared across iterations)
        self.slot_blocks = nn.ModuleList([
            SlotResponsibilityBlock(Q=self.Q, D_model=self.D, dropout=self.dropout, tau_min=self.tau_min)
            for _ in range(self.num_layers)
        ])

        # Neural MP blocks (2 per layer, shared across iterations)
        self.mp_blocks = nn.ModuleList([
            nn.ModuleList([
                NeuralMPBlock(self.D, self.heads, self.Q, dropout=self.dropout, mlp_ratio=self.mlp_ratio)
                for _ in range(2)
            ])
            for _ in range(self.num_layers)
        ])

        # Residual update blocks (shared across iterations)
        self.update_blocks = nn.ModuleList([
            ResidualUpdateBlock(self.D, dropout=self.dropout, ffn_ratio=2)
            for _ in range(self.num_layers)
        ])

        # Output
        self.out_norm = nn.LayerNorm(self.D)
        self.output_proj = nn.Linear(self.D, self.Q)

        # Tanner graph cache
        self.register_buffer("_cn_nbr_idx", None, persistent=False)
        self.register_buffer("_cn_nbr_coef", None, persistent=False)
        self.register_buffer("_vn_nbr_idx", None, persistent=False)
        self.register_buffer("_vn_nbr_coef", None, persistent=False)
        self._cache_key = None

        # H matrix (set externally before training)
        self.register_buffer("H_matrix", None, persistent=False)

    def set_H_matrix(self, H: torch.Tensor):
        self.H_matrix = H
        self._build_tanner_cache(H)

    def _build_tanner_cache(self, H: torch.Tensor) -> None:
        cache_key = (H.shape, H.device)
        if self._cache_key == cache_key and self._cn_nbr_idx is not None:
            return

        M, N = H.shape
        device = H.device
        H_cpu = H.detach().cpu()

        cn_lists, cn_coefs = [], []
        dc_max = 0
        for j in range(M):
            idx = (H_cpu[j] != 0).nonzero(as_tuple=False).squeeze(-1).tolist()
            if isinstance(idx, int):
                idx = [idx]
            coef = [int(H_cpu[j, i].item()) for i in idx] if idx else []
            cn_lists.append(idx)
            cn_coefs.append(coef)
            dc_max = max(dc_max, len(idx))

        vn_lists, vn_coefs = [], []
        dv_max = 0
        for i in range(N):
            idx = (H_cpu[:, i] != 0).nonzero(as_tuple=False).squeeze(-1).tolist()
            if isinstance(idx, int):
                idx = [idx]
            coef = [int(H_cpu[j, i].item()) for j in idx] if idx else []
            vn_lists.append(idx)
            vn_coefs.append(coef)
            dv_max = max(dv_max, len(idx))

        dc_max = max(1, dc_max)
        dv_max = max(1, dv_max)

        cn_nbr_idx = torch.full((M, dc_max), -1, dtype=torch.long)
        cn_nbr_coef = torch.zeros((M, dc_max), dtype=torch.long)
        for j in range(M):
            n = len(cn_lists[j])
            if n > 0:
                cn_nbr_idx[j, :n] = torch.tensor(cn_lists[j], dtype=torch.long)
                cn_nbr_coef[j, :n] = torch.tensor(cn_coefs[j], dtype=torch.long)

        vn_nbr_idx = torch.full((N, dv_max), -1, dtype=torch.long)
        vn_nbr_coef = torch.zeros((N, dv_max), dtype=torch.long)
        for i in range(N):
            n = len(vn_lists[i])
            if n > 0:
                vn_nbr_idx[i, :n] = torch.tensor(vn_lists[i], dtype=torch.long)
                vn_nbr_coef[i, :n] = torch.tensor(vn_coefs[i], dtype=torch.long)

        self._cn_nbr_idx = cn_nbr_idx.to(device)
        self._cn_nbr_coef = cn_nbr_coef.to(device)
        self._vn_nbr_idx = vn_nbr_idx.to(device)
        self._vn_nbr_coef = vn_nbr_coef.to(device)
        self._cache_key = cache_key

    def _single_pass(self, Y, VN):
        """One forward pass through A+B layers."""
        B = Y.shape[0]
        U = self.W_base[:self.Q]

        for slot_block, mp_blocks_layer, update_block in zip(
            self.slot_blocks, self.mp_blocks, self.update_blocks
        ):
            VN_tilde = slot_block(VN=VN, Y=Y, W_base=self.W_base)

            VN_tilde_flat = VN_tilde.reshape(B * self.K, self.N, self.D)
            for mp_block in mp_blocks_layer:
                VN_tilde_flat = mp_block(
                    VN_tilde_flat,
                    self._cn_nbr_idx, self._cn_nbr_coef,
                    self._vn_nbr_idx, self._vn_nbr_coef,
                    U, self.perm_table,
                )
            VN_tilde = VN_tilde_flat.reshape(B, self.K, self.N, self.D)

            VN = update_block(VN_tilde)

        VN = self.out_norm(VN)
        logits = self.output_proj(VN)
        return logits

    def forward(self, Y: torch.Tensor, num_iters: Optional[int] = None) -> torch.Tensor:
        """
        Iterative prediction: T passes through A+B with feedback.

        Args:
            Y: [B, N, Q] soft scores from channel
            num_iters: override self.num_iters (useful at eval time)

        Returns:
            logits: [B, K, N, Q] final predicted logits
        """
        B, N, Q = Y.shape
        device = Y.device
        T = num_iters if num_iters is not None else self.num_iters

        assert self.H_matrix is not None, "H matrix not set. Call set_H_matrix() first."
        self._build_tanner_cache(self.H_matrix)

        U = self.W_base[:self.Q]
        prev_logits = None

        for iteration in range(T):
            if prev_logits is None:
                # First iteration: slot_init only
                VN = self.slot_init.view(1, self.K, 1, self.D).expand(B, -1, N, -1).clone()
            else:
                # Subsequent: embed previous soft predictions + slot_init
                probs = F.softmax(prev_logits, dim=-1)  # [B, K, N, Q]
                x_sym = torch.einsum('bknq,qd->bknd', probs, U)
                VN = x_sym + self.slot_init.view(1, self.K, 1, self.D)

            prev_logits = self._single_pass(Y, VN)

        return prev_logits

    def hungarian_loss(self, pred_logits, gt_codewords):
        """Permutation-invariant loss using Hungarian algorithm."""
        batch_size = pred_logits.shape[0]
        K = self.K

        total_loss = 0.0

        for b in range(batch_size):
            cost_matrix = torch.zeros(K, K, device=pred_logits.device)

            for i in range(K):
                for j in range(K):
                    ce = F.cross_entropy(
                        pred_logits[b, i],
                        gt_codewords[b, j],
                        reduction='sum'
                    )
                    cost_matrix[i, j] = ce

            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            for i, j in zip(row_ind, col_ind):
                total_loss += F.cross_entropy(
                    pred_logits[b, i],
                    gt_codewords[b, j],
                    reduction='mean'
                )

        return total_loss / batch_size

    def compute_accuracy(self, pred_logits, gt_codewords):
        """Compute symbol and codeword accuracy with Hungarian matching."""
        batch_size = pred_logits.shape[0]
        K = self.K
        N = self.N

        pred_codewords = pred_logits.argmax(dim=-1)

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

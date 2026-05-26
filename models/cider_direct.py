"""
DiMP-NoGRU-OneShot: One-shot prediction using DiMP-NoGRU architecture.

Same architecture as DiMP-NoGRU but without diffusion:
- Takes only Y as input (soft scores from channel)
- Predicts K codewords in one forward pass
- Uses Hungarian matching loss for permutation-invariant training

This is a fair comparison between iterative diffusion vs one-shot prediction
using the same underlying neural architecture.
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


# ============================================================
# Sinusoidal Position Embedding (kept for compatibility)
# ============================================================
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        if half_dim <= 0:
            return torch.zeros(t.shape[0], self.dim, device=device)
        scale = math.log(10000) / max(1, half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ============================================================
# adaLN Modulation
# ============================================================
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


# ============================================================
# FeedForward
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, D: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D * mlp_ratio, D),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Edge Self-Attention with Extrinsic Mask
# ============================================================
class EdgeSelfAttention(nn.Module):
    def __init__(self, D_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert D_model % num_heads == 0
        self.D = D_model
        self.H = num_heads
        self.dh = D_model // num_heads

        self.q_proj = nn.Linear(D_model, D_model)
        self.k_proj = nn.Linear(D_model, D_model)
        self.v_proj = nn.Linear(D_model, D_model)
        self.o_proj = nn.Linear(D_model, D_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.dh)

    def forward(self, edge_tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        BM, d_c, D = edge_tokens.shape

        Q = self.q_proj(edge_tokens).view(BM, d_c, self.H, self.dh)
        K = self.k_proj(edge_tokens).view(BM, d_c, self.H, self.dh)
        V = self.v_proj(edge_tokens).view(BM, d_c, self.H, self.dh)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        diag_mask = torch.eye(d_c, dtype=torch.bool, device=edge_tokens.device)
        scores = scores.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), -1e4)

        key_mask = ~pad_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(key_mask, -1e4)

        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(key_mask, 0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(BM, d_c, D)

        out = self.o_proj(out)
        return self.dropout(out)


# ============================================================
# Slot Responsibility Block (simplified for one-shot)
# ============================================================
class SlotResponsibilityBlock(nn.Module):
    def __init__(self, Q: int, D_model: int, dropout: float = 0.1, tau_min: float = 0.2):
        super().__init__()
        self.Q = Q
        self.D = D_model
        self.tau_min = tau_min

        self.tau_0 = nn.Parameter(torch.tensor(1.0))
        self.beta_0 = nn.Parameter(torch.tensor(1.0))

        self.vn_norm = nn.LayerNorm(D_model, elementwise_affine=False)

        self.trust_gate = nn.Sequential(
            nn.Linear(2 * D_model, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, VN: torch.Tensor, Y: torch.Tensor, W_base: torch.Tensor) -> torch.Tensor:
        B, K, N, D = VN.shape
        Q = self.Q

        # Fixed temperature for one-shot (no time-based schedule)
        tau = F.softplus(self.tau_0) + self.tau_min
        beta = F.softplus(self.beta_0)

        vn_norm = self.vn_norm(VN)
        score = torch.einsum('bknd,qd->bknq', vn_norm, W_base[:Q, :])
        score = score + beta * Y.unsqueeze(1)
        resp = F.softmax(score / tau, dim=1)

        y_weighted = resp * Y.unsqueeze(1)
        y_sym = torch.einsum('bknq,qd->bknd', y_weighted, W_base[:Q, :])
        y_assigned = y_sym

        gate_input = torch.cat([VN, y_assigned], dim=-1)
        g = self.trust_gate(gate_input)
        VN_tilde = g * VN + (1 - g) * y_assigned

        return VN_tilde


# ============================================================
# Neural MP Block
# ============================================================
class NeuralMPBlock(nn.Module):
    def __init__(self, D_model: int, num_heads: int, Q_field: int, dropout: float = 0.1, mlp_ratio: int = 4):
        super().__init__()
        self.D = D_model
        self.Q = Q_field

        self.vn_norm = nn.LayerNorm(D_model)
        self.edge_attn = EdgeSelfAttention(D_model, num_heads, dropout)
        self.edge_norm = nn.LayerNorm(D_model)

        self.energy_head = nn.Sequential(
            nn.Linear(D_model, D_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model * mlp_ratio, D_model),
            nn.Dropout(dropout),
        )
        self.energy_norm = nn.LayerNorm(D_model)

        self.agg_norm = nn.LayerNorm(D_model)
        self.agg_gate = nn.Sequential(
            nn.Linear(2 * D_model, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def _apply_gf_perm(self, V, coef, U, perm_table, transpose=False):
        BM, d_c, D = V.shape
        M = coef.shape[0]
        B = BM // M
        Q = U.shape[0]

        V_sym = torch.einsum('bmd,qd->bmq', V, U)

        coef_safe = coef.clamp(min=0, max=Q-1)
        perm_indices = perm_table[coef_safe]

        perm_expanded = perm_indices.unsqueeze(0).expand(B, -1, -1, -1)
        perm_expanded = perm_expanded.reshape(BM, d_c, Q)

        if transpose:
            V_perm = torch.zeros_like(V_sym)
            V_perm.scatter_(-1, perm_expanded, V_sym)
        else:
            V_perm = torch.gather(V_sym, -1, perm_expanded)

        V_out = torch.einsum('bmq,qd->bmd', V_perm, U)

        return V_out

    def forward(self, vn, cn_nbr_idx, cn_nbr_coef, vn_nbr_idx, vn_nbr_coef, U, perm_table):
        B, N, D = vn.shape
        M, d_c = cn_nbr_idx.shape

        vn_norm = self.vn_norm(vn)

        idx_safe = cn_nbr_idx.clamp(min=0)
        edge_msg = vn_norm[:, idx_safe, :]
        edge_msg = edge_msg.reshape(B * M, d_c, D)

        pad_mask = (cn_nbr_idx >= 0)
        pad_mask = pad_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * M, d_c)

        edge_msg_norm = self._apply_gf_perm(edge_msg, cn_nbr_coef, U, perm_table, transpose=False)

        edge_out = self.edge_attn(edge_msg_norm, pad_mask)
        edge_out = edge_msg_norm + edge_out
        edge_out = self.edge_norm(edge_out)

        energy_z = self.energy_head(edge_out)
        energy_z = edge_out + energy_z
        energy_z = self.energy_norm(energy_z)

        energy_x = self._apply_gf_perm(energy_z, cn_nbr_coef, U, perm_table, transpose=True)
        energy_x = energy_x.reshape(B, M, d_c, D)

        pad_mask_4d = pad_mask.reshape(B, M, d_c).unsqueeze(-1)
        energy_x = energy_x * pad_mask_4d.float()

        edge_dst = cn_nbr_idx.reshape(-1)
        edge_msg = energy_x.reshape(B, M * d_c, D)

        valid_mask = (edge_dst >= 0)
        edge_dst_safe = edge_dst.clamp(min=0)

        valid_mask_exp = valid_mask.unsqueeze(0).unsqueeze(-1).float()
        edge_msg = edge_msg * valid_mask_exp

        vn_agg = torch.zeros(B, N, D, device=vn.device, dtype=vn.dtype)
        edge_dst_exp = edge_dst_safe.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        vn_agg.scatter_add_(1, edge_dst_exp, edge_msg)

        degree = torch.zeros(N, device=vn.device, dtype=vn.dtype)
        degree.scatter_add_(0, edge_dst_safe, valid_mask.float())
        degree = degree.clamp(min=1.0)
        vn_agg = vn_agg / degree.sqrt().view(1, N, 1)

        vn_agg = self.agg_norm(vn_agg)

        gate_input = torch.cat([vn_agg, vn], dim=-1)
        g = self.agg_gate(gate_input)
        vn_out = g * vn_agg + (1 - g) * vn

        return vn_out


# ============================================================
# Residual Update Block
# ============================================================
class ResidualUpdateBlock(nn.Module):
    def __init__(self, D_model: int, dropout: float = 0.1, ffn_ratio: int = 2):
        super().__init__()
        self.D = D_model
        self.norm = nn.LayerNorm(D_model)
        self.ffn = nn.Sequential(
            nn.Linear(D_model, D_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model * ffn_ratio, D_model),
            nn.Dropout(dropout),
        )

    def forward(self, VN_tilde: torch.Tensor) -> torch.Tensor:
        VN_norm = self.norm(VN_tilde)
        VN_new = VN_tilde + self.ffn(VN_norm)
        return VN_new


# ============================================================
# DiMP-NoGRU-OneShot: Main Model (LightningModule)
# ============================================================
class DiMPNoGRUOneShot(L.LightningModule):
    """
    One-shot prediction using DiMP-NoGRU architecture.

    Takes Y (soft scores) and predicts K codewords directly without diffusion.
    Uses Hungarian matching loss for permutation-invariant training.
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

        # Slot responsibility blocks
        self.slot_blocks = nn.ModuleList([
            SlotResponsibilityBlock(Q=self.Q, D_model=self.D, dropout=self.dropout, tau_min=self.tau_min)
            for _ in range(self.num_layers)
        ])

        # Neural MP blocks (2 per layer)
        self.mp_blocks = nn.ModuleList([
            nn.ModuleList([
                NeuralMPBlock(self.D, self.heads, self.Q, dropout=self.dropout, mlp_ratio=self.mlp_ratio)
                for _ in range(2)
            ])
            for _ in range(self.num_layers)
        ])

        # Residual update blocks
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
        """Set the H matrix for message passing."""
        self.H_matrix = H
        self._build_tanner_cache(H)

    def _build_tanner_cache(self, H: torch.Tensor) -> None:
        """Build sparse neighbor lists from H matrix."""
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

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        One-shot prediction.

        Args:
            Y: [B, N, Q] soft scores from channel

        Returns:
            logits: [B, K, N, Q] predicted codeword logits
        """
        B, N, Q = Y.shape
        device = Y.device

        # Ensure H matrix is set
        assert self.H_matrix is not None, "H matrix not set. Call set_H_matrix() first."
        self._build_tanner_cache(self.H_matrix)

        U = self.W_base[:self.Q]

        # Initialize VN from slot_init only (no X_t input)
        # Each slot starts with just the slot embedding
        VN = self.slot_init.view(1, self.K, 1, self.D).expand(B, -1, N, -1).clone()

        for slot_block, mp_blocks_layer, update_block in zip(
            self.slot_blocks, self.mp_blocks, self.update_blocks
        ):
            # (1) Slot responsibility
            VN_tilde = slot_block(VN=VN, Y=Y, W_base=self.W_base)

            # (2) Neural message passing
            VN_tilde_flat = VN_tilde.reshape(B * self.K, self.N, self.D)
            for mp_block in mp_blocks_layer:
                VN_tilde_flat = mp_block(
                    VN_tilde_flat,
                    self._cn_nbr_idx, self._cn_nbr_coef,
                    self._vn_nbr_idx, self._vn_nbr_coef,
                    U, self.perm_table,
                )
            VN_tilde = VN_tilde_flat.reshape(B, self.K, self.N, self.D)

            # (3) Residual update
            VN = update_block(VN_tilde)

        VN = self.out_norm(VN)
        logits = self.output_proj(VN)

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

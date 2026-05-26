"""
No Tanner graph message passing.
Uses simple residual update and slot responsibility only.

Architecture:
  INIT: VN = X_soft @ W_base + slot_init

  For each layer:
    (1) VN_tilde <- SlotResponsibility(VN, Y, W_base)
    (2) VN <- VN_tilde + FFN(VN_tilde)  # residual, no MP
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from utils.gf import build_gf_perm_table


# ============================================================
# Sinusoidal Position / Time Embedding
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
# Slot Responsibility Block
# ============================================================
class SlotResponsibilityBlock(nn.Module):
    """Slot responsibility mechanism using shared W_base."""
    def __init__(
        self,
        Q: int,
        D_model: int,
        dropout: float = 0.1,
        tau_min: float = 0.2,
    ):
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

    def _get_schedule(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g_t = torch.cos(math.pi / 2 * (1 - t))
        tau_max = F.softplus(self.tau_0) + self.tau_min
        tau = self.tau_min + (tau_max - self.tau_min) * g_t
        beta = F.softplus(self.beta_0)
        tau = tau.view(-1, 1, 1, 1)
        beta = beta.view(-1, 1, 1, 1)
        return tau, beta

    def forward(
        self,
        VN: torch.Tensor,         # [B, K, N, D]
        Y: torch.Tensor,          # [B, N, Q]
        t: torch.Tensor,          # [B]
        W_base: torch.Tensor,     # [Q+1, D]
    ) -> torch.Tensor:
        B, K, N, D = VN.shape
        Q = self.Q

        tau, beta = self._get_schedule(t)

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
# Residual Update Block (replaces GRU)
# ============================================================
class ResidualUpdateBlock(nn.Module):
    """Simple residual update with AdaLN + FFN (no recurrent state)."""
    def __init__(self, D_model: int, dropout: float = 0.1, ffn_ratio: int = 2):
        super().__init__()
        self.D = D_model
        self.norm = nn.LayerNorm(D_model, elementwise_affine=False)
        self.adaLN = nn.Linear(D_model, 2 * D_model)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        self.ffn = nn.Sequential(
            nn.Linear(D_model, D_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model * ffn_ratio, D_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        VN_tilde: torch.Tensor,  # [B, K, N, D]
        VN: torch.Tensor,        # [B, K, N, D] (unused, kept for API compatibility)
        t_emb: torch.Tensor,     # [B, D]
    ) -> torch.Tensor:
        B, K, N, D = VN_tilde.shape

        # Simple residual: VN_new = VN_tilde + FFN(VN_tilde)
        shift, scale = self.adaLN(t_emb).chunk(2, dim=-1)
        VN_norm = modulate(self.norm(VN_tilde), shift.view(B, 1, 1, D), scale.view(B, 1, 1, D))
        VN_new = VN_tilde + self.ffn(VN_norm)

        return VN_new


# ============================================================
# DiMP-NoGRU-NoMP: Main Model
# ============================================================
class DiMP(nn.Module):
    """
    Architecture per layer:
    (1) SlotResponsibility: fuse VN with channel evidence Y
    (2) Residual: VN = VN_tilde + FFN(VN_tilde)

    No Tanner graph message passing - just slot competition and residual update.
    """
    def __init__(
        self,
        Q: int,
        N: int,
        K: int,
        M: int,
        D_model: int = 256,
        num_layers: int = 8,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        tau_min: float = 0.2,
        slot_init_scale: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.Q = int(Q)
        self.N = int(N)
        self.K = int(K)
        self.M = int(M)
        self.D = int(D_model)
        self.num_layers = int(num_layers)
        self.heads = int(heads)

        self.MASK_TOKEN = self.Q

        # Shared symbol embedding
        W_base_init = torch.empty(self.Q + 1, self.D)
        nn.init.orthogonal_(W_base_init)
        self.W_base = nn.Parameter(W_base_init.contiguous())

        # GF permutation table (kept for API compatibility, but not used)
        perm_table = build_gf_perm_table(self.Q)
        self.register_buffer('perm_table', perm_table)

        # Slot initialization
        rng_state = torch.get_rng_state()
        torch.manual_seed(0)
        slot_init = torch.zeros(self.K, self.D)
        nn.init.orthogonal_(slot_init)
        torch.set_rng_state(rng_state)
        self.slot_init = nn.Parameter(slot_init * slot_init_scale)

        # Time embedding
        self.time_embed = SinusoidalPositionEmbedding(self.D)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.D, self.D * 4),
            nn.GELU(),
            nn.Linear(self.D * 4, self.D),
        )

        # Slot responsibility blocks
        self.slot_blocks = nn.ModuleList([
            SlotResponsibilityBlock(Q=self.Q, D_model=self.D, dropout=dropout, tau_min=tau_min)
            for _ in range(self.num_layers)
        ])

        # Residual update blocks (instead of GRU) - NO MP blocks
        self.update_blocks = nn.ModuleList([
            ResidualUpdateBlock(self.D, dropout=dropout, ffn_ratio=2)
            for _ in range(self.num_layers)
        ])

        # Output
        self.out_norm = nn.LayerNorm(self.D)
        self.output_proj = nn.Linear(self.D, self.Q)

    def _build_initial_vn(
        self,
        X_t: torch.Tensor,
        soft_input: bool,
    ) -> torch.Tensor:
        """Build initial VN from X_t using shared W_base."""
        if soft_input:
            X_soft = X_t
            B, K, N, Q_dim = X_soft.shape
            x_sym = torch.einsum('bknq,qd->bknd', X_soft, self.W_base[:Q_dim, :])
        else:
            B, K, N = X_t.shape
            X_idx = X_t.long().clamp(0, self.Q)
            x_sym = self.W_base[X_idx]

        VN = x_sym + self.slot_init.view(1, K, 1, self.D)
        return VN

    def forward(
        self,
        X_t: torch.Tensor,
        Y_mag: torch.Tensor,
        syn: torch.Tensor,
        t: torch.Tensor,
        H: torch.Tensor,
        soft_input: bool = False,
    ) -> torch.Tensor:
        """Returns logits [B, K, N, Q]."""
        if soft_input:
            B, K, N, Q = X_t.shape
        else:
            B, K, N = X_t.shape

        device = X_t.device

        t = t.float().clamp(0.0, 1.0)
        t_emb = self.time_mlp(self.time_embed(t))

        VN = self._build_initial_vn(X_t, soft_input)

        for slot_block, update_block in zip(self.slot_blocks, self.update_blocks):
            # (1) Slot responsibility
            VN_tilde = slot_block(VN=VN, Y=Y_mag, t=t, W_base=self.W_base)

            # (2) Residual update (NO MP step)
            VN = update_block(VN_tilde=VN_tilde, VN=VN, t_emb=t_emb)

        VN = self.out_norm(VN)
        logits = self.output_proj(VN)

        return logits

    def get_hidden_states(
        self,
        X_t: torch.Tensor,
        Y_mag: torch.Tensor,
        syn: torch.Tensor,
        t: torch.Tensor,
        H: torch.Tensor,
        soft_input: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (logits, hidden_states, t_emb)"""
        if soft_input:
            B, K, N, Q = X_t.shape
        else:
            B, K, N = X_t.shape

        t = t.float().clamp(0.0, 1.0)
        t_emb = self.time_mlp(self.time_embed(t))

        VN = self._build_initial_vn(X_t, soft_input)

        for slot_block, update_block in zip(self.slot_blocks, self.update_blocks):
            VN_tilde = slot_block(VN=VN, Y=Y_mag, t=t, W_base=self.W_base)
            VN = update_block(VN_tilde=VN_tilde, VN=VN, t_emb=t_emb)

        VN = self.out_norm(VN)
        hidden_states = VN.reshape(B, K * self.N, self.D)
        logits = self.output_proj(VN)

        return logits, hidden_states, t_emb

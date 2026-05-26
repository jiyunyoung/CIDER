"""
DiMP-NoGRU-NoSlot: Diffusion Message Passing without GRU and Slot Responsibility

Double ablation: no recurrent state AND no slot competition.
Uses simple residual update and broadcasts Y to all slots equally.

Architecture:
  INIT: VN = X_soft @ W_base + slot_init

  For each layer:
    (1) VN_tilde = VN + Y_embed  # simple Y fusion, no competition
    (2) VN_tilde <- NeuralMP(VN_tilde)
    (3) VN <- VN_tilde + FFN(VN_tilde)  # residual instead of GRU
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
# Simple Y Fusion (replaces SlotResponsibility)
# ============================================================
class SimpleYFusion(nn.Module):
    """
    Simple Y fusion without slot competition.
    All slots receive the same Y embedding - no partitioning.
    """
    def __init__(self, Q: int, D_model: int, dropout: float = 0.1):
        super().__init__()
        self.Q = Q
        self.D = D_model

        # Project Y to embedding space
        self.y_proj = nn.Sequential(
            nn.Linear(Q, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
        )

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(2 * D_model, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        VN: torch.Tensor,         # [B, K, N, D]
        Y: torch.Tensor,          # [B, N, Q]
        t: torch.Tensor,          # [B] (unused, kept for API)
        W_base: torch.Tensor,     # [Q+1, D] (unused, kept for API)
    ) -> torch.Tensor:
        B, K, N, D = VN.shape

        # Project Y to D-dim: [B, N, D]
        Y_emb = self.y_proj(Y)

        # Broadcast to all slots: [B, K, N, D]
        Y_emb = Y_emb.unsqueeze(1).expand(-1, K, -1, -1)

        # Gated fusion
        gate_input = torch.cat([VN, Y_emb], dim=-1)
        g = self.gate(gate_input)
        VN_tilde = g * VN + (1 - g) * Y_emb

        return VN_tilde


# ============================================================
# Edge Self-Attention with Extrinsic Mask
# ============================================================
class EdgeSelfAttention(nn.Module):
    """
    Edge-based self-attention for check node processing.
    Implements extrinsic property: each edge (j,i) attends to other edges (j,l) where l≠i.
    """
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

    def forward(
        self,
        edge_tokens: torch.Tensor,  # [B*M, d_c, D]
        pad_mask: torch.Tensor,     # [B*M, d_c]
    ) -> torch.Tensor:
        BM, d_c, D = edge_tokens.shape

        Q = self.q_proj(edge_tokens).view(BM, d_c, self.H, self.dh)
        K = self.k_proj(edge_tokens).view(BM, d_c, self.H, self.dh)
        V = self.v_proj(edge_tokens).view(BM, d_c, self.H, self.dh)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Extrinsic mask: exclude self (diagonal)
        diag_mask = torch.eye(d_c, dtype=torch.bool, device=edge_tokens.device)
        scores = scores.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), -1e4)

        # Padding mask
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
# Neural MP Block
# ============================================================
class NeuralMPBlock(nn.Module):
    """Neural Message Passing block using edge-based self-attention."""
    def __init__(self, D_model: int, num_heads: int, Q_field: int, dropout: float = 0.1, mlp_ratio: int = 4):
        super().__init__()
        self.D = D_model
        self.Q = Q_field

        self.vn_norm = nn.LayerNorm(D_model, elementwise_affine=False)
        self.edge_attn = EdgeSelfAttention(D_model, num_heads, dropout)
        self.edge_norm = nn.LayerNorm(D_model, elementwise_affine=False)

        self.energy_head = nn.Sequential(
            nn.Linear(D_model, D_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model * mlp_ratio, D_model),
            nn.Dropout(dropout),
        )
        self.energy_norm = nn.LayerNorm(D_model, elementwise_affine=False)

        self.agg_norm = nn.LayerNorm(D_model, elementwise_affine=False)
        self.agg_gate = nn.Sequential(
            nn.Linear(2 * D_model, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
            nn.Sigmoid(),
        )

        self.adaLN = nn.Linear(D_model, 4 * D_model)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

        self.dropout = nn.Dropout(dropout)

    def _apply_gf_perm(
        self,
        V: torch.Tensor,
        coef: torch.Tensor,
        U: torch.Tensor,
        perm_table: torch.Tensor,
        transpose: bool = False,
    ) -> torch.Tensor:
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

    def forward(
        self,
        vn: torch.Tensor,
        t_emb: torch.Tensor,
        cn_nbr_idx: torch.Tensor,
        cn_nbr_coef: torch.Tensor,
        vn_nbr_idx: torch.Tensor,
        vn_nbr_coef: torch.Tensor,
        U: torch.Tensor,
        perm_table: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = vn.shape
        M, d_c = cn_nbr_idx.shape

        shift1, scale1, shift2, scale2 = self.adaLN(t_emb).chunk(4, dim=-1)

        vn_norm = modulate(self.vn_norm(vn), shift1.unsqueeze(1), scale1.unsqueeze(1))

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

        vn_agg = modulate(self.agg_norm(vn_agg), shift2.unsqueeze(1), scale2.unsqueeze(1))

        gate_input = torch.cat([vn_agg, vn], dim=-1)
        g = self.agg_gate(gate_input)
        vn_out = g * vn_agg + (1 - g) * vn

        return vn_out


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
# DiMP-NoGRU-NoSlot: Main Model
# ============================================================
class DiMP(nn.Module):
    """
    DiMP without GRU and without Slot Responsibility.

    Architecture per layer:
    (1) SimpleYFusion: broadcast Y to all slots (no competition)
    (2) NeuralMP: edge-based message passing with exact GF permutations
    (3) Residual: VN = VN_tilde + FFN(VN_tilde)
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

        # Time embedding
        self.time_embed = SinusoidalPositionEmbedding(self.D)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.D, self.D * 4),
            nn.GELU(),
            nn.Linear(self.D * 4, self.D),
        )

        # Simple Y fusion blocks (no slot competition)
        self.y_fusion_blocks = nn.ModuleList([
            SimpleYFusion(Q=self.Q, D_model=self.D, dropout=dropout)
            for _ in range(self.num_layers)
        ])

        # Neural MP blocks (2 per layer)
        self.mp_blocks = nn.ModuleList([
            nn.ModuleList([
                NeuralMPBlock(self.D, self.heads, self.Q, dropout=dropout, mlp_ratio=mlp_ratio)
                for _ in range(2)
            ])
            for _ in range(self.num_layers)
        ])

        # Residual update blocks (instead of GRU)
        self.update_blocks = nn.ModuleList([
            ResidualUpdateBlock(self.D, dropout=dropout, ffn_ratio=2)
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
        assert H is not None
        self._build_tanner_cache(H)

        if soft_input:
            B, K, N, Q = X_t.shape
        else:
            B, K, N = X_t.shape

        device = X_t.device

        U = self.W_base[:self.Q]

        t = t.float().clamp(0.0, 1.0)
        t_emb = self.time_mlp(self.time_embed(t))
        t_emb_flat = t_emb.unsqueeze(1).expand(B, K, self.D).reshape(B * K, self.D)

        VN = self._build_initial_vn(X_t, soft_input)

        for y_fusion, mp_blocks_layer, update_block in zip(
            self.y_fusion_blocks, self.mp_blocks, self.update_blocks
        ):
            # (1) Simple Y fusion (no slot competition)
            VN_tilde = y_fusion(VN=VN, Y=Y_mag, t=t, W_base=self.W_base)

            # (2) Neural message passing
            VN_tilde_flat = VN_tilde.reshape(B * K, self.N, self.D)
            for mp_block in mp_blocks_layer:
                VN_tilde_flat = mp_block(
                    VN_tilde_flat, t_emb_flat,
                    self._cn_nbr_idx, self._cn_nbr_coef,
                    self._vn_nbr_idx, self._vn_nbr_coef,
                    U, self.perm_table,
                )
            VN_tilde = VN_tilde_flat.reshape(B, K, self.N, self.D)

            # (3) Residual update (instead of GRU)
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
        assert H is not None
        self._build_tanner_cache(H)

        if soft_input:
            B, K, N, Q = X_t.shape
        else:
            B, K, N = X_t.shape

        U = self.W_base[:self.Q]

        t = t.float().clamp(0.0, 1.0)
        t_emb = self.time_mlp(self.time_embed(t))
        t_emb_flat = t_emb.unsqueeze(1).expand(B, K, self.D).reshape(B * K, self.D)

        VN = self._build_initial_vn(X_t, soft_input)

        for y_fusion, mp_blocks_layer, update_block in zip(
            self.y_fusion_blocks, self.mp_blocks, self.update_blocks
        ):
            VN_tilde = y_fusion(VN=VN, Y=Y_mag, t=t, W_base=self.W_base)
            VN_tilde_flat = VN_tilde.reshape(B * K, self.N, self.D)
            for mp_block in mp_blocks_layer:
                VN_tilde_flat = mp_block(
                    VN_tilde_flat, t_emb_flat,
                    self._cn_nbr_idx, self._cn_nbr_coef,
                    self._vn_nbr_idx, self._vn_nbr_coef,
                    U, self.perm_table,
                )
            VN_tilde = VN_tilde_flat.reshape(B, K, self.N, self.D)
            VN = update_block(VN_tilde=VN_tilde, VN=VN, t_emb=t_emb)

        VN = self.out_norm(VN)
        hidden_states = VN.reshape(B, K * self.N, self.D)
        logits = self.output_proj(VN)

        return logits, hidden_states, t_emb

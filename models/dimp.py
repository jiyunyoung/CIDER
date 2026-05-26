"""
DIMP: Diffusion Message Passing for ECC Demixing

Architecture (per iteration):
  (1) VN ← Observation(Y)     # claim + demix
  (2) CN ← VN cross-attn      # build parity context
  (3) VN ← CN cross-attn      # verify + correct

Key design:
  - Demixing: VN ← Y (slots compete to explain Y, no slot-to-slot shortcuts)
  - Error correction: VN ← CN (parity-driven verification)
  - No VN self-attention, no CN → CN, no GF arithmetic in layers
  - CN initialized as learned parameters (neutral critics)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


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
# adaLN Modulation (for Adapter compatibility)
# ============================================================
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class DDiTFinalLayer(nn.Module):
    def __init__(self, D_model: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(D_model)
        self.linear = nn.Linear(D_model, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * D_model)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.linear(x)


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
# Observation Encoder (Y → embedding with position)
# ============================================================
class ObservationEncoder(nn.Module):
    """
    Encodes observation Y into position-dependent embeddings.

    y_emb[n] represents: "What does the channel say about symbol n?"
    Includes:
      - Soft likelihood over GF(q) symbols at position n
      - Noise/reliability at that position
      - Shared position embedding (same as VN)

    Position dependence is critical - without it, slots can't decide
    which positions they should explain, leading to collapse.
    """
    def __init__(self, Q: int, D_model: int, pos_embedding: nn.Embedding, dropout: float = 0.1):
        super().__init__()
        self.D = D_model

        # Project soft scores to embedding: LayerNorm → Linear → GELU → Dropout → Linear
        self.proj = nn.Sequential(
            nn.LayerNorm(Q),
            nn.Linear(Q, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
        )

        # Shared position embedding (from parent DiMP)
        self.pos_embedding = pos_embedding

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """Y: [B, N, Q] -> [B, N, D]"""
        B, N, Q = Y.shape
        device = Y.device

        # L1 normalize to probability (NOT log_softmax - that kills signal for small Y values)
        Y_prob = Y.float() / (Y.float().sum(dim=-1, keepdim=True) + 1e-8)  # [B, N, Q]
        y_emb = self.proj(Y_prob)  # [B, N, D]

        # Add shared position embedding
        pos = self.pos_embedding(torch.arange(N, device=device))  # [N, D]
        y_emb = y_emb + pos.unsqueeze(0)  # [B, N, D]

        return y_emb


# ============================================================
# Sparse Neighbor Cross-Attention (Tanner graph message passing)
# ============================================================
class SparseNeighborCrossAttention(nn.Module):
    """
    Cross-attention where queries attend only to their Tanner graph neighbors.

    For CN←VN: CN_j attends to {VN_i : H[j,i] != 0}
    For VN←CN: VN_i attends to {CN_j : H[j,i] != 0}

    GF(q) coefficients H[j,i] are used for coefficient-conditioned affine
    transform of values (no attention bias - that doesn't match BP semantics).

    V_eff = V * (1 + scale(coef_emb)) + shift(coef_emb)
    """
    def __init__(self, D_model: int, num_heads: int, Q_field: int, dropout: float = 0.1):
        super().__init__()
        assert D_model % num_heads == 0
        self.D = D_model
        self.H = num_heads
        self.dh = D_model // num_heads

        self.q_proj = nn.Linear(D_model, D_model)
        self.k_proj = nn.Linear(D_model, D_model)
        self.v_proj = nn.Linear(D_model, D_model)
        self.o_proj = nn.Linear(D_model, D_model)

        # Coefficient-conditioned affine transform (Option B)
        # coef_emb: maps GF(q) coefficient to embedding
        # coef_scale/shift: maps embedding to per-head scale/shift for value transform
        self.coef_emb = nn.Embedding(Q_field, num_heads)
        self.coef_scale = nn.Linear(num_heads, self.dh)
        self.coef_shift = nn.Linear(num_heads, self.dh)
        # Initialize: coef_emb with small random (so coefficients are distinguishable)
        # scale/shift with zeros (start with identity transform)
        nn.init.normal_(self.coef_emb.weight, mean=0, std=0.02)
        nn.init.zeros_(self.coef_scale.weight)
        nn.init.zeros_(self.coef_scale.bias)
        nn.init.zeros_(self.coef_shift.weight)
        nn.init.zeros_(self.coef_shift.bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.dh)

    def forward(
        self,
        q_nodes: torch.Tensor,    # [B, Lq, D]
        kv_nodes: torch.Tensor,   # [B, Lk, D]
        nbr_idx: torch.Tensor,    # [Lq, max_deg] indices into kv_nodes, -1 for padding
        nbr_coef: torch.Tensor,   # [Lq, max_deg] GF(q) coefficients
    ) -> torch.Tensor:
        """Returns: [B, Lq, D]"""
        B, Lq, D = q_nodes.shape
        _, Lk, _ = kv_nodes.shape
        max_deg = nbr_idx.shape[1]

        # Project Q, K, V
        Q = self.q_proj(q_nodes).view(B, Lq, self.H, self.dh)
        K = self.k_proj(kv_nodes).view(B, Lk, self.H, self.dh)
        V = self.v_proj(kv_nodes).view(B, Lk, self.H, self.dh)

        # Gather neighbor K, V
        idx_safe = nbr_idx.clamp(min=0)
        K_nbr = K[:, idx_safe, :, :]  # [B, Lq, max_deg, H, dh]
        V_nbr = V[:, idx_safe, :, :]  # [B, Lq, max_deg, H, dh]

        # Coefficient-conditioned affine transform of values
        # V_eff = V * (1 + scale) + shift
        coef_e = self.coef_emb(nbr_coef.clamp(min=0))  # [Lq, max_deg, H]
        coef_scale = self.coef_scale(coef_e)  # [Lq, max_deg, dh]
        coef_shift = self.coef_shift(coef_e)  # [Lq, max_deg, dh]
        # Expand for batch and heads: [1, Lq, max_deg, 1, dh]
        coef_scale = coef_scale.unsqueeze(0).unsqueeze(3)
        coef_shift = coef_shift.unsqueeze(0).unsqueeze(3)
        # Apply affine transform: [B, Lq, max_deg, H, dh]
        V_nbr = V_nbr * (1 + coef_scale) + coef_shift

        # Attention scores (no coefficient bias)
        scores = (Q.unsqueeze(2) * K_nbr).sum(dim=-1) * self.scale  # [B, Lq, max_deg, H]

        # Mask padded positions
        pad_mask = (nbr_idx < 0).unsqueeze(0).unsqueeze(-1)
        scores = scores.masked_fill(pad_mask, -1e9)

        # Softmax over neighbors
        attn = F.softmax(scores, dim=2)
        attn = attn.masked_fill(pad_mask, 0.0)
        attn = self.dropout(attn)

        # Weighted sum
        out = (attn.unsqueeze(-1) * V_nbr).sum(dim=2)  # [B, Lq, H, dh]

        # Degree normalization
        deg = (~pad_mask.squeeze(-1)).sum(dim=2).clamp(min=1).float()
        out = out / deg.unsqueeze(-1).unsqueeze(-1).sqrt()

        out = out.reshape(B, Lq, D)
        out = self.o_proj(out)
        return self.dropout(out)


# ============================================================
# Slot Competition Block (Dense Self-Attention for Demixing)
# ============================================================
class SlotCompetitionBlock(nn.Module):
    """
    Simple dense self-attention block for slot competition.

    Structure per iteration:
        (0) VN ← VN + alpha_y * Y_emb   (channel anchor, like BP)
        (1) VN ← VN + DenseSelfAttn(VN) (slots + positions compete)
        (2) VN ← VN + FFN(VN)           (refinement)

    Key invariants:
        - No CN information here
        - No parity features here
        - No masking here
        - Soft embeddings only
        - Same Y for all slots
        - Slot differentiation comes from self-attn
    """
    def __init__(self, D_model: int, num_heads: int, dropout: float = 0.1, mlp_ratio: int = 4):
        super().__init__()
        self.D = D_model

        # Y anchor weight (like BP channel likelihood)
        self.alpha_y = nn.Parameter(torch.ones(1))

        # Dense self-attention over all VN tokens
        self.norm1 = nn.LayerNorm(D_model, elementwise_affine=False)
        self.dense_attn = nn.MultiheadAttention(D_model, num_heads, dropout=dropout, batch_first=True)

        # FFN
        self.norm2 = nn.LayerNorm(D_model, elementwise_affine=False)
        self.ff = FeedForward(D_model, mlp_ratio, dropout)

        # adaLN for time conditioning
        self.adaLN = nn.Linear(D_model, 4 * D_model)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        vn: torch.Tensor,    # [B, K, N, D]
        y_emb: torch.Tensor, # [B, N, D]
        t_emb: torch.Tensor, # [B, D]
    ) -> torch.Tensor:
        """Returns: updated vn [B, K, N, D]"""
        B, K, N, D = vn.shape

        # (0) Channel / observation anchor
        # Y anchor (LLR analogue) - keeps hypotheses grounded
        # Broadcast y_emb to [B, K, N, D]
        y_broadcast = y_emb.unsqueeze(1).expand(B, K, N, D)
        vn = vn + self.alpha_y * y_broadcast

        # Flatten to [B, K*N, D] for dense self-attention
        vn_flat = vn.reshape(B, K * N, D)

        # adaLN modulation
        shift1, scale1, shift2, scale2 = self.adaLN(t_emb).chunk(4, dim=-1)

        # (1) Dense self-attention on VN
        # Slots + positions can interact - this is where demixing happens
        vn_norm = modulate(self.norm1(vn_flat), shift1.unsqueeze(1), scale1.unsqueeze(1))
        vn_attn, _ = self.dense_attn(vn_norm, vn_norm, vn_norm, need_weights=False)
        vn_flat = vn_flat + self.dropout(vn_attn)

        # (2) Feedforward refinement
        vn_flat = vn_flat + self.ff(modulate(self.norm2(vn_flat), shift2.unsqueeze(1), scale2.unsqueeze(1)))

        # Reshape back to [B, K, N, D]
        vn = vn_flat.reshape(B, K, N, D)

        return vn


# ============================================================
# Tanner Message Block (CN ↔ VN only, no observation)
# ============================================================
class TannerMessageBlock(nn.Module):
    """
    Tanner graph message passing with adaLN:
    (1) CN ← VN (sparse cross-attention) - build parity context
    (2) VN ← CN (sparse cross-attention) - verify + correct

    Note: VN ← Y is handled by SlotCompetitionBlock separately.
    """
    def __init__(self, D_model: int, num_heads: int, Q_field: int, dropout: float = 0.1, mlp_ratio: int = 4):
        super().__init__()

        # (1) CN ← VN cross-attention
        self.cn_norm_q = nn.LayerNorm(D_model, elementwise_affine=False)
        self.cn_norm_kv = nn.LayerNorm(D_model, elementwise_affine=False)
        self.cn_cross_attn = SparseNeighborCrossAttention(D_model, num_heads, Q_field, dropout)
        self.cn_norm_ff = nn.LayerNorm(D_model, elementwise_affine=False)
        self.cn_ff = FeedForward(D_model, mlp_ratio, dropout)
        self.cn_adaLN = nn.Linear(D_model, 4 * D_model)
        nn.init.zeros_(self.cn_adaLN.weight)
        nn.init.zeros_(self.cn_adaLN.bias)

        # (2) VN ← CN cross-attention
        self.vn_norm_q = nn.LayerNorm(D_model, elementwise_affine=False)
        self.vn_norm_kv = nn.LayerNorm(D_model, elementwise_affine=False)
        self.vn_cross_attn = SparseNeighborCrossAttention(D_model, num_heads, Q_field, dropout)
        self.vn_norm_ff = nn.LayerNorm(D_model, elementwise_affine=False)
        self.vn_ff = FeedForward(D_model, mlp_ratio, dropout)
        self.vn_adaLN = nn.Linear(D_model, 4 * D_model)
        nn.init.zeros_(self.vn_adaLN.weight)
        nn.init.zeros_(self.vn_adaLN.bias)

    def forward(
        self,
        vn: torch.Tensor,          # [B*K, N, D]
        cn: torch.Tensor,          # [B*K, M, D]
        t_emb: torch.Tensor,       # [B*K, D] time embedding for adaLN
        cn_nbr_idx: torch.Tensor,  # [M, dc_max]
        cn_nbr_coef: torch.Tensor, # [M, dc_max]
        vn_nbr_idx: torch.Tensor,  # [N, dv_max]
        vn_nbr_coef: torch.Tensor, # [N, dv_max]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: (updated_vn, updated_cn)"""

        # (1) CN ← VN (build parity context)
        cn_shift1, cn_scale1, cn_shift2, cn_scale2 = self.cn_adaLN(t_emb).chunk(4, dim=-1)
        cn_q = modulate(self.cn_norm_q(cn), cn_shift1.unsqueeze(1), cn_scale1.unsqueeze(1))
        vn_kv = self.cn_norm_kv(vn)
        cn_delta = self.cn_cross_attn(cn_q, vn_kv, cn_nbr_idx, cn_nbr_coef)
        cn = cn + cn_delta
        cn = cn + self.cn_ff(modulate(self.cn_norm_ff(cn), cn_shift2.unsqueeze(1), cn_scale2.unsqueeze(1)))

        # (2) VN ← CN (verify + correct)
        vn_shift1, vn_scale1, vn_shift2, vn_scale2 = self.vn_adaLN(t_emb).chunk(4, dim=-1)
        vn_q = modulate(self.vn_norm_q(vn), vn_shift1.unsqueeze(1), vn_scale1.unsqueeze(1))
        cn_kv = self.vn_norm_kv(cn)
        vn_delta = self.vn_cross_attn(vn_q, cn_kv, vn_nbr_idx, vn_nbr_coef)
        vn = vn + vn_delta
        vn = vn + self.vn_ff(modulate(self.vn_norm_ff(vn), vn_shift2.unsqueeze(1), vn_scale2.unsqueeze(1)))

        return vn, cn


# ============================================================
# DIMP: Diffusion Message Passing (Main Model)
# ============================================================
class DiMP(nn.Module):
    """
    DIMP: Diffusion Message Passing for ECC Demixing.

    Per iteration:
    (1) VN ← Y        # claim + demix
    (2) CN ← VN       # build parity context
    (3) VN ← CN       # verify + correct
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

        # === Embeddings ===
        self.sym_embedding = nn.Embedding(self.Q + 1, self.D)  # +1 for MASK
        self.pos_embedding = nn.Embedding(self.N, self.D)
        self.check_idx_embedding = nn.Embedding(self.M, self.D)

        # Learnable slot-specific initialization for VN (symmetry breaking)
        self.slot_init = nn.Parameter(torch.randn(self.K, self.D) * 0.02)

        # Learnable CN initialization (neutral critics, small norm to prevent early over-confidence)
        self.cn_init = nn.Parameter(torch.randn(self.M, self.D) * 0.01)

        # Observation encoder (Y → embedding with shared position embedding)
        self.obs_encoder = ObservationEncoder(self.Q, self.D, self.pos_embedding, dropout=dropout)

        # Time embedding
        self.time_embed = SinusoidalPositionEmbedding(self.D)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.D, self.D * 4),
            nn.GELU(),
            nn.Linear(self.D * 4, self.D),
        )

        # VN fusion: [sym, pos, y_emb] → D (time via adaLN, not concat)
        self.vn_fuse = nn.Sequential(
            nn.Linear(3 * self.D, 4 * self.D),
            nn.LayerNorm(4 * self.D),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.D, self.D),
        )

        # Slot competition blocks (VN ← Y with dense slot attention)
        self.slot_blocks = nn.ModuleList([
            SlotCompetitionBlock(self.D, self.heads, dropout=dropout, mlp_ratio=mlp_ratio)
            for _ in range(self.num_layers)
        ])

        # Tanner message blocks (CN ↔ VN sparse attention)
        self.tanner_blocks = nn.ModuleList([
            TannerMessageBlock(self.D, self.heads, self.Q, dropout=dropout, mlp_ratio=mlp_ratio)
            for _ in range(self.num_layers)
        ])

        # Output (K independent MLP heads)
        self.out_norm = nn.LayerNorm(self.D)
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.D, self.D),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.D, self.Q),
            )
            for _ in range(self.K)
        ])

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

        # CN neighbors (check j → which variables)
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

        # VN neighbors (variable i → which checks)
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

    def _build_vn_tokens(
        self,
        X_t: torch.Tensor,
        y_emb: torch.Tensor,
        soft_input: bool = False
    ) -> torch.Tensor:
        """
        Build initial VN tokens [B, K, N, D].
        Time conditioning is via adaLN in blocks, not concatenation.

        Args:
            X_t: Either [B, K, N] hard symbols or [B, K, N, Q] soft beliefs
            y_emb: [B, N, D] observation embedding (from obs_encoder)
            soft_input: If True, X_t is soft beliefs [B, K, N, Q]
        """
        if soft_input:
            B, K, N, Q = X_t.shape
            device = X_t.device
            # Weighted sum of embeddings (exclude MASK token)
            sym = torch.einsum('bknq,qd->bknd', X_t, self.sym_embedding.weight[:Q])
        else:
            B, K, N = X_t.shape
            device = X_t.device
            sym = self.sym_embedding(X_t)  # [B, K, N, D]

        pos = self.pos_embedding(torch.arange(N, device=device)).view(1, 1, N, self.D).expand(B, K, -1, -1)

        # Expand y_emb to [B, K, N, D] (same observation for all slots)
        y_exp = y_emb.unsqueeze(1).expand(B, K, N, self.D)

        # Fuse [sym, pos, y_emb] → VN
        vn = self.vn_fuse(torch.cat([sym, pos, y_exp], dim=-1))

        # Add slot-specific initialization for symmetry breaking
        slot_init = self.slot_init.view(1, K, 1, self.D).expand(B, -1, N, -1)
        vn = vn + slot_init

        return vn

    def _build_cn_tokens(self, B: int, K: int, device: torch.device) -> torch.Tensor:
        """Build initial CN tokens [B, K, M, D] from learned initialization."""
        # CN starts as neutral critics with learned initialization
        cn = self.cn_init.view(1, 1, self.M, self.D).expand(B, K, -1, -1)

        # Add check index embedding
        chk = self.check_idx_embedding(torch.arange(self.M, device=device)).view(1, 1, self.M, self.D)
        cn = cn + chk.expand(B, K, -1, -1)

        return cn.clone()  # Clone to allow gradient flow

    def forward(
        self,
        X_t: torch.Tensor,    # [B, K, N] hard or [B, K, N, Q] soft
        Y_mag: torch.Tensor,  # [B, N, Q]
        syn: torch.Tensor,    # [B, K, M] (unused in new architecture)
        t: torch.Tensor,      # [B]
        H: torch.Tensor,      # [M, N]
        soft_input: bool = False,  # If True, X_t is soft beliefs [B, K, N, Q]
    ) -> torch.Tensor:
        """
        Returns logits [B, K, N, Q].

        Option C: VN flows through layers, Y re-injected each layer.
        - Build VN once from X_t
        - Each layer: VN += alpha_y * y_emb (anchor to observation)
        - VN accumulates features through layers
        - Output logits once at the end
        """
        assert H is not None, "H matrix must be provided"
        self._build_tanner_cache(H)

        if soft_input:
            B, K, N, Q = X_t.shape
        else:
            B, K, N = X_t.shape

        M = self.M
        device = X_t.device

        # Time embedding
        t = t.float().clamp(0.0, 1.0)
        t_emb = self.time_mlp(self.time_embed(t))  # [B, D]

        # Encode observation Y (constant across layers)
        y_emb = self.obs_encoder(Y_mag)  # [B, N, D]

        # Build VN once from input with Y included (like DiBP)
        vn = self._build_vn_tokens(X_t, y_emb, soft_input=soft_input)  # [B, K, N, D]

        # Initialize CN
        cn = self._build_cn_tokens(B, K, device)  # [B, K, M, D]
        cn = cn.reshape(B * K, M, self.D)

        # Expand t_emb for B*K batch size (for adaLN in Tanner blocks)
        t_emb_flat = t_emb.unsqueeze(1).expand(B, K, self.D).reshape(B * K, self.D)

        # VN flows through layers, Y re-injected each layer
        for slot_block, tanner_block in zip(self.slot_blocks, self.tanner_blocks):
            # (1) Slot competition: VN + Y anchor, dense self-attn, FFN
            vn = slot_block(vn, y_emb, t_emb)  # [B, K, N, D]

            # Flatten for Tanner message passing
            vn = vn.reshape(B * K, N, self.D)

            # (2-3) CN ← VN, VN ← CN: Tanner graph message passing
            vn, cn = tanner_block(
                vn, cn, t_emb_flat,
                self._cn_nbr_idx, self._cn_nbr_coef,
                self._vn_nbr_idx, self._vn_nbr_coef,
            )

            # Reshape back for next slot_block
            vn = vn.reshape(B, K, N, self.D)

        # Output logits once at the end using K independent heads
        vn = self.out_norm(vn)  # [B, K, N, D]

        # Apply K independent output heads
        slot_logits = []
        for k in range(self.K):
            slot_logits.append(self.output_heads[k](vn[:, k, :, :]))  # [B, N, Q]
        logits = torch.stack(slot_logits, dim=1)  # [B, K, N, Q]

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

        M = self.M
        device = X_t.device

        t = t.float().clamp(0.0, 1.0)
        t_emb = self.time_mlp(self.time_embed(t))

        y_emb = self.obs_encoder(Y_mag)

        # Build VN once from input with Y included
        vn = self._build_vn_tokens(X_t, y_emb, soft_input=soft_input)

        cn = self._build_cn_tokens(B, K, device)
        cn = cn.reshape(B * K, M, self.D)

        t_emb_flat = t_emb.unsqueeze(1).expand(B, K, self.D).reshape(B * K, self.D)

        # VN flows through layers
        for slot_block, tanner_block in zip(self.slot_blocks, self.tanner_blocks):
            vn = slot_block(vn, y_emb, t_emb)
            vn = vn.reshape(B * K, N, self.D)
            vn, cn = tanner_block(
                vn, cn, t_emb_flat,
                self._cn_nbr_idx, self._cn_nbr_coef,
                self._vn_nbr_idx, self._vn_nbr_coef,
            )
            vn = vn.reshape(B, K, N, self.D)

        vn = self.out_norm(vn)
        hidden_states = vn.reshape(B, K * N, self.D)

        # Apply K independent output heads
        slot_logits = []
        for k in range(self.K):
            slot_logits.append(self.output_heads[k](vn[:, k, :, :]))
        logits = torch.stack(slot_logits, dim=1)

        return logits, hidden_states, t_emb

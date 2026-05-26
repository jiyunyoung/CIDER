"""
cider with gru
Architecture:
  INIT: VN = X_soft @ W_base + slot_init

  For each layer:
    (1) VN_tilde <- SlotResponsibility(VN, Y, W_base)
    (2) VN_tilde, CN <- TannerMP(VN_tilde, CN, T_all)
        - CN <- VN: V' = T_a @ V (normalize)
        - VN <- CN: V' = T_a^T @ V (denormalize)
    (3) VN <- GRU(VN_tilde, VN)
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
    """
    Edge-based self-attention for check node processing.
    Implements extrinsic property: each edge (j,i) attends to other edges (j,l) where l≠i.

    Batched implementation:
    - All checks processed in parallel
    - Edges padded to max degree d_c
    - Off-diagonal mask for extrinsic property
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
        edge_tokens: torch.Tensor,  # [B*M, d_c, D] - edge messages per check
        pad_mask: torch.Tensor,     # [B*M, d_c] - True for valid edges
    ) -> torch.Tensor:
        """
        Args:
            edge_tokens: [B*M, d_c, D] edge messages (normalized)
            pad_mask: [B*M, d_c] True = valid, False = padding

        Returns:
            [B*M, d_c, D] updated edge tokens after extrinsic attention
        """
        BM, d_c, D = edge_tokens.shape

        # Project to Q, K, V
        Q = self.q_proj(edge_tokens).view(BM, d_c, self.H, self.dh)
        K = self.k_proj(edge_tokens).view(BM, d_c, self.H, self.dh)
        V = self.v_proj(edge_tokens).view(BM, d_c, self.H, self.dh)

        # Attention scores: [BM, H, d_c, d_c]
        Q = Q.permute(0, 2, 1, 3)  # [BM, H, d_c, dh]
        K = K.permute(0, 2, 1, 3)  # [BM, H, d_c, dh]
        V = V.permute(0, 2, 1, 3)  # [BM, H, d_c, dh]

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [BM, H, d_c, d_c]

        # === Extrinsic mask: exclude self (diagonal) ===
        diag_mask = torch.eye(d_c, dtype=torch.bool, device=edge_tokens.device)
        scores = scores.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), -1e4)  # fp16 safe

        # === Padding mask ===
        # Mask out attention to/from padded positions
        # pad_mask: [BM, d_c], True = valid
        # For keys: mask columns where key is padded
        key_mask = ~pad_mask.unsqueeze(1).unsqueeze(2)  # [BM, 1, 1, d_c]
        scores = scores.masked_fill(key_mask, -1e4)  # fp16 safe

        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(key_mask, 0.0)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, V)  # [BM, H, d_c, dh]
        out = out.permute(0, 2, 1, 3).reshape(BM, d_c, D)  # [BM, d_c, D]

        out = self.o_proj(out)
        return self.dropout(out)


# ============================================================
# Slot Responsibility Block
# ============================================================
class SlotResponsibilityBlock(nn.Module):
    """
    Slot responsibility mechanism using shared W_base.
    """
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
        # beta is constant - always use channel evidence Y
        beta = F.softplus(self.beta_0)
        tau = tau.view(-1, 1, 1, 1)
        beta = beta.view(-1, 1, 1, 1)
        return tau, beta

    def forward(
        self,
        VN: torch.Tensor,         # [B, K, N, D]
        Y: torch.Tensor,          # [B, N, Q]
        t: torch.Tensor,          # [B]
        W_base: torch.Tensor,     # [Q+1, D] shared embedding
    ) -> torch.Tensor:
        B, K, N, D = VN.shape
        Q = self.Q

        tau, beta = self._get_schedule(t)

        # === Step 1: Project VN to Q-dim using shared W ===
        vn_norm = self.vn_norm(VN)  # [B, K, N, D]
        score = torch.einsum('bknd,qd->bknq', vn_norm, W_base[:Q, :])

        # === Step 2: Add channel evidence ===
        score = score + beta * Y.unsqueeze(1)

        # === Step 3: Responsibility ===
        resp = F.softmax(score / tau, dim=1)

        # === Step 4: Assigned evidence using shared W ===
        y_weighted = resp * Y.unsqueeze(1)  # [B, K, N, Q]
        y_sym = torch.einsum('bknq,qd->bknd', y_weighted, W_base[:Q, :])
        y_assigned = y_sym

        # === Step 5: Trust-gated fusion ===
        gate_input = torch.cat([VN, y_assigned], dim=-1)
        g = self.trust_gate(gate_input)
        VN_tilde = g * VN + (1 - g) * y_assigned

        return VN_tilde


# ============================================================
# Neural MP Block (replaces cross-attention with edge self-attention)
# ============================================================
class NeuralMPBlock(nn.Module):
    """
    Neural Message Passing block using edge-based self-attention.

    Architecture:
    (1) m_{i→j} = W_msg @ h_i              [linear projection to message]
    (2) m̃_{(j,i)} = T_{H[j,i]} @ m        [exact GF normalize via factorized]
    (3) c_{(j,i)} = SparseAttn(m̃)         [extrinsic self-attn, batched]
    (4) e^z_{(j,i)} = MLP(c)               [non-linear energy head]
    (5) e^x_{j→i} = T^T_{H[j,i]} @ e^z    [exact GF denormalize via factorized]
    (6) h_i' = g * sum_j(e^x) + (1-g) * h_i [gated aggregation - Option B]

    """
    def __init__(self, D_model: int, num_heads: int, Q_field: int, dropout: float = 0.1, mlp_ratio: int = 4):
        super().__init__()
        self.D = D_model
        self.Q = Q_field

        # (1) VN normalization (no projection - attention Q/K/V handles it)
        self.vn_norm = nn.LayerNorm(D_model, elementwise_affine=False)

        # (3) Edge self-attention (extrinsic)
        self.edge_attn = EdgeSelfAttention(D_model, num_heads, dropout)
        self.edge_norm = nn.LayerNorm(D_model, elementwise_affine=False)

        # (4) Energy head MLP
        self.energy_head = nn.Sequential(
            nn.Linear(D_model, D_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model * mlp_ratio, D_model),
            nn.Dropout(dropout),
        )
        self.energy_norm = nn.LayerNorm(D_model, elementwise_affine=False)

        # (6) VN gated aggregation (Option B) with time-conditioned adaLN
        self.agg_norm = nn.LayerNorm(D_model, elementwise_affine=False)
        self.agg_gate = nn.Sequential(
            nn.Linear(2 * D_model, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
            nn.Sigmoid(),
        )

        # AdaLN for time conditioning (shift1/scale1 for vn_norm, shift2/scale2 for agg_norm)
        self.adaLN = nn.Linear(D_model, 4 * D_model)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

        self.dropout = nn.Dropout(dropout)

    def _apply_gf_perm(
        self,
        V: torch.Tensor,           # [B*M, d_c, D]
        coef: torch.Tensor,        # [M, d_c]
        U: torch.Tensor,           # [Q, D]
        perm_table: torch.Tensor,  # [Q, Q]
        transpose: bool = False,
    ) -> torch.Tensor:
        """
        Apply factorized GF permutation: V' = ((V @ U^T) permuted by P_a) @ U

        Args:
            V: [BM, d_c, D] input values
            coef: [M, d_c] coefficients (H[j,i] values)
            U: [Q, D] symbol embedding
            perm_table: [Q, Q] permutation table
            transpose: if True, apply P^T (denormalization)

        Returns:
            [BM, d_c, D] transformed values
        """
        BM, d_c, D = V.shape
        M = coef.shape[0]
        B = BM // M
        Q = U.shape[0]

        # Step 1: Project to symbol space
        V_sym = torch.einsum('bmd,qd->bmq', V, U)  # [BM, d_c, Q]

        # Step 2: Get permutation indices
        coef_safe = coef.clamp(min=0, max=Q-1)  # [M, d_c]
        perm_indices = perm_table[coef_safe]    # [M, d_c, Q]

        # Expand for batch dimension
        perm_expanded = perm_indices.unsqueeze(0).expand(B, -1, -1, -1)  # [B, M, d_c, Q]
        perm_expanded = perm_expanded.reshape(BM, d_c, Q)  # [BM, d_c, Q]

        # Step 3: Apply permutation
        if transpose:
            # Denormalization: scatter (inverse permutation)
            V_perm = torch.zeros_like(V_sym)
            V_perm.scatter_(-1, perm_expanded, V_sym)
        else:
            # Normalization: gather
            V_perm = torch.gather(V_sym, -1, perm_expanded)

        # Step 4: Project back to embedding space
        V_out = torch.einsum('bmq,qd->bmd', V_perm, U)  # [BM, d_c, D]

        return V_out

    def forward(
        self,
        vn: torch.Tensor,           # [B, N, D]
        t_emb: torch.Tensor,        # [B, D]
        cn_nbr_idx: torch.Tensor,   # [M, d_c] - which VNs each CN connects to
        cn_nbr_coef: torch.Tensor,  # [M, d_c] - H[j,i] coefficients
        vn_nbr_idx: torch.Tensor,   # [N, d_v] - which CNs each VN connects to
        vn_nbr_coef: torch.Tensor,  # [N, d_v] - H[j,i] coefficients
        U: torch.Tensor,            # [Q, D] - symbol embedding
        perm_table: torch.Tensor,   # [Q, Q] - GF permutation table
    ) -> torch.Tensor:
        """
        Returns updated VN embeddings [B, N, D].
        Note: No CN state needed - edge messages are computed on-the-fly.
        """
        B, N, D = vn.shape
        M, d_c = cn_nbr_idx.shape
        d_v = vn_nbr_idx.shape[1]

        # AdaLN modulation
        shift1, scale1, shift2, scale2 = self.adaLN(t_emb).chunk(4, dim=-1)

        # === (1) VN → gather edges (no projection, attention Q/K/V handles it) ===
        vn_norm = modulate(self.vn_norm(vn), shift1.unsqueeze(1), scale1.unsqueeze(1))

        # Gather messages for each edge (j, i)
        # cn_nbr_idx[j, k] = i means check j connects to VN i at position k
        idx_safe = cn_nbr_idx.clamp(min=0)  # [M, d_c]
        edge_msg = vn_norm[:, idx_safe, :]  # [B, M, d_c, D]
        edge_msg = edge_msg.reshape(B * M, d_c, D)  # [B*M, d_c, D]

        # Padding mask: True = valid edge
        pad_mask = (cn_nbr_idx >= 0)  # [M, d_c]
        pad_mask = pad_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * M, d_c)  # [B*M, d_c]

        # === (2) Exact GF normalize ===
        edge_msg_norm = self._apply_gf_perm(edge_msg, cn_nbr_coef, U, perm_table, transpose=False)

        # === (3) Edge self-attention (extrinsic) ===
        edge_out = self.edge_attn(edge_msg_norm, pad_mask)  # [B*M, d_c, D]
        edge_out = edge_msg_norm + edge_out  # residual
        edge_out = self.edge_norm(edge_out)

        # === (4) Energy head MLP ===
        energy_z = self.energy_head(edge_out)  # [B*M, d_c, D]
        energy_z = edge_out + energy_z  # residual
        energy_z = self.energy_norm(energy_z)

        # === (5) Exact GF denormalize ===
        energy_x = self._apply_gf_perm(energy_z, cn_nbr_coef, U, perm_table, transpose=True)
        energy_x = energy_x.reshape(B, M, d_c, D)

        # Zero out padded edges
        pad_mask_4d = pad_mask.reshape(B, M, d_c).unsqueeze(-1)  # [B, M, d_c, 1]
        energy_x = energy_x * pad_mask_4d.float()

        # === (6) VN aggregation (Option B: gated sum with sqrt normalization) ===
        # Vectorized scatter-add: cn_nbr_idx[j, k] = i means edge (j,k) goes to VN i
        # energy_x[b, j, k, :]: message from check j to VN i

        # Flatten edges
        edge_dst = cn_nbr_idx.reshape(-1)  # [M * d_c]
        edge_msg = energy_x.reshape(B, M * d_c, D)  # [B, M * d_c, D]

        # Create valid mask and clamp indices for scatter
        valid_mask = (edge_dst >= 0)  # [M * d_c]
        edge_dst_safe = edge_dst.clamp(min=0)  # [M * d_c], safe for indexing

        # Zero out invalid edges in messages
        valid_mask_exp = valid_mask.unsqueeze(0).unsqueeze(-1).float()  # [1, M*d_c, 1]
        edge_msg = edge_msg * valid_mask_exp  # [B, M * d_c, D]

        # Scatter add to VN positions
        vn_agg = torch.zeros(B, N, D, device=vn.device, dtype=vn.dtype)
        edge_dst_exp = edge_dst_safe.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)  # [B, M*d_c, D]
        vn_agg.scatter_add_(1, edge_dst_exp, edge_msg)

        # Compute degree per VN and normalize by sqrt(degree)
        degree = torch.zeros(N, device=vn.device, dtype=vn.dtype)
        degree.scatter_add_(0, edge_dst_safe, valid_mask.float())  # [N]
        degree = degree.clamp(min=1.0)  # avoid division by zero
        vn_agg = vn_agg / degree.sqrt().view(1, N, 1)  # [B, N, D]

        # Apply adaLN to vn_agg before gating (time-conditioned parity信頼度)
        vn_agg = modulate(self.agg_norm(vn_agg), shift2.unsqueeze(1), scale2.unsqueeze(1))

        # Gated fusion: g * msg_agg + (1-g) * vn
        gate_input = torch.cat([vn_agg, vn], dim=-1)  # [B, N, 2D]
        g = self.agg_gate(gate_input)  # [B, N, D]
        vn_out = g * vn_agg + (1 - g) * vn

        return vn_out


# ============================================================
# GRU Update Block
# ============================================================
class GRUUpdateBlock(nn.Module):
    """GRU-based belief update + AdaLN + FFN."""
    def __init__(self, D_model: int, dropout: float = 0.1, ffn_ratio: int = 2):
        super().__init__()
        self.D = D_model
        self.gru = nn.GRUCell(D_model, D_model)
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
        VN: torch.Tensor,        # [B, K, N, D]
        t_emb: torch.Tensor,     # [B, D]
    ) -> torch.Tensor:
        B, K, N, D = VN.shape
        VN_tilde_flat = VN_tilde.reshape(B * K * N, D)
        VN_flat = VN.reshape(B * K * N, D)
        VN_new_flat = self.gru(VN_tilde_flat, VN_flat)
        VN_new = VN_new_flat.reshape(B, K, N, D)

        shift, scale = self.adaLN(t_emb).chunk(2, dim=-1)
        VN_new = modulate(self.norm(VN_new), shift.view(B, 1, 1, D), scale.view(B, 1, 1, D))
        VN_new = VN_new + self.ffn(VN_new)
        return VN_new


# ============================================================
# Main Model with Neural MP
# ============================================================
class DiMP(nn.Module):
    """
    Key features:
    - W_base (U): [Q+1, D] - shared symbol embedding
    - T_a = U^T @ P_a @ U: exact GF permutation lifted to embedding space
    - Neural MP: edge self-attention with extrinsic mask (replaces cross-attention)
    - VN aggregation: gated sum (Option B)

    Architecture per layer:
    (1) SlotResponsibility: fuse VN with channel evidence Y
    (2) NeuralMP: edge-based message passing with exact GF permutations
    (3) GRU: belief update
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

        # === Shared symbol embedding (U in formulation) ===
        # Ensure contiguous for efficient access
        W_base_init = torch.empty(self.Q + 1, self.D)
        nn.init.orthogonal_(W_base_init)
        self.W_base = nn.Parameter(W_base_init.contiguous())

        # === GF permutation table (fixed, not learned) ===
        # perm_table[a, s] = a^{-1} * s in GF(Q)
        # NOTE: requires_grad=False automatically via register_buffer
        perm_table = build_gf_perm_table(self.Q)
        self.register_buffer('perm_table', perm_table)

        # === Slot initialization (deterministic orthogonal for slot differentiation) ===
        # Use fixed seed for reproducible orthogonal init
        rng_state = torch.get_rng_state()
        torch.manual_seed(0)
        slot_init = torch.zeros(self.K, self.D)
        nn.init.orthogonal_(slot_init)
        torch.set_rng_state(rng_state)  # restore RNG state
        self.slot_init = nn.Parameter(slot_init * slot_init_scale)

        # === Time embedding ===
        self.time_embed = SinusoidalPositionEmbedding(self.D)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.D, self.D * 4),
            nn.GELU(),
            nn.Linear(self.D * 4, self.D),
        )

        # === Slot responsibility blocks ===
        self.slot_blocks = nn.ModuleList([
            SlotResponsibilityBlock(Q=self.Q, D_model=self.D, dropout=dropout, tau_min=tau_min)
            for _ in range(self.num_layers)
        ])

        # === Neural MP blocks (2 per layer) ===
        self.mp_blocks = nn.ModuleList([
            nn.ModuleList([
                NeuralMPBlock(self.D, self.heads, self.Q, dropout=dropout, mlp_ratio=mlp_ratio)
                for _ in range(2)
            ])
            for _ in range(self.num_layers)
        ])

        # === GRU update blocks ===
        self.gru_blocks = nn.ModuleList([
            GRUUpdateBlock(self.D, dropout=dropout, ffn_ratio=2)
            for _ in range(self.num_layers)
        ])

        # === Output ===
        self.out_norm = nn.LayerNorm(self.D)
        self.output_proj = nn.Linear(self.D, self.Q)

        # === Tanner graph cache ===
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
            # Soft input: [B, K, N, Q] probabilities
            X_soft = X_t
            B, K, N, Q_dim = X_soft.shape
            x_sym = torch.einsum('bknq,qd->bknd', X_soft, self.W_base[:Q_dim, :])
        else:
            # Discrete input: [B, K, N] integers (MaskGIT-style)
            B, K, N = X_t.shape
            X_idx = X_t.long().clamp(0, self.Q)  # clamp to valid range [0, Q]
            x_sym = self.W_base[X_idx]  # direct lookup [B, K, N, D]

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

        # Get U (symbol embedding) for factorized permutation
        U = self.W_base[:self.Q]  # [Q, D]

        # Time embedding
        t = t.float().clamp(0.0, 1.0)
        t_emb = self.time_mlp(self.time_embed(t))
        t_emb_flat = t_emb.unsqueeze(1).expand(B, K, self.D).reshape(B * K, self.D)

        # === INITIALIZATION ===
        VN = self._build_initial_vn(X_t, soft_input)

        # === LAYERS ===
        for slot_block, mp_blocks_layer, gru_block in zip(
            self.slot_blocks, self.mp_blocks, self.gru_blocks
        ):
            # (1) Slot responsibility
            VN_tilde = slot_block(VN=VN, Y=Y_mag, t=t, W_base=self.W_base)

            # (2) Neural message passing (4 rounds per layer)
            VN_tilde_flat = VN_tilde.reshape(B * K, self.N, self.D)
            for mp_block in mp_blocks_layer:
                VN_tilde_flat = mp_block(
                    VN_tilde_flat, t_emb_flat,
                    self._cn_nbr_idx, self._cn_nbr_coef,
                    self._vn_nbr_idx, self._vn_nbr_coef,
                    U, self.perm_table,
                )
            VN_tilde = VN_tilde_flat.reshape(B, K, self.N, self.D)

            # (3) GRU belief update
            VN = gru_block(VN_tilde=VN_tilde, VN=VN, t_emb=t_emb)

        # === OUTPUT ===
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

        for slot_block, mp_blocks_layer, gru_block in zip(
            self.slot_blocks, self.mp_blocks, self.gru_blocks
        ):
            VN_tilde = slot_block(VN=VN, Y=Y_mag, t=t, W_base=self.W_base)
            VN_tilde_flat = VN_tilde.reshape(B * K, self.N, self.D)
            for mp_block in mp_blocks_layer:
                VN_tilde_flat = mp_block(
                    VN_tilde_flat, t_emb_flat,
                    self._cn_nbr_idx, self._cn_nbr_coef,
                    self._vn_nbr_idx, self._vn_nbr_coef,
                    U, self.perm_table,
                )
            VN_tilde = VN_tilde_flat.reshape(B, K, self.N, self.D)
            VN = gru_block(VN_tilde=VN_tilde, VN=VN, t_emb=t_emb)

        VN = self.out_norm(VN)
        hidden_states = VN.reshape(B, K * self.N, self.D)
        logits = self.output_proj(VN)

        return logits, hidden_states, t_emb

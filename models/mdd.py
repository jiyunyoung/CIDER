"""
DiT Backbone with Parity Embedding

Embeddings (concatenated):
- sym_emb: Symbol embedding of X_t
- pos_emb: Position embedding
- mag_emb: Magnitude from inner decoder Y
- parity_emb: Dynamic GF(q) contributions from revealed positions

Time conditioning: adaLN modulation (not concatenated)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.gf import get_gf


# ============================================================
# Sinusoidal Position Embedding
# ============================================================
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ============================================================
# adaLN Modulation
# ============================================================
def modulate(x, shift, scale):
    """Apply adaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


# ============================================================
# Transformer Block with adaLN
# ============================================================
class TransformerBlock(nn.Module):
    """Transformer block with adaLN for time conditioning."""
    def __init__(self, D_model, cond_dim, heads=4, dropout=0.1, mlp_ratio=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(D_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(D_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(D_model, elementwise_affine=False)

        self.ff = nn.Sequential(
            nn.Linear(D_model, D_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model * mlp_ratio, D_model),
            nn.Dropout(dropout)
        )

        # adaLN modulation: predicts shift1, scale1, shift2, scale2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 4 * D_model)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        """
        x: [B, L, D] tokens
        c: [B, D] time conditioning
        """
        # Get modulation parameters
        shift1, scale1, shift2, scale2 = self.adaLN_modulation(c).chunk(4, dim=-1)
        shift1 = shift1.unsqueeze(1)
        scale1 = scale1.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)

        # Self-attention with adaLN
        x_norm = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN with adaLN
        x_norm = modulate(self.norm2(x), shift2, scale2)
        x = x + self.ff(x_norm)

        return x


# ============================================================
# DDiT Final Layer with adaLN (For Adapter)
# ============================================================
class DDiTFinalLayer(nn.Module):
    def __init__(self, D_model, out_channels, cond_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(D_model, elementwise_affine=False)
        self.linear = nn.Linear(D_model, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * D_model)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.linear(x)


# ============================================================
# Magnitude Encoder
# ============================================================
class MagEncoder(nn.Module):
    def __init__(self, Q, D_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(Q, D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_model, D_model),
            nn.LayerNorm(D_model),
        )

    def forward(self, scores):
        """scores: [B, N, Q] -> [B, N, D]"""
        scores_norm = (scores - scores.mean(dim=-1, keepdim=True)) / (scores.std(dim=-1, keepdim=True) + 1e-6)
        return self.proj(scores_norm)


# ============================================================
# DiT Denoiser with Proper Parity Embedding
# ============================================================
class DiT(nn.Module):
    """
    DiT with parity embedding:
    - parity_emb: Dynamic GF(q) contributions from revealed positions
    - Time via adaLN (not concatenated)
    """
    def __init__(self,
                 Q: int,
                 N: int,
                 K: int,
                 M: int,
                 D_model: int = 256,
                 num_layers: int = 8,
                 heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.Q = Q
        self.N = N
        self.K = K
        self.M = M
        self.D = D_model
        self.MASK_TOKEN = Q

        # GF(q) tables (cached as buffers)
        gf = get_gf(Q)
        self.register_buffer('gf_exp_table', torch.tensor(gf.exp_table, dtype=torch.long))
        self.register_buffer('gf_log_table', torch.tensor(gf.log_table, dtype=torch.long))

        # Symbol embedding (+1 for MASK token)
        self.sym_embedding = nn.Embedding(Q + 1, D_model)

        # Position embedding
        self.pos_embedding = nn.Embedding(N, D_model)

        # Learnable slot-specific initialization tokens for symmetry breaking
        # These provide strong initial differentiation between slots
        self.slot_init = nn.Parameter(torch.randn(K, D_model) * 0.02)

        # Magnitude encoder
        self.mag_encoder = MagEncoder(Q, D_model, dropout=dropout)

        # Contribution embedding (for parity_emb: H[m,n] * X_t[n])
        self.contrib_embedding = nn.Embedding(Q, D_model)

        # Timestep embedding (for adaLN conditioning)
        self.time_embed = SinusoidalPositionEmbedding(D_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(D_model, D_model * 4),
            nn.GELU(),
            nn.Linear(D_model * 4, D_model)
        )

        # Concat fusion: 4*D -> D
        # sym, pos, mag, parity_emb are D each
        fusion_input_dim = 4 * D_model
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, 4 * D_model),
            nn.LayerNorm(4 * D_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * D_model, D_model)
        )

        # Transformer layers with adaLN
        self.layers = nn.ModuleList([
            TransformerBlock(D_model, D_model, heads, dropout, mlp_ratio)
            for _ in range(num_layers)
        ])

        # Output
        self.final_layer = DDiTFinalLayer(D_model, Q, D_model)

    def gf_multiply(self, a, b):
        """GF(q) multiplication using cached tables."""
        a_zero = (a == 0)
        b_zero = (b == 0)
        either_zero = a_zero | b_zero

        a_safe = a.clamp(min=1)
        b_safe = b.clamp(min=1)

        result = self.gf_exp_table[self.gf_log_table[a_safe] + self.gf_log_table[b_safe]]
        return result.masked_fill(either_zero, 0)

    def build_parity_emb(self, X_t, H, mask):
        """
        Build dynamic parity embedding from revealed positions.

        X_t: [B, K, N] current tokens
        H: [M, N] parity check matrix
        mask: [B, K, N] True = masked
        Returns: [B, K, N, D] per-position parity embedding
        """
        B, K, N = X_t.shape
        M = H.shape[0]

        # Replace MASK tokens (Q) with 0 for GF(q) operations
        # Masked positions will be zeroed out anyway by reveal_mask
        X_clean = X_t.clone()
        X_clean[X_clean >= self.Q] = 0

        # GF(q) multiply: contrib[m,n] = H[m,n] * X_t[n]
        H_exp = H.unsqueeze(0).unsqueeze(0)  # [1, 1, M, N]
        X_exp = X_clean.unsqueeze(2)          # [B, K, 1, N]

        contrib = self.gf_multiply(H_exp, X_exp)  # [B, K, M, N]

        # Zero out masked positions (unknown contribution)
        reveal_mask = (~mask).unsqueeze(2)  # [B, K, 1, N]
        contrib = contrib * reveal_mask.long()

        # Embed contributions
        contrib_emb = self.contrib_embedding(contrib)  # [B, K, M, N, D]

        # Mask non-connections in H
        H_mask = (H != 0).float().view(1, 1, M, N, 1)
        contrib_emb = contrib_emb * H_mask

        # Sum over M checks for each position n
        parity_emb = contrib_emb.sum(dim=2)  # [B, K, N, D]

        return parity_emb

    def forward(self,
                X_t: torch.Tensor,
                Y_mag: torch.Tensor,
                syn: torch.Tensor,
                t: torch.Tensor,
                H: torch.Tensor,
                mask: torch.Tensor = None,
                soft_input: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X_t: [B, K, N] hard symbols or [B, K, N, Q] soft beliefs
            Y_mag: [B, N, Q] soft scores from inner decoder
            syn: [B, K, M] syndrome (unused, kept for interface)
            t: [B] timestep (mask ratio)
            H: [M, N] parity-check matrix
            mask: [B, K, N] mask (True = masked). If None, inferred from X_t.
            soft_input: If True, X_t is soft beliefs [B, K, N, Q]

        Returns:
            logits: [B, K, N, Q]
        """
        if soft_input:
            B, K, N, Q = X_t.shape
            # Convert soft to hard for DiT (it doesn't support soft natively)
            X_t = X_t.argmax(dim=-1)
        else:
            B, K, N = X_t.shape
        device = X_t.device

        # Infer mask if not provided
        if mask is None:
            mask = (X_t == self.MASK_TOKEN)

        # 1. Time embedding for adaLN conditioning
        t_emb = self.time_mlp(self.time_embed(t))  # [B, D]

        # 2. Symbol embedding
        sym_emb = self.sym_embedding(X_t)  # [B, K, N, D]

        # 3. Position embedding
        pos_idx = torch.arange(N, device=device)
        pos_emb = self.pos_embedding(pos_idx)  # [N, D]
        pos_emb = pos_emb.view(1, 1, N, -1).expand(B, K, -1, -1)

        # 4. Magnitude embedding
        mag_emb = self.mag_encoder(Y_mag)  # [B, N, D]
        mag_emb = mag_emb.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, N, D]

        # 5. Dynamic parity embedding with (1-γt) gating
        # Emphasize structural cues late in denoising (when t is small, more revealed)
        parity_emb = self.build_parity_emb(X_t, H, mask)  # [B, K, N, D]
        parity_gate = (1 - t).view(B, 1, 1, 1)  # [B, 1, 1, 1]
        parity_emb = parity_emb * parity_gate

        # 6. Concat fusion (no time - it's via adaLN)
        tokens = self.fusion_proj(torch.cat([
            sym_emb,
            pos_emb,
            mag_emb,
            parity_emb
        ], dim=-1))  # [B, K, N, D]

        # 7. Add slot-specific initialization for symmetry breaking
        slot_init_broadcast = self.slot_init.view(1, K, 1, self.D).expand(B, -1, N, -1)
        tokens = tokens + slot_init_broadcast

        # 8. Flatten and transform
        tokens = tokens.view(B, K * N, -1)
        for layer in self.layers:
            tokens = layer(tokens, t_emb)

        # 9. Output
        logits = self.final_layer(tokens, t_emb)  # [B, K*N, Q]
        logits = logits.view(B, K, N, -1)

        return logits

    def get_hidden_states(self,
                          X_t: torch.Tensor,
                          Y_mag: torch.Tensor,
                          syn: torch.Tensor,
                          t: torch.Tensor,
                          H: torch.Tensor,
                          mask: torch.Tensor = None):
        """Forward pass that also returns hidden states for adapter training."""
        B, K, N = X_t.shape
        device = X_t.device

        if mask is None:
            mask = (X_t == self.MASK_TOKEN)

        t_emb = self.time_mlp(self.time_embed(t))

        sym_emb = self.sym_embedding(X_t)
        pos_emb = self.pos_embedding(torch.arange(N, device=device)).view(1, 1, N, -1).expand(B, K, -1, -1)
        mag_emb = self.mag_encoder(Y_mag).unsqueeze(1).expand(-1, K, -1, -1)
        parity_emb = self.build_parity_emb(X_t, H, mask)
        parity_gate = (1 - t).view(B, 1, 1, 1)
        parity_emb = parity_emb * parity_gate

        tokens = self.fusion_proj(torch.cat([sym_emb, pos_emb, mag_emb, parity_emb], dim=-1))
        # Add slot-specific initialization for symmetry breaking
        slot_init_broadcast = self.slot_init.view(1, K, 1, self.D).expand(B, -1, N, -1)
        tokens = tokens + slot_init_broadcast
        tokens = tokens.view(B, K * N, -1)

        for layer in self.layers:
            tokens = layer(tokens, t_emb)

        hidden_states = tokens  # [B, K*N, D]
        logits = self.final_layer(tokens, t_emb)

        return logits.view(B, K, N, self.Q), hidden_states, t_emb

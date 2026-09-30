"""
CIDER-Iterative-V2 (ablation A2): strong non-diffusion iterative refinement.

Answers the reviewer request for "an ablation comparing masked diffusion against
a stronger non-diffusion iterative refinement model using the same demixing and
parity-aware modules".

The backbone is models.cider.DiMP *verbatim*: identical Module A (slot
responsibility / demixing), identical Module B (parity-aware neural MP),
identical depth, width, heads and parameter count. Only the inference state
differs:

    CIDER    X_t is a partially-masked token sequence, t is the diffusion time,
             and positions are progressively committed by a mask schedule.
    This     X_t is the previous iteration's soft posterior, t is the (normalised)
             iteration index, and every position is re-predicted at every
             iteration. Nothing is ever masked or committed.

Iteration 0 feeds the MASK embedding W_base[Q] as the "no information yet" input,
which is the same tensor CIDER's first denoiser call sees. There is no mask
schedule, no stochastic masking, and no partial commitment at any point.

What makes this *stronger* than models/cider_iterative.py (ablation A1), which
is built from the unconditioned cider_direct blocks and is therefore strictly
smaller than CIDER:

  1. Iteration conditioning via adaLN. A1 has none, so it cannot behave
     differently at iteration 1 vs iteration 8; CIDER gets this for free from
     its timestep embedding. Here t = 1 - i/(T-1) reuses CIDER's own learned
     cosine schedule (soft responsibility early, sharp late).
  2. Exact parameter and NFE matching with CIDER (same backbone object; the
     only extra weights are ~1k damping-MLP parameters).
  3. Learned damping on the logit trajectory, so soft feedback cannot oscillate:
     z_t = z_{t-1} + alpha(t) * (raw_t - z_{t-1}), alpha initialised near 1.
  4. Deep supervision across iterations (unroll_mode='full'), with the Hungarian
     assignment computed once at the final iterate and reused at every
     iteration so the slot-to-user assignment cannot flip mid-trajectory.
  5. Optional GRU hidden-state feedback (feedback='hidden'), which carries a
     D-dimensional state between iterations. Note that this is a *wider*
     inter-step channel than CIDER itself has: masked diffusion only carries a
     discrete token sequence between steps. feedback='soft' (default) carries
     the Q-simplex posterior, which is already wider than CIDER's channel.
  6. num_iters is overridable at eval time, so T can be swept at test time
     without retraining.

Training budget. unroll_mode='sampled' (default) mirrors CIDER exactly: one
gradient-bearing denoiser call per optimiser step, at a uniformly sampled
iteration index, with the preceding iterations run under no_grad. This keeps
the comparison budget-matched. unroll_mode='full' backprops through all T
unrolls with deep supervision -- more compute than CIDER gets, i.e. it favours
the baseline -- and is practical for small T.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from scipy.optimize import linear_sum_assignment
from typing import Optional, Tuple

from models.cider import DiMP


# ============================================================
# Vectorised permutation-invariant helpers
# ============================================================
def _ce_cost_matrix(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """cost[b,i,j] = sum_n -log p(gt[b,j,n] | pred row i). Shapes [B,K,N,Q], [B,K,N]."""
    B, K, N, Q = logits.shape
    logp = F.log_softmax(logits.float(), dim=-1)
    idx = gt.long().unsqueeze(1).expand(B, K, K, N).unsqueeze(-1)
    gathered = torch.gather(logp.unsqueeze(2).expand(B, K, K, N, Q), 4, idx).squeeze(-1)
    return -gathered.sum(-1)


def _hamming_cost_matrix(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """cost[b,i,j] = #{n : pred[b,i,n] != gt[b,j,n]}. Shapes [B,K,N], [B,K,N]."""
    return (pred.unsqueeze(2) != gt.unsqueeze(1)).sum(-1).float()


def _assign(cost: torch.Tensor) -> torch.Tensor:
    """Hungarian per batch element. cost [B,K,K] -> col_ind [B,K] (gt index per pred row)."""
    cost_np = cost.detach().cpu().numpy()
    cols = np.stack([linear_sum_assignment(cost_np[b])[1] for b in range(cost_np.shape[0])])
    return torch.as_tensor(cols, device=cost.device, dtype=torch.long)


# ============================================================
# A2 model
# ============================================================
class DiMPIterativeV2(L.LightningModule):
    """Iteration-conditioned, damped, deep-supervised refinement over CIDER's backbone."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.Q = int(config.data.Q)
        self.N = int(config.data.N)
        self.K = int(config.data.get('K', config.data.get('K_max', 2)))
        self.M = int(config.data.M)

        model_cfg = config.model
        self.D = int(model_cfg.D_model)
        # Default T to CIDER's inference_steps (injected by the size config) so the
        # NFE count matches the diffusion model it is being compared against.
        num_iters = model_cfg.get('num_iters', None)
        if num_iters is None:
            num_iters = model_cfg.get('inference_steps', 20)
        self.num_iters = int(num_iters)
        self.feedback = str(model_cfg.get('feedback', 'soft'))
        self.unroll_mode = str(model_cfg.get('unroll_mode', 'sampled'))
        self.deep_supervision = str(model_cfg.get('deep_supervision', 'linear'))

        if self.feedback not in ('soft', 'hard', 'hidden'):
            raise ValueError(f"feedback must be 'soft'/'hard'/'hidden', got {self.feedback!r}")
        if self.unroll_mode not in ('full', 'sampled'):
            raise ValueError(f"unroll_mode must be 'full' or 'sampled', got {self.unroll_mode!r}")
        if self.deep_supervision not in ('linear', 'uniform', 'final'):
            raise ValueError(f"unknown deep_supervision: {self.deep_supervision!r}")

        # === CIDER backbone, unmodified ===
        self.backbone = DiMP(
            Q=self.Q, N=self.N, K=self.K, M=self.M,
            D_model=self.D,
            num_layers=int(model_cfg.num_layers),
            heads=int(model_cfg.heads),
            mlp_ratio=int(model_cfg.get('mlp_ratio', 4)),
            dropout=float(model_cfg.get('dropout', 0.1)),
            tau_min=float(model_cfg.get('tau_min', 0.2)),
            slot_init_scale=float(model_cfg.get('slot_init_scale', 0.5)),
            mp_per_layer=int(model_cfg.get('mp_per_layer', 2)),
        )

        # === Learned damping alpha(t) in (0,1), init ~0.9 (near-undamped) ===
        # A function of continuous t rather than a per-index table, so a test-time
        # sweep over T stays well defined.
        self.damp_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 1),
        )
        nn.init.zeros_(self.damp_mlp[-1].weight)
        nn.init.constant_(self.damp_mlp[-1].bias, 2.197)  # sigmoid(2.197) ~= 0.9

        # === Optional hidden-state feedback ===
        if self.feedback == 'hidden':
            self.state_gru = nn.GRUCell(self.D, self.D)

        # Optimiser settings are read from the SAME keys diffusion.py uses, so a
        # single `optim.lr=...` override configures CIDER and this ablation
        # identically. Falls back to the baseline-style keys if optim is absent.
        self.config = config
        self.register_buffer("H_matrix", None, persistent=False)

    # --------------------------------------------------------
    def set_H_matrix(self, H: torch.Tensor):
        self.H_matrix = H
        self.backbone._build_tanner_cache(H)

    # --------------------------------------------------------
    def _t_for_iter(self, i: int, T: int, B: int, device) -> torch.Tensor:
        """Iteration index -> diffusion-style time in [0,1]: t=1 at i=0 (no info), t=0 at i=T-1."""
        frac = 0.0 if T <= 1 else i / (T - 1)
        return torch.full((B,), 1.0 - frac, device=device, dtype=torch.float32)

    def _step_once(
        self,
        Y: torch.Tensor,
        t: torch.Tensor,
        prev_logits: Optional[torch.Tensor],
        prev_vn: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One refinement pass. Returns (damped logits, VN state)."""
        B = Y.shape[0]
        H = self.H_matrix

        if prev_logits is None:
            # Iteration 0: the "nothing known yet" input, i.e. the MASK embedding.
            X_in = torch.full((B, self.K, self.N), self.backbone.MASK_TOKEN,
                              device=Y.device, dtype=torch.long)
            soft_input = False
            vn_init = None
        elif self.feedback == 'hard':
            # Discrete token feedback: the input distribution the CIDER backbone is
            # actually trained on when use_soft_input=false. Note argmax blocks the
            # inter-iteration gradient, so this is meant for unroll_mode='sampled'
            # (where earlier iterations are no_grad anyway); under 'full' it makes
            # the unroll effectively 1-step for BPTT purposes.
            X_in = prev_logits.argmax(dim=-1)
            soft_input = False
            vn_init = None
        else:
            X_in = F.softmax(prev_logits, dim=-1)
            soft_input = True
            vn_init = None
            if self.feedback == 'hidden' and prev_vn is not None:
                cand = self.backbone._build_initial_vn(X_in, soft_input=True)
                vn_init = self.state_gru(
                    cand.reshape(-1, self.D), prev_vn.reshape(-1, self.D)
                ).view_as(cand)

        raw, vn = self.backbone(
            X_in, Y, None, t, H, soft_input=soft_input,
            vn_init=vn_init, return_vn=True,
        )

        if prev_logits is None:
            return raw, vn

        alpha = torch.sigmoid(self.damp_mlp(t.view(-1, 1))).view(-1, 1, 1, 1)
        return prev_logits + alpha * (raw - prev_logits), vn

    # --------------------------------------------------------
    def forward(self, Y: torch.Tensor, num_iters: Optional[int] = None,
                return_all: bool = False):
        """
        Args:
            Y: [B, N, Q] channel evidence
            num_iters: override the trained T (test-time sweep)
            return_all: also return the per-iteration logits
        Returns:
            logits [B, K, N, Q], or (logits, [logits_0 .. logits_{T-1}])
        """
        assert self.H_matrix is not None, "H matrix not set. Call set_H_matrix() first."
        T = num_iters if num_iters is not None else self.num_iters

        logits, vn, traj = None, None, []
        for i in range(T):
            t = self._t_for_iter(i, T, Y.shape[0], Y.device)
            logits, vn = self._step_once(Y, t, logits, vn)
            if return_all:
                traj.append(logits)

        return (logits, traj) if return_all else logits

    # --------------------------------------------------------
    def _supervision_weights(self, T: int, device) -> torch.Tensor:
        if self.deep_supervision == 'final':
            w = torch.zeros(T, device=device)
            w[-1] = 1.0
            return w
        if self.deep_supervision == 'uniform':
            w = torch.ones(T, device=device)
        else:  # linear: later iterations matter more
            w = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        return w / w.sum()

    def _matched_loss(self, logits: torch.Tensor, gt: torch.Tensor,
                      col: torch.Tensor) -> torch.Tensor:
        """CE under a fixed slot->user assignment."""
        B, K, N, Q = logits.shape
        gt_perm = torch.gather(gt.long(), 1, col.unsqueeze(-1).expand(B, K, N))
        return F.cross_entropy(logits.reshape(B * K * N, Q), gt_perm.reshape(-1))

    def _compute_loss(self, Y: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Deep-supervised (full) or single-sampled-iteration (sampled) loss."""
        T = self.num_iters

        if self.unroll_mode == 'sampled':
            # Budget-matched with CIDER: one gradient-bearing pass per step.
            i_star = int(torch.randint(0, T, (1,)).item())
            logits, vn = None, None
            with torch.no_grad():
                for i in range(i_star):
                    t = self._t_for_iter(i, T, Y.shape[0], Y.device)
                    logits, vn = self._step_once(Y, t, logits, vn)
            t = self._t_for_iter(i_star, T, Y.shape[0], Y.device)
            logits, _ = self._step_once(Y, t, logits, vn)
            col = _assign(_ce_cost_matrix(logits, gt))
            return self._matched_loss(logits, gt, col)

        _, traj = self.forward(Y, return_all=True)
        # One assignment, taken at the final iterate and reused everywhere, so the
        # slot->user correspondence cannot flip mid-trajectory.
        col = _assign(_ce_cost_matrix(traj[-1], gt))
        w = self._supervision_weights(len(traj), Y.device)
        return sum(w[i] * self._matched_loss(traj[i], gt, col) for i in range(len(traj)))

    # --------------------------------------------------------
    @torch.no_grad()
    def compute_accuracy(self, logits: torch.Tensor, gt: torch.Tensor) -> dict:
        pred = logits.argmax(dim=-1)
        col = _assign(_hamming_cost_matrix(pred, gt.long()))
        B, K, N = pred.shape
        gt_perm = torch.gather(gt.long(), 1, col.unsqueeze(-1).expand(B, K, N))
        matches = (pred == gt_perm)
        return {
            'symbol_acc': matches.float().mean().item(),
            'codeword_acc': matches.all(dim=-1).float().mean().item(),
        }

    # --------------------------------------------------------
    def training_step(self, batch, batch_idx):
        Y, gt = batch
        loss = self._compute_loss(Y, gt)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def _eval_step(self, batch, stage: str):
        Y, gt = batch
        logits = self.forward(Y)
        col = _assign(_ce_cost_matrix(logits, gt))
        loss = self._matched_loss(logits, gt, col)
        metrics = self.compute_accuracy(logits, gt)

        self.log(f'{stage}/loss', loss, prog_bar=(stage == 'val'), sync_dist=True)
        self.log(f'{stage}/symbol_acc', metrics['symbol_acc'], prog_bar=(stage == 'val'), sync_dist=True)
        self.log(f'{stage}/codeword_acc', metrics['codeword_acc'], sync_dist=True)
        if stage == 'val':
            self.log('val/accuracy', metrics['symbol_acc'], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, 'test')

    # --------------------------------------------------------
    def configure_optimizers(self):
        """Byte-for-byte the same recipe as diffusion.py: AdamW + linear warmup
        into cosine decay, reading the same config keys. Only the model differs."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        cfg = self.config
        optim_cfg = cfg.get('optim', None)
        if optim_cfg is not None:
            lr = optim_cfg.get('lr', 1e-4)
            weight_decay = optim_cfg.get('weight_decay', 0.01)
            beta1 = optim_cfg.get('beta1', 0.9)
            beta2 = optim_cfg.get('beta2', 0.99)
            eps = optim_cfg.get('eps', 1e-8)
        else:
            lr = cfg.training.get('learning_rate', 1e-3)
            weight_decay = cfg.training.optimizer.get('weight_decay', 0.01)
            beta1, beta2, eps = 0.9, 0.99, 1e-8

        optimizer = AdamW(self.parameters(), lr=lr, betas=(beta1, beta2),
                          eps=eps, weight_decay=weight_decay)

        warmup_epochs = cfg.training.get('warmup_epochs', 10)
        num_epochs = cfg.training.get('num_epochs', 100)
        min_lr_ratio = cfg.training.get('min_lr_ratio', 0.01)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs),
                                  eta_min=lr * min_lr_ratio),
            ],
            milestones=[warmup_epochs],
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch',
                             'frequency': 1, 'name': 'trainer/lr'},
        }

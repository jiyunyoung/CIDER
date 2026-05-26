"""
Exponential Moving Average (EMA) for model weights.

This version is SAFE for:
- PyTorch Lightning
- DDP / FSDP / ZeRO
- Models with complex parameter structures
- Adapters / LoRA
- Any reordering of parameters

Instead of relying on parameter order (unsafe),
EMA uses `named_parameters()` for stable matching.
"""

import torch
import copy


class ExponentialMovingAverage:
    """
    Maintains an exponential moving average of model parameters.

    Usage:
        ema = ExponentialMovingAverage(model, decay=0.9999)

        # During training loop, AFTER optimizer.step():
        ema.update(model)

        # For validation:
        ema.store(model)
        ema.copy_to(model)
        ... validation ...
        ema.restore(model)
    """

    def __init__(self, model, decay=0.9999, use_num_updates=True):
        """
        Args:
            model: nn.Module to track
            decay: EMA decay factor (0.999 - 0.9999 recommended)
            use_num_updates: if True, warmup EMA decay early in training
        """
        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay must be in [0,1]")

        self.decay = decay
        self.use_num_updates = use_num_updates
        self.num_updates = 0

        # Initialize shadow parameters by name (SAFE)
        self.shadow_params = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        # Buffer for storing original parameters during evaluation
        self.backup_params = {}

    # ------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------

    def to(self, device):
        """Move EMA shadow params to a device."""
        for name in self.shadow_params:
            self.shadow_params[name] = self.shadow_params[name].to(device)

    # ------------------------------------------------------------
    #  EMA Update
    # ------------------------------------------------------------

    @torch.no_grad()
    def update(self, model):
        """
        Update EMA weights using:
            shadow = decay * shadow + (1 - decay) * weight

        Should be called AFTER optimizer.step() each iteration.
        """

        # Warmup logic (same as Stable Diffusion, ADM, DiT)
        if self.use_num_updates:
            self.num_updates += 1
            decay = min(self.decay,
                        (1 + self.num_updates) / (10 + self.num_updates))
        else:
            decay = self.decay

        one_minus_decay = 1.0 - decay

        # Update per parameter by NAME (SAFE)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            shadow = self.shadow_params[name]

            # shadow = decay * shadow + (1-decay) * param
            shadow.sub_(one_minus_decay * (shadow - param))

    # ------------------------------------------------------------
    #  Copy EMA → Model (for evaluation)
    # ------------------------------------------------------------

    @torch.no_grad()
    def copy_to(self, model):
        """
        Load EMA parameters into the model.
        Use together with store()/restore().
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow_params[name].data)

    # ------------------------------------------------------------
    #  Store & Restore (swap in/out EMA)
    # ------------------------------------------------------------

    def store(self, model):
        """
        Save current model parameters before loading EMA params.
        """
        self.backup_params = {
            name: p.clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def restore(self, model):
        """
        Restore parameters saved with store().
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup_params[name].data)

    # ------------------------------------------------------------
    #  State Dict Support
    # ------------------------------------------------------------

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params=copy.deepcopy(self.shadow_params),
        )

    def load_state_dict(self, state_dict):
        """Load EMA state."""
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = copy.deepcopy(state_dict["shadow_params"])


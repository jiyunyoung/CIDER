"""
Model checkpointing utilities for saving and loading training state.
"""

import torch
import os
from pathlib import Path


class CheckpointManager:
    """Manage model checkpoints during training."""

    def __init__(self, checkpoint_dir='models/', keep_best=True, keep_last_n=3):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best: Whether to keep the best model (lowest val loss)
            keep_last_n: Number of last checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.keep_last_n = keep_last_n
        self.best_val_loss = float('inf')
        self.last_checkpoints = []

    def save(self, state, val_loss=None, epoch=None, is_best=False):
        """
        Save checkpoint.

        Args:
            state: Dict with keys 'epoch', 'model_state_dict', 'optimizer_state_dict',
                   'train_losses', 'val_losses'
            val_loss: Validation loss (for best model tracking)
            epoch: Epoch number
            is_best: Whether this is the best model so far
        """
        if epoch is None:
            epoch = state.get('epoch', 0)

        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save(state, checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_path}")

        # Track last checkpoints for cleanup
        self.last_checkpoints.append(checkpoint_path)
        if len(self.last_checkpoints) > self.keep_last_n:
            old_ckpt = self.last_checkpoints.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()
                print(f"  Removed old checkpoint: {old_ckpt.name}")

        # Best checkpoint
        if self.keep_best and val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self.checkpoint_dir / 'best_model.pt'
                torch.save(state, best_path)
                print(f"✓ New best model! Val loss: {val_loss:.4f} → {best_path.name}")
                is_best = True

        return is_best

    def load(self, checkpoint_path, device='cuda'):
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load to

        Returns:
            state: Loaded state dict
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
        return state

    def load_best(self, device='cuda'):
        """Load the best model."""
        best_path = self.checkpoint_dir / 'best_model.pt'
        if not best_path.exists():
            raise FileNotFoundError("No best model saved yet")
        return self.load(best_path, device)

    def load_latest(self, device='cuda'):
        """Load the latest checkpoint."""
        if not self.last_checkpoints:
            raise FileNotFoundError("No checkpoints saved")
        return self.load(self.last_checkpoints[-1], device)

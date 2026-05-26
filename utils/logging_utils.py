"""
Logging utilities with optional WandB integration.
Inspired by MDMECC_SNU's approach but simplified.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any


class WandBLogger:
    """Simple WandB logger wrapper."""

    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Initialize WandB logger.

        Args:
            config: wandB config dict with keys: project, entity, notes, tags, etc.
            enabled: Whether to actually log to wandB
        """
        self.enabled = enabled and config.get('enabled', True)
        self.config = config

        if self.enabled:
            try:
                import wandb
                self.wandb = wandb

                # Initialize wandB run
                self.run = wandb.init(
                    project=config.get('project', 'slot-attention-demixing'),
                    entity=config.get('entity', None),
                    notes=config.get('notes', ''),
                    tags=config.get('tags', []),
                    name=config.get('name', None),
                )
                print(f"✓ WandB initialized: {self.run.url if self.run else 'offline'}")
            except ImportError:
                print("⚠️  WandB not installed. Install with: pip install wandb")
                self.enabled = False
                self.wandb = None
                self.run = None

    def log(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to wandB."""
        if not self.enabled or self.run is None:
            return

        if step is not None:
            self.wandb.log(metrics, step=step)
        else:
            self.wandb.log(metrics)

    def log_config(self, config: Dict[str, Any]):
        """Log config to wandB."""
        if not self.enabled or self.run is None:
            return
        self.wandb.config.update(config)

    def finish(self):
        """Finish wandB run."""
        if self.enabled and self.run is not None:
            self.wandb.finish()


class GPUMonitor:
    """Simple GPU monitor (no psutil dependency)."""

    @staticmethod
    def print_gpu_stats():
        """Print current GPU statistics."""
        if not torch.cuda.is_available():
            print("No CUDA GPU available")
            return

        print("\n" + "="*70)
        print("GPU Status")
        print("="*70)

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")

            # Current memory usage
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Free: {(props.total_memory / 1e9) - reserved:.2f} GB")

    @staticmethod
    def get_gpu_memory_usage(device=0):
        """Get current GPU memory usage in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(device) / 1e9

    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ GPU cache cleared")

    @staticmethod
    def optimize_gpu_settings():
        """Set GPU optimization flags."""
        if not torch.cuda.is_available():
            return

        # Enable cuDNN auto-tuner for better performance
        torch.backends.cudnn.benchmark = True
        print("✓ Enabled cuDNN benchmark mode")

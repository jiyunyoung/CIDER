"""
GPU monitoring and optimization utilities.
"""

import torch
import psutil


class GPUMonitor:
    """Monitor GPU memory and compute usage."""

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
    def print_cpu_stats():
        """Print current CPU statistics."""
        print("\n" + "="*70)
        print("CPU Status")
        print("="*70)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        print(f"CPU Usage: {cpu_percent}%")
        print(f"Memory: {memory.used / 1e9:.2f} GB / {memory.total / 1e9:.2f} GB ({memory.percent}%)")

    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ GPU cache cleared")

    @staticmethod
    def get_gpu_memory_usage(device=0):
        """Get current GPU memory usage in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(device) / 1e9

    @staticmethod
    def optimize_gpu_settings():
        """Set GPU optimization flags."""
        if not torch.cuda.is_available():
            return

        # Enable cuDNN auto-tuner for better performance
        torch.backends.cudnn.benchmark = True
        print("✓ Enabled cuDNN benchmark mode")

        # Use reduced precision where possible
        # torch.set_float32_matmul_precision('high')  # For TF32 on A100
        print("✓ GPU optimization settings applied")

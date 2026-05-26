"""
Unified data interface for Slot Attention experiments.

Provides a single function to get dataloaders for any dataset,
making it easy to swap datasets via configuration.
"""

import os
from torch.utils.data import DataLoader
from omegaconf import DictConfig


def get_dataloaders(config: DictConfig):
    """
    Get train/val/test dataloaders based on config.

    This is the unified data interface - add new datasets here
    without modifying training code.

    Args:
        config: Hydra configuration object

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader (optional, can be None)
    """
    dataset_name = config.data.name

    # On-the-fly Q-ary datasets (mixed Eb/N0)
    if dataset_name.startswith('onthefly'):
        return _get_onthefly_dataloaders(config)

    # End-to-end datasets (with inner decoder soft outputs Y)
    # Accept e2e_*, tiny_*, small_*, moderate_*, large_* naming conventions
    e2e_prefixes = ('e2e', 'tiny', 'small', 'moderate', 'large')
    if dataset_name.startswith(e2e_prefixes):
        return _get_e2e_dataloaders(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def _get_e2e_dataloaders(config: DictConfig):
    """
    Get dataloaders for end-to-end demixing (with inner decoder outputs).

    Data format:
        Y: [B, N, Q] - soft scores from inner decoder
        gt_codewords: [B, K, N] - ground truth codewords

    Args:
        config: Hydra configuration

    Returns:
        train_loader, val_loader, test_loader
    """
    from data.data import E2EDataset

    # Extract config
    data_dir = os.path.expanduser(config.data_dir)
    batch_size = config.training.batch_size
    num_workers = config.training.num_workers
    pin_memory = config.training.get('pin_memory', True)
    persistent_workers = config.training.get('persistent_workers', True) if num_workers > 0 else False

    # Check mode - only load required splits
    mode = config.get('mode', 'train')
    eval_split = config.get('eval_split', 'test')

    # Create datasets based on mode
    train_dataset, val_dataset, test_dataset = None, None, None

    if mode in ('train', 'prism_adapter', 'prism_head'):
        # Training modes need train and val
        train_dataset = E2EDataset(data_dir, split='train', device='cpu')
        val_dataset = E2EDataset(data_dir, split='val', device='cpu')
        print(f"✓ Loaded E2E train dataset: {len(train_dataset)} samples")
        print(f"✓ Loaded E2E val dataset: {len(val_dataset)} samples")
        ref_dataset = train_dataset
    else:
        # Test/eval mode - only load the required split
        if eval_split == 'train':
            train_dataset = E2EDataset(data_dir, split='train', device='cpu')
            print(f"✓ Loaded E2E train dataset: {len(train_dataset)} samples")
            ref_dataset = train_dataset
        elif eval_split == 'val':
            val_dataset = E2EDataset(data_dir, split='val', device='cpu')
            print(f"✓ Loaded E2E val dataset: {len(val_dataset)} samples")
            ref_dataset = val_dataset
        else:
            test_dataset = E2EDataset(data_dir, split='test', device='cpu')
            print(f"✓ Loaded E2E test dataset: {len(test_dataset)} samples")
            ref_dataset = test_dataset

    print(f"  Y shape: [N={ref_dataset.Y.shape[1]}, Q={ref_dataset.Y.shape[2]}]")
    print(f"  X0 shape: [K={ref_dataset.gt_codewords.shape[1]}, N={ref_dataset.gt_codewords.shape[2]}]")
    if ref_dataset.H_matrix is not None:
        print(f"  H_matrix shape: {ref_dataset.H_matrix.shape}")

    # Create dataloaders (only for loaded datasets)
    train_loader = None
    val_loader = None
    test_loader = None

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )

    print(f"✓ Created E2E dataloaders (batch_size={batch_size}, num_workers={num_workers})")

    return train_loader, val_loader, test_loader


def _get_onthefly_dataloaders(config: DictConfig):
    """
    Get dataloaders for Q-ary LDPC with on-the-fly generation and mixed Eb/N0.
    """
    from data.data_onthefly import QaryOnTheFlyDataset
    import os

    batch_size = config.training.batch_size
    num_workers = config.training.num_workers
    pin_memory = config.training.get('pin_memory', True)
    persistent_workers = config.training.get('persistent_workers', True) if num_workers > 0 else False

    data_dir = os.path.expanduser(config.data_dir)
    h_matrix_file = config.data.get('h_matrix_file', 'H_matrix.pt')
    h_matrix_path = os.path.join(data_dir, h_matrix_file)

    K = config.data.get('K_max', 2)
    Eb_dB = config.data.get('Eb_dB', 10.0)
    n_s = config.data.get('n_s', 24)
    sigma2 = config.data.get('sigma2', 1.0)
    Eb_min = config.data.get('Eb_min', None)
    Eb_max = config.data.get('Eb_max', None)
    Eb_range = (Eb_min, Eb_max) if Eb_min is not None and Eb_max is not None else None

    mode = config.get('mode', 'train')
    train_loader, val_loader, test_loader = None, None, None

    if mode in ('train', 'prism_adapter', 'prism_head'):
        train_dataset = QaryOnTheFlyDataset(
            h_matrix_path, K=K, Eb_dB=Eb_dB, n_s=n_s, sigma2=sigma2,
            num_samples=70000, fixed_seed=None, Eb_range=Eb_range)
        # Val uses fixed Eb/N0 (midpoint) for consistent evaluation
        val_Eb = (Eb_min + Eb_max) / 2 if Eb_range else Eb_dB
        val_dataset = QaryOnTheFlyDataset(
            h_matrix_path, K=K, Eb_dB=val_Eb, n_s=n_s, sigma2=sigma2,
            num_samples=5000, fixed_seed=99999, Eb_range=None)

        print(f"✓ On-the-fly train dataset: {len(train_dataset)} samples/epoch")
        if Eb_range:
            print(f"  Eb/N0 range: [{Eb_min}, {Eb_max}] dB (mixed)")
        else:
            print(f"  Eb/N0: {Eb_dB} dB (fixed)")
        print(f"  Q={train_dataset.Q}, L={train_dataset.L}, K={K}, n_s={n_s}")
        print(f"✓ On-the-fly val dataset: {len(val_dataset)} samples (Eb/N0={val_Eb}dB, fixed seed)")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, persistent_workers=persistent_workers,
            pin_memory=pin_memory)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, persistent_workers=persistent_workers,
            pin_memory=pin_memory)
    else:
        num_test = config.data.get('num_test_samples', 5000)
        test_dataset = QaryOnTheFlyDataset(
            h_matrix_path, K=K, Eb_dB=Eb_dB, n_s=n_s, sigma2=sigma2,
            num_samples=num_test, fixed_seed=199999, Eb_range=None)
        print(f"✓ On-the-fly test dataset: {len(test_dataset)} samples (Eb/N0={Eb_dB}dB)")

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, persistent_workers=persistent_workers,
            pin_memory=pin_memory)

    print(f"✓ Created on-the-fly dataloaders (batch_size={batch_size})")
    return train_loader, val_loader, test_loader

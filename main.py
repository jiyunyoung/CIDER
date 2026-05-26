#!/usr/bin/env python3
"""
Main entry point for experiments.
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import WandbLogger

from diffusion import Diffusion
from models import MLP, CNN, GNN, Transformer, MPA, NBP, CIDER_direct, CIDER_GRU_direct, CIDER_iterative
from dataloader import get_dataloaders

# Baseline models (non-diffusion, one-shot prediction)
BASELINE_MODELS = {'mlp':MLP, 'cnn':CNN, 'transformer':Transformer, 'gnn':GNN, 'nbp':NBP, 'mpa':MPA, 'cider_direct':CIDER_direct, 'cider_gru_direct':CIDER_GRU_direct, 'cider_iterative':CIDER_iterative}


def load_H_matrix(config):
    """Load H_matrix directly from data directory."""
    data_dir = config.data_dir
    H_file = os.path.join(data_dir, 'H_matrix.pt')
    if os.path.exists(H_file):
        H_data = torch.load(H_file, weights_only=True)
        return H_data.get('H_matrix', None)
    return None


def get_model_class(config):
    """Get model class based on config."""
    model_name = config.model.get('name', None)
    backbone_type = config.model.get('backbone_type', None)
    # Check both name and backbone_type for baseline detection
    if model_name in BASELINE_MODELS:
        return BASELINE_MODELS[model_name]
    if backbone_type in BASELINE_MODELS:
        return BASELINE_MODELS[backbone_type]
    return Diffusion



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    """Main entry point with mode routing."""
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    print(OmegaConf.to_yaml(config))
    print("="*60 + "\n")

    if config.mode == 'train':
        _train(config)
    elif config.mode == 'prism_head':
        _prism_head_train(config)
    elif config.mode == 'eval':
        _eval(config)
    elif config.mode == 'test':
        _test(config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


def _train(config: DictConfig):
    """
    Step 1: Train backbone.
    """
    print("\n" + "="*60)
    print("TRAINING MODE (Step 1: Backbone)")
    print("="*60 + "\n")

    if hasattr(config, 'seed'):
        L.seed_everything(config.seed)

    # Get dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print()

    # Initialize model
    print("Initializing model...")
    ModelClass = get_model_class(config)
    model = ModelClass(config)

    # Set H matrix from dataset (only for diffusion models)
    H = load_H_matrix(config)
    model_name = config.model.get('name', '')
    backbone_type = config.model.get('backbone_type', '')
    is_baseline = model_name in BASELINE_MODELS or backbone_type in BASELINE_MODELS
    if hasattr(model, 'set_H_matrix'):
        model.set_H_matrix(H)

    # Load weights from checkpoint (for transfer learning)
    checkpoint_path = config.get('checkpoint_path', None)
    if checkpoint_path:
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)

        # Load with strict=False to allow missing/extra keys (e.g., slot_init size change)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)} (will use random init)")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} (ignored)")

        # Load frozen_keys if present (for diffusion_transfer.py)
        if hasattr(model, 'frozen_keys') and 'frozen_keys' in ckpt:
            model.frozen_keys = set(ckpt['frozen_keys'])
            print(f"  Loaded {len(model.frozen_keys)} frozen keys")

    if is_baseline:
        print(f"✓ Baseline model initialized: {config.model.get('backbone_type', config.model.get('name', 'unknown'))}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print(f"✓ Diffusion model initialized: {config.model.get('backbone_type', 'dimp')}")
        print(f"  - D_model: {config.model.get('D_model', 256)}")
        print(f"  - Layers: {config.model.get('num_layers', 8)}")
        print(f"  - Heads: {config.model.get('heads', 8)}")
        print(f"  - use_soft_input: {config.model.get('use_soft_input', True)}")
        print(f"  - H matrix shape: {H.shape}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Setup callbacks
    callbacks = []

    exp_name = config.get('experiment_name', None) or 'dit_ecc'
    model_checkpoint_dir = os.path.join(config.checkpoint_dir, exp_name)
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_checkpoint_dir,
        filename='best_model',
        monitor=config.training.monitor,
        mode=config.training.mode,
        save_top_k=config.training.save_top_k,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(RichModelSummary(max_depth=2))
    callbacks.append(RichProgressBar())

    # Setup loggers
    loggers = []
    if config.training.wandb.enabled:
        wandb_logger = WandbLogger(
            project=config.training.wandb.project,
            entity=config.training.wandb.entity,
            name=config.training.wandb.get('name', None),
            notes=config.training.wandb.notes,
            tags=config.training.wandb.tags,
        )
        loggers.append(wandb_logger)
        print("✓ WandB logger initialized")

    # Create Trainer
    device = config.training.device if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.set_float32_matmul_precision('high')

    trainer = L.Trainer(
        max_epochs=config.training.num_epochs,
        callbacks=callbacks,
        logger=loggers if loggers else True,
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
        precision=config.training.precision if device == 'cuda' else 32,
        log_every_n_steps=config.training.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        overfit_batches=config.training.get('overfit_batches', 0),
        gradient_clip_val=config.training.get('gradient_clip_val', None),
    )
    print(f"✓ Trainer initialized (device={device}, precision={config.training.precision})")
    print()

    # Train
    print("Starting training...")
    resume_ckpt_path = config.get('resume_ckpt_path', None)
    if resume_ckpt_path:
        print(f"Resuming training from: {resume_ckpt_path}")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=resume_ckpt_path)

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best model: {model_checkpoint_dir}/best_model.ckpt")
    print(f"Last model: {model_checkpoint_dir}/last.ckpt")
    print("="*60 + "\n")

def _prism_head_train(config: DictConfig):
    """
    Train token quality head on frozen backbone.

    Trains a lightweight head to predict P(token is correct) for each position.
    Uses PRISM-style sampling: mask -> sample fills -> supervise quality.

    Requires:
        checkpoint_path: Path to trained backbone checkpoint
    """
    print("\n" + "="*60)
    print("QUALITY HEAD TRAINING MODE")
    print("="*60 + "\n")

    # Check checkpoint path
    if not hasattr(config, 'checkpoint_path') or config.checkpoint_path is None:
        raise ValueError("Must specify checkpoint_path for prism_head mode")

    if hasattr(config, 'seed'):
        L.seed_everything(config.seed)

    # Load backbone checkpoint
    print(f"Loading backbone from: {config.checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    backbone_config = OmegaConf.create(ckpt['hyper_parameters']['config'])

    # Load backbone model
    backbone_model = Diffusion(backbone_config)
    backbone_model.load_state_dict(ckpt['state_dict'], strict=False)

    # Load EMA weights if available
    if 'ema' in ckpt and 'shadow_params' in ckpt['ema']:
        print("Loading EMA weights into backbone...")
        shadow_params = ckpt['ema']['shadow_params']
        for name, param in backbone_model.backbone.named_parameters():
            if name in shadow_params:
                param.data.copy_(shadow_params[name])
        print("✓ EMA weights loaded")

    # Get frozen backbone
    backbone = backbone_model.backbone
    backbone.eval()

    # Get dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print()

    # Load H matrix
    H = load_H_matrix(config)

    # Build quality head config
    qh_config = {
        'prism_head': config.get('prism_head', {}),
        'sampler': config.get('sampler', {}),
        'masking': {
            'gamma_min': backbone_config.model.get('mask_gamma_min', 0.1),
            'gamma_max': backbone_config.model.get('mask_gamma_max', 1.0),
        },
        'lr': config.get('lr', 1e-3),
        'weight_decay': config.get('weight_decay', 0.01),
    }

    # Create QualityHeadTrainer
    from models.prism_head import QualityHeadTrainer
    model = QualityHeadTrainer(qh_config, backbone, H)

    print(f"✓ Model initialized with TokenQualityHead")
    print(f"  - Backbone: {backbone_config.model.get('backbone_type', 'unknown')} (frozen)")
    print(f"  - Prism head params: {sum(p.numel() for p in model.quality_head.parameters()):,}")
    print(f"  - k_per_slot: {qh_config['sampler'].get('k_per_slot', 4)}")
    print(f"  - n_y: {qh_config['sampler'].get('n_y', 8)}")
    print()

    # Setup callbacks
    callbacks = []

    # Save prism head in same directory as backbone checkpoint
    model_checkpoint_dir = config.get('experiment_name', None) or os.path.dirname(config.checkpoint_path)
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_checkpoint_dir,
        filename='best_prism_head',
        monitor='val/accuracy',
        mode='max',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(RichModelSummary(max_depth=2))
    callbacks.append(RichProgressBar())

    # Setup loggers
    loggers = []
    if config.training.wandb.enabled:
        wandb_logger = WandbLogger(
            project=config.training.wandb.get('project', 'prism-head'),
            entity=config.training.wandb.entity,
            name=config.training.wandb.get('name', f"prism_{os.path.basename(model_checkpoint_dir)}"),
            notes="Prism head training",
            tags=['prism_head'],
        )
        loggers.append(wandb_logger)
        print("✓ WandB logger initialized")

    # Create Trainer
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    num_epochs = config.get('num_epochs', 50)

    trainer = L.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=loggers if loggers else True,
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
        devices=1,
        precision=config.training.precision if device.type == 'cuda' else 32,
        log_every_n_steps=config.training.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
    )
    print(f"✓ Trainer initialized (device={device})")
    print()

    # Train
    print("Starting quality head training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\n" + "="*60)
    print("Quality head training completed!")
    print(f"Best model: {model_checkpoint_dir}/best_prism_head.ckpt")
    print(f"Last model: {model_checkpoint_dir}/last.ckpt")
    print("="*60 + "\n")


def _eval(config: DictConfig):
    """Evaluation mode."""
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60 + "\n")

    if not hasattr(config, 'checkpoint_path') or config.checkpoint_path is None:
        raise ValueError("Must specify checkpoint_path for eval mode")

    # Get dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print()

    # Select split
    eval_split = config.get('eval_split', 'test')
    if eval_split == 'train':
        eval_loader = train_loader
    elif eval_split == 'val':
        eval_loader = val_loader
    else:
        eval_loader = test_loader
    print(f"Evaluating on {eval_split.upper()} set")
    print()

    # Load model
    print(f"Loading model from: {config.checkpoint_path}")

    # Check if baseline or diffusion
    model_name = config.model.get('name', '')
    backbone_type = config.model.get('backbone_type', '')
    is_baseline = model_name in BASELINE_MODELS or backbone_type in BASELINE_MODELS

    if is_baseline:
        # Baseline models
        ModelClass = get_model_class(config)
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
        model = ModelClass(config)
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        # Diffusion models - use config from checkpoint for correct architecture
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')

        # Use checkpoint's config for model architecture (like visualize.py does)
        if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
            ckpt_config = OmegaConf.create(checkpoint['hyper_parameters']['config'])
            OmegaConf.set_struct(config, False)
            # Use checkpoint config entirely for architecture
            config.model = ckpt_config.model
            config.data = ckpt_config.data
            OmegaConf.set_struct(config, True)
            print(f"Using model config from checkpoint: D_model={config.model.D_model}, num_layers={config.model.num_layers}")
            print(f"  inference_steps={config.model.get('inference_steps', 'N/A')}")

        model = Diffusion(config)
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Filter out "H" key if present
        state_dict = {k: v for k, v in state_dict.items() if k != 'H'}
        model.load_state_dict(state_dict, strict=False)

        # Load EMA weights into backbone if available
        if 'ema' in checkpoint and 'shadow_params' in checkpoint['ema']:
            print("Loading EMA weights into backbone...")
            shadow_params = checkpoint['ema']['shadow_params']
            # Update model parameters directly (not a copy)
            for name, param in model.backbone.named_parameters():
                if name in shadow_params:
                    param.data.copy_(shadow_params[name])
            # Also update buffers
            for name, buf in model.backbone.named_buffers():
                if name in shadow_params:
                    buf.copy_(shadow_params[name])
            print("✓ EMA weights loaded")

        # Apply inference-time overrides (use top-level config keys)
        # Override use_soft_input if specified
        use_soft_input_override = config.get('use_soft_input', None)
        if use_soft_input_override is not None:
            print(f">>> Overriding use_soft_input: {config.model.get('use_soft_input', True)} -> {use_soft_input_override}")
            OmegaConf.set_struct(config, False)
            config.model.use_soft_input = use_soft_input_override
            OmegaConf.set_struct(config, True)
            # Re-initialize model with updated config
            model = Diffusion(config)
            model.load_state_dict(state_dict, strict=False)
            # Reload EMA if available
            if 'ema' in checkpoint and 'shadow_params' in checkpoint['ema']:
                shadow_params = checkpoint['ema']['shadow_params']
                for name, param in model.backbone.named_parameters():
                    if name in shadow_params:
                        param.data.copy_(shadow_params[name])

        inference_steps = config.get('inference_steps', None)
        if inference_steps is not None:
            print(f">>> Overriding inference_steps: {config.model.get('inference_steps', 'N/A')} -> {inference_steps}")
            OmegaConf.set_struct(config, False)
            config.model.inference_steps = inference_steps
            OmegaConf.set_struct(config, True)

        slot_init_scale = config.get('slot_init_scale', None)
        if slot_init_scale is not None and hasattr(model.backbone, 'slot_init'):
            original_norm = model.backbone.slot_init.data.norm().item()
            model.backbone.slot_init.data.mul_(slot_init_scale)
            new_norm = model.backbone.slot_init.data.norm().item()
            print(f">>> SCALING slot_init by {slot_init_scale} (norm: {original_norm:.4f} -> {new_norm:.4f}) <<<")

    H = load_H_matrix(config)
    if hasattr(model, 'set_H_matrix'):
        model.set_H_matrix(H)
    print("✓ Model loaded")
    print()

    # Create Trainer
    device = config.training.device if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.set_float32_matmul_precision('high')

    trainer = L.Trainer(
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
        precision=config.training.precision if device == 'cuda' else 32,
        callbacks=[RichModelSummary(max_depth=2), RichProgressBar()],
    )

    # Evaluate (use validate instead of test - same logic, avoids needing separate test_step)
    print("Running evaluation...")
    results = trainer.validate(model, dataloaders=eval_loader)

    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    for key, value in results[0].items():
        print(f"{key}: {value}")
    print("="*60 + "\n")


def _test(config: DictConfig):
    """
    Test mode with random_slot_first=False by default.

    This is for fair evaluation without random slot symmetry breaking.
    Use random_slot_first=true to enable it.
    """
    print("\n" + "="*60)
    print("TEST MODE")
    print("="*60 + "\n")

    if not hasattr(config, 'checkpoint_path') or config.checkpoint_path is None:
        raise ValueError("Must specify checkpoint_path for test mode")

    # Set random_slot_first=False by default for test mode
    OmegaConf.set_struct(config, False)
    if config.get('random_slot_first') is None:
        config.random_slot_first = False
    OmegaConf.set_struct(config, True)
    print(f"random_slot_first: {config.random_slot_first}")

    # Get dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print()

    # Select split
    eval_split = config.get('eval_split', 'test')
    if eval_split == 'train':
        eval_loader = train_loader
    elif eval_split == 'val':
        eval_loader = val_loader
    else:
        eval_loader = test_loader
    print(f"Testing on {eval_split.upper()} set")
    print()

    # Load model
    print(f"Loading model from: {config.checkpoint_path}")

    # Check if baseline or diffusion
    model_name = config.model.get('name', '')
    backbone_type = config.model.get('backbone_type', '')
    is_baseline = model_name in BASELINE_MODELS or backbone_type in BASELINE_MODELS

    if is_baseline:
        # Baseline models
        ModelClass = get_model_class(config)
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
        model = ModelClass(config)
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        # Diffusion models - use config from checkpoint for correct architecture
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')

        # Use checkpoint's config for model architecture (like visualize.py does)
        if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
            ckpt_config = OmegaConf.create(checkpoint['hyper_parameters']['config'])
            OmegaConf.set_struct(config, False)
            # Use checkpoint config entirely for architecture
            config.model = ckpt_config.model
            config.data = ckpt_config.data
            OmegaConf.set_struct(config, True)
            print(f"Using model config from checkpoint: D_model={config.model.D_model}, num_layers={config.model.num_layers}")
            print(f"  inference_steps={config.model.get('inference_steps', 'N/A')}")

        model = Diffusion(config)
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Filter out "H" key if present
        state_dict = {k: v for k, v in state_dict.items() if k != 'H'}
        model.load_state_dict(state_dict, strict=False)

        # Load EMA weights into backbone if available
        if 'ema' in checkpoint and 'shadow_params' in checkpoint['ema']:
            print("Loading EMA weights into backbone...")
            shadow_params = checkpoint['ema']['shadow_params']
            # Update model parameters directly (not a copy)
            for name, param in model.backbone.named_parameters():
                if name in shadow_params:
                    param.data.copy_(shadow_params[name])
            # Also update buffers
            for name, buf in model.backbone.named_buffers():
                if name in shadow_params:
                    buf.copy_(shadow_params[name])
            print("✓ EMA weights loaded")

        # Apply inference-time overrides (use top-level config keys)
        # Override use_soft_input if specified
        use_soft_input_override = config.get('use_soft_input', None)
        if use_soft_input_override is not None:
            print(f">>> Overriding use_soft_input: {config.model.get('use_soft_input', True)} -> {use_soft_input_override}")
            OmegaConf.set_struct(config, False)
            config.model.use_soft_input = use_soft_input_override
            OmegaConf.set_struct(config, True)
            # Re-initialize model with updated config
            model = Diffusion(config)
            model.load_state_dict(state_dict, strict=False)
            # Reload EMA if available
            if 'ema' in checkpoint and 'shadow_params' in checkpoint['ema']:
                shadow_params = checkpoint['ema']['shadow_params']
                for name, param in model.backbone.named_parameters():
                    if name in shadow_params:
                        param.data.copy_(shadow_params[name])

        # Override inference_steps if specified
        inference_steps = config.get('inference_steps', None)
        if inference_steps is not None:
            print(f">>> Overriding inference_steps: {config.model.get('inference_steps', 'N/A')} -> {inference_steps}")
            OmegaConf.set_struct(config, False)
            config.model.inference_steps = inference_steps
            OmegaConf.set_struct(config, True)

        # Scale slot_init if requested
        slot_init_scale = config.get('slot_init_scale', None)
        if slot_init_scale is not None and hasattr(model.backbone, 'slot_init'):
            original_norm = model.backbone.slot_init.data.norm().item()
            model.backbone.slot_init.data.mul_(slot_init_scale)
            new_norm = model.backbone.slot_init.data.norm().item()
            print(f">>> SCALING slot_init by {slot_init_scale} (norm: {original_norm:.4f} -> {new_norm:.4f}) <<<")

    H = load_H_matrix(config)
    if hasattr(model, 'set_H_matrix'):
        model.set_H_matrix(H)
    print("✓ Model loaded")
    print()

    # Create Trainer
    device = config.training.device if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.set_float32_matmul_precision('high')

    trainer = L.Trainer(
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
        precision=config.training.precision if device == 'cuda' else 32,
        callbacks=[RichModelSummary(max_depth=2), RichProgressBar()],
    )

    # Run test with timing
    print("Running test...")
    import time
    start_time = time.perf_counter()
    results = trainer.validate(model, dataloaders=eval_loader)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    num_samples = len(eval_loader.dataset)
    time_per_sample = total_time / num_samples * 1000  # ms

    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    for key, value in results[0].items():
        print(f"{key}: {value}")

    # Compute and display SER (Symbol Error Rate) and CER (Codeword Error Rate)
    res = results[0]
    symbol_acc = res.get('val/symbol_acc', res.get('val/accuracy', None))
    # codeword_acc: baselines use val/codeword_acc, diffusion uses val/micro_recall
    codeword_acc = res.get('val/codeword_acc', res.get('val/micro_recall', None))

    print()
    print("Error Rates:")
    if symbol_acc is not None:
        ser = 1.0 - symbol_acc
        print(f"  SER (Symbol Error Rate):   {ser:.6f} ({ser*100:.4f}%)")
    if codeword_acc is not None:
        cer = 1.0 - codeword_acc
        print(f"  CER (Codeword Error Rate): {cer:.6f} ({cer*100:.4f}%)")
    print()
    print("Timing:")
    print(f"  Total time:      {total_time:.2f} sec")
    print(f"  Samples:         {num_samples}")
    print(f"  Time per sample: {time_per_sample:.2f} ms")
    print(f"  Throughput:      {num_samples/total_time:.1f} samples/sec")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

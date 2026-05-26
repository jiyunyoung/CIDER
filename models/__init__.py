"""
Model architectures.
"""

# CIDER
from .cider import DiMP as CIDER

# CIDER without module A
from .cider_noA import DiMP as CIDER_NoSlot

# CIDER without module B
from .cider_noB import DiMP as CIDER_NoMP

# CIDER direct
from .cider_direct import DiMPNoGRUOneShot as CIDER_direct

# CIDER iterative (no masking)
from .cider_iterative import DiMPIterative as CIDER_iterative

# MDD
from .mdd import DiT, DDiTFinalLayer

# CIDER with GRU
from .cider_gru import DiMP as CIDER_GRU

# CIDER with GRU direct
from .cider_gru_direct import DiMPRev2OneShot as CIDER_GRU_direct

# PRSIM head
from .prism_head import TokenQualityHead, PRISMSampler, QualityHeadTrainer

# EMA
from .ema import ExponentialMovingAverage

# Baselines (No diffusion)
from .mlp import MLPDemixer as MLP
from .cnn import CNNDemixer as CNN
from .transformer import TransformerDemixer as Transformer
from .gnn import GNNDemixer as GNN
from .nbp import UnfoldedBPDemixer as NBP
from .mpa import DiMPOneShot as MPA

__all__ = [
    # CIDER
    'CIDER',
    'CIDER_NoSlot',
    'CIDER_NoMP',
    'CIDER_direct',
    'CIDER_iterative',
    'CIDER_GRU',
    'CIDER_GRU_direct',
    # MDD
    'DiT',
    'DDiTFinalLayer',
    # PRISM
    'TokenQualityHead',
    'PRISMSampler',
    'QualityHeadTrainer',
    # Baselines
    'MLP',
    'CNN',
    'Transformer',
    'GNN',
    'NBP',
    'MPA',
    # EMA
    'ExponentialMovingAverage',
]

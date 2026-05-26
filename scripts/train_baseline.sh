#!/bin/bash
#SBATCH -J train_baseline                # Job name
#SBATCH -o logs/%x_%j.out               # Output file
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=32000
#SBATCH -t 48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --requeue

# ============================================================
# Baseline Training Script
#
# Usage:
#   ./train_baseline.sh                                    # defaults
#   ./train_baseline.sh tiny_ldpc mlp                      # data + model
#   ./train_baseline.sh tiny_ldpc cnn
#   ./train_baseline.sh tiny_ldpc transformer
#   ./train_baseline.sh tiny_ldpc gnn
#   ./train_baseline.sh tiny_ldpc nbp
#   ./train_baseline.sh moderate_ldpc cider_direct
#   ./train_baseline.sh moderate_ldpc cider_gru_direct
#
# Config groups:
#   - data: tiny_ldpc, small_ldpc, moderate_ldpc
#   - model: mlp, cnn, transformer, gnn, nbp
#            cider_direct, cider_gru_direct, mpa
#
# Note: Baselines have fixed architecture (no size config needed)
# ============================================================

set -e

DATA="${1:-tiny_ldpc}"
MODEL="${2:-mlp}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Experiment name
EXPERIMENT="${DATA}_${MODEL}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT}"

# Create directories
mkdir -p logs
mkdir -p "$CHECKPOINT_DIR"

# Export environment
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "Baseline Training"
echo "============================================================"
echo "Data: $DATA | Model: $MODEL"
echo "Experiment: $EXPERIMENT"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "============================================================"

# Run training
python -u main.py \
  mode=train \
  data=$DATA \
  model=$MODEL \
  checkpoint_dir=checkpoints \
  experiment_name=$EXPERIMENT \
  optim.lr=1e-3 \
  optim.weight_decay=0.01 \
  training.batch_size=128 \
  training.num_epochs=100 \
  training.num_workers=4 \
  training.warmup_epochs=10 \
  training.wandb.enabled=true \
  training.wandb.project=muecc-demixing \
  training.wandb.name=$EXPERIMENT \
  $EXTRA_ARGS

echo "============================================================"
echo "Training complete!"
echo "Checkpoint saved to: $CHECKPOINT_DIR"
echo "============================================================"

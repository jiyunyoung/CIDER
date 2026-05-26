#!/bin/bash
#SBATCH -J train_muecc                  # Job name
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
# Diffusion Training Script
#
# Usage:
#   ./train_diffusion.sh                                    # defaults
#   ./train_diffusion.sh tiny_ldpc tiny cider               # data + size + model
#   ./train_diffusion.sh moderate_ldpc moderate cider_gru   # moderate size
#
# Config groups (independently selectable):
#   - data: tiny_ldpc, small_ldpc, moderate_ldpc
#   - size: tiny, small, moderate, large
#   - model: cider, cider_gru, cider_noA, cider_noB, mdd
# ============================================================

set -e

DATA="${1:-tiny_ldpc}"
SIZE="${2:-tiny}"
MODEL="${3:-cider}"
shift 3 2>/dev/null || shift 2 2>/dev/null || shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Experiment name
EXPERIMENT="${DATA}_${SIZE}_${MODEL}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT}"

# Create directories
mkdir -p logs
mkdir -p "$CHECKPOINT_DIR"

# Export environment
export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "Data: $DATA | Size: $SIZE | Model: $MODEL"
echo "Experiment: $EXPERIMENT"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "============================================================"

# Run training
python -u main.py \
  mode=train \
  data=$DATA \
  size=$SIZE \
  model=$MODEL \
  checkpoint_dir=checkpoints \
  experiment_name=$EXPERIMENT \
  optim.lr=1e-3 \
  optim.weight_decay=1e-4 \
  training.batch_size=128 \
  training.num_epochs=100 \
  training.num_workers=4 \
  training.warmup_epochs=10 \
  training.wandb.enabled=true \
  training.wandb.project=muecc-demixing \
  training.wandb.name=$EXPERIMENT \
  $EXTRA_ARGS

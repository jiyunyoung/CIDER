#!/bin/bash
# ============================================================
# PRISM Head Training Script
#
# Trains TokenQualityHead on frozen backbone using PRISM sampling.
#
# Usage:
#   ./scripts/train_prism_head.sh <data> <size> <backbone> [extra_args...]
#
# Examples:
#   ./scripts/train_prism_head.sh moderate_ldpc moderate cider
#   ./scripts/train_prism_head.sh moderate_ldpc moderate cider sampler.k_per_slot=8
#   ./scripts/train_prism_head.sh tiny_ldpc tiny cider_gru sampler.temperature=1.2
# ============================================================

set -e

# Required arguments
DATA="${1:?Usage: $0 <data> <size> <backbone> [extra_args...]}"
SIZE="${2:?Usage: $0 <data> <size> <backbone> [extra_args...]}"
BACKBONE="${3:?Usage: $0 <data> <size> <backbone> [extra_args...]}"
shift 3 2>/dev/null || true
EXTRA_ARGS="$@"

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Find backbone checkpoint
BACKBONE_EXP="${DATA}_${SIZE}_${BACKBONE}"
BACKBONE_CKPT="checkpoints/${BACKBONE_EXP}/best_model.ckpt"

if [ ! -f "$BACKBONE_CKPT" ]; then
    BACKBONE_CKPT="checkpoints/${BACKBONE_EXP}/last.ckpt"
    if [ ! -f "$BACKBONE_CKPT" ]; then
        echo "Error: Backbone checkpoint not found at checkpoints/${BACKBONE_EXP}/"
        echo "Please train the backbone first:"
        echo "  ./scripts/train_diffusion.sh $DATA $SIZE $BACKBONE"
        exit 1
    fi
fi

echo "============================================================"
echo "PRISM Head Training"
echo "============================================================"
echo "Data: $DATA"
echo "Size: $SIZE"
echo "Backbone: $BACKBONE"
echo "Checkpoint: $BACKBONE_CKPT"
echo "Extra args: $EXTRA_ARGS"
echo "============================================================"

# Run training (load prism config from configs/adapter/)
python -u main.py \
    mode=prism_head \
    data=$DATA \
    size=$SIZE \
    checkpoint_path="$BACKBONE_CKPT" \
    +adapter@_global_=prism \
    $EXTRA_ARGS

echo "============================================================"
echo "Training complete!"
echo "============================================================"

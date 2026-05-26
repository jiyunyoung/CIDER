#!/bin/bash
# ============================================================
# Baseline Evaluation Script
#
# Usage:
#   ./eval_baseline.sh <data> <model> [options]
#
# Baseline models:
#   ./eval_baseline.sh tiny_ldpc baseline_mlp
#   ./eval_baseline.sh tiny_ldpc baseline_cnn
#   ./eval_baseline.sh tiny_ldpc baseline_transformer
#   ./eval_baseline.sh tiny_ldpc baseline_gnn
#   ./eval_baseline.sh tiny_ldpc baseline_nbp
#   ./eval_baseline.sh tiny_ldpc cider_direct
#   ./eval_baseline.sh tiny_ldpc cider_gru_direct
#   ./eval_baseline.sh tiny_ldpc mpa
#
# Supported models:
#   - baseline_mlp, baseline_cnn, baseline_transformer
#   - baseline_gnn, baseline_nbp (uses H matrix)
#   - cider_direct, cider_gru_direct, mpa (one-shot)
#
# Uses mode=test for evaluation
# ============================================================

set -e

DATA="${1:-tiny_ldpc}"
SIZE="${2:-tiny}"
MODEL="${3:-baseline_mlp}"
shift 3 2>/dev/null || shift 2 2>/dev/null || shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Determine if baseline or diffusion model
BASELINES="baseline_mlp baseline_cnn baseline_transformer baseline_gnn baseline_nbp cider_direct cider_gru_direct mpa"
IS_BASELINE=false
for b in $BASELINES; do
    if [ "$MODEL" == "$b" ]; then
        IS_BASELINE=true
        break
    fi
done

# Find checkpoint based on model type
CHECKPOINT=""
if [ "$IS_BASELINE" = true ]; then
    # Baselines: checkpoints/${DATA}_${MODEL}/best_model.ckpt
    pattern="${DATA}_${MODEL}"
    if [ -f "checkpoints/${pattern}/best_model.ckpt" ]; then
        CHECKPOINT="checkpoints/${pattern}/best_model.ckpt"
    fi
else
    # Diffusion: checkpoints/${DATA}_${SIZE}_${MODEL}/best_model.ckpt
    for pattern in "${DATA}_${SIZE}_${MODEL}" "${DATA}_${MODEL}"; do
        if [ -f "checkpoints/${pattern}/best_model.ckpt" ]; then
            CHECKPOINT="checkpoints/${pattern}/best_model.ckpt"
            break
        fi
    done
fi

# Check if checkpoint exists
if [ -z "$CHECKPOINT" ]; then
    echo "Checkpoint not found. Tried:"
    if [ "$IS_BASELINE" = true ]; then
        echo "  - checkpoints/${DATA}_${MODEL}/best_model.ckpt"
    else
        echo "  - checkpoints/${DATA}_${SIZE}_${MODEL}/best_model.ckpt"
        echo "  - checkpoints/${DATA}_${MODEL}/best_model.ckpt"
    fi
    echo ""
    echo "Available checkpoints:"
    ls -d checkpoints/*/ 2>/dev/null | head -20 || echo "  (none)"
    exit 1
fi

echo "============================================================"
echo "Test (random_slot_first=False)"
echo "============================================================"
if [ "$IS_BASELINE" = true ]; then
    echo "Data: $DATA | Model: $MODEL (baseline)"
else
    echo "Data: $DATA | Size: $SIZE | Model: $MODEL"
fi
echo "Checkpoint: $CHECKPOINT"
echo "============================================================"

if [ "$IS_BASELINE" = true ]; then
    python -u main.py \
        mode=test \
        data=$DATA \
        model=$MODEL \
        checkpoint_path=$CHECKPOINT \
        $EXTRA_ARGS
else
    python -u main.py \
        mode=test \
        data=$DATA \
        size=$SIZE \
        model=$MODEL \
        checkpoint_path=$CHECKPOINT \
        $EXTRA_ARGS
fi

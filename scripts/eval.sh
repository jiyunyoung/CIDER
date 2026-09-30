#!/bin/bash
# ============================================================
# Test/Evaluation Script
#
# Usage:
#   ./eval.sh <data> <size> <model> [options]
#
# Diffusion models:
#   ./eval.sh tiny_ldpc tiny cider
#   ./eval.sh small_ldpc tiny cider_gru eval_split=val
#
# Baseline models:
#   ./eval.sh tiny_ldpc _ mlp
#   ./eval.sh tiny_ldpc _ cnn
#   ./eval.sh tiny_ldpc _ transformer
#
# Options:
#   random_slot_first=true   Enable random slot first reveal (default: false)
#   +inference_steps=N       Override inference steps T. Diffusion models default
#                            to the paper's T per size (tiny 12, small 16,
#                            moderate 20, large 28). Note: model.inference_steps=N
#                            has no effect in test mode, because the model config
#                            is restored from the checkpoint.
#   +fast_sampler=true       Vectorized K<=2 sampler (faster on GPU; not bit-identical
#                            to the default sampler, see Diffusion._sample_discrete_fast)
#
# Supported models:
#   - cider, cider_gru (diffusion message passing)
#   - mlp, cnn, transformer, gnn, nbp, mpa (one-shot)
#
# Uses mode=test (random_slot_first=False by default for fair evaluation)
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

# Determine if baseline or diffusion model
BASELINES="mlp cnn transformer gnn nbp cider_direct cider_gru_direct mpa"
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

# Paper inference steps T per size (tab:app_exact_model_sizing). The model
# config is restored from the checkpoint in test mode, so T must be passed as
# +inference_steps; the size yamls' inference_steps only sets T_train (and
# the default T) for newly trained models.
declare -A PAPER_T=([tiny]=12 [small]=16 [moderate]=20 [large]=28)
T_ARG=""
if [ "$IS_BASELINE" = false ] && [[ "$EXTRA_ARGS" != *inference_steps=* ]] && [ -n "${PAPER_T[$SIZE]}" ]; then
    T_ARG="+inference_steps=${PAPER_T[$SIZE]}"
fi

echo "============================================================"
echo "Test (random_slot_first=False)"
echo "============================================================"
if [ "$IS_BASELINE" = true ]; then
    echo "Data: $DATA | Model: $MODEL (baseline)"
else
    echo "Data: $DATA | Size: $SIZE | Model: $MODEL | ${T_ARG:-inference_steps from args/checkpoint}"
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
        $T_ARG $EXTRA_ARGS
fi

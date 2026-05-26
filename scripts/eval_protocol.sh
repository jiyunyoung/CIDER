#!/bin/bash
# ============================================================
# Protocol-Level Evaluation for Random Access
#
# Simulates random access with ZC preambles and LDPC demixing.
# Uses per-K inference steps configured in eval_protocol.py.
#
# Usage:
#   ./eval_protocol.sh [options]
#   ./eval_protocol.sh --user_counts 10 15 20 25 30
#   ./eval_protocol.sh --num_frames 500 --user_counts 10 20 30 40
#
# Options:
#   --checkpoint_dir PATH        Checkpoint directory (default: checkpoints/protocol_scale)
#                                Expected layout: <dir>/K{1..8}/best_model.ckpt
#   --data_dir PATH              Data directory (default: data/gen_data/datasets/protocol_Eb10)
#                                Expected layout: <dir>/K{1..8}/test_data.pt
#   --user_counts N...           Active user counts K_a to sweep
#                                (default: 10 15 20 25 30)
#   --num_frames N               Frames per user count (default: 1000)
#   --batch_size N               Inference batch size (default: 32)
#   --K_range KMIN KMAX          Decoder K range [min,max), K>=KMAX is overflow
#                                (default: 1 9 -> K=1..8)
#   --save_results PATH          Save results to JSON file
#
# Preamble allocation:
#   (default)                    Flexible at --target_load 5 (set in this script)
#   --target_load F              Override the default load (e.g. --target_load 3.0)
#
#   To use FIXED preambles (--num_preambles) or EXPLICIT per-K_a counts
#   (--preambles_per_count), call inference/eval_protocol.py directly —
#   this wrapper hard-codes --target_load and Python's argparse picks
#   --target_load over --num_preambles when both are present.
#
# Per-K inference steps (configured in eval_protocol.py):
#   K=2: 12, K=3: 24, K=4: 42, K=5: 60, K=6: 50, K=7: 62, K=8: 74
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default to flexible preambles at constant per-slot load — but only if the
# user didn't pass an explicit slot-allocation flag. Argparse uses an elif
# chain, so silently injecting --target_load would override --num_preambles
# and --preambles_per_count without warning.
DEFAULT_TARGET_LOAD=5

USER_SET_SLOTS=0
for arg in "$@"; do
    case "$arg" in
        --target_load|--target_load=*|--num_preambles|--num_preambles=*|--preambles_per_count|--preambles_per_count=*)
            USER_SET_SLOTS=1
            break
            ;;
    esac
done

if [ "$USER_SET_SLOTS" = "1" ]; then
    python -u inference/eval_protocol.py "$@"
else
    python -u inference/eval_protocol.py --target_load "$DEFAULT_TARGET_LOAD" "$@"
fi

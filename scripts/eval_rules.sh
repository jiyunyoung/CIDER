#!/bin/bash
# ============================================================
# Evaluate Rule-Based Decoders (sic_bp, top_j_es)
#
# Usage:
#   ./scripts/eval_rules.sh <decoder> <data> [options]
#
# Examples:
#   ./scripts/eval_rules.sh sic_bp tiny_ldpc
#   ./scripts/eval_rules.sh top_j_es tiny_ldpc
#   ./scripts/eval_rules.sh sic_bp moderate_ldpc --max_iters 100
#
# Decoders:
#   - sic_bp: SIC-BP (iterative BP with explain-away)
#   - top_j_es: Top-J exhaustive search with parity pruning
#
# Options (sic_bp):
#   --max_iters N         Maximum BP iterations (default: 50)
#   --damping F           Message damping factor (default: 0.1)
#   --explain_strength F  Explain-away strength (default: 1.0)
#
# Options (top_j_es):
#   --proposal_width N    Top-L candidates per position (default: 2)
#
# Common options:
#   --split train/val/test  Data split (default: test)
# ============================================================

set -e

DECODER="${1:-sic_bp}"
DATA="${2:-tiny_ldpc}"
shift 2 2>/dev/null || true
EXTRA_ARGS="$@"

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Rule-Based Decoder Evaluation"
echo "============================================================"
echo "Decoder: $DECODER"
echo "Data: $DATA"
echo "Extra args: $EXTRA_ARGS"
echo "============================================================"

python -u inference/eval_rules.py "$DECODER" "$DATA" $EXTRA_ARGS

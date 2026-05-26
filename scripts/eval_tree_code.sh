#!/bin/bash
# ============================================================
# Stitching decoder on tree code (J=3)
# ============================================================
set -e
cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

NUM_TEST="${1:-15000}"
LOG_DIR="logs/tree_code"
DATA_DIR=~/data/demix/tiny_tree
mkdir -p "$LOG_DIR"

# Generate data if needed
if [ ! -f "$DATA_DIR/test_data.pt" ]; then
    echo "Generating tree code test data..."
    H_PATH="data/gen_data/H_tiny_tree.pt"
    if [ ! -f "$H_PATH" ]; then
        python data/gen_data/construct_tree_code.py --q 64 --L 12 --seed 42 --output "$H_PATH"
    fi
    python data/gen_data/generate_data_from_H.py \
        --h_matrix "$H_PATH" \
        --K 2 --num_train 0 --num_val 0 --num_test $NUM_TEST \
        --n_s 24 --Eb 10.0 --sigma2 1.0 --seed 42 --output "$DATA_DIR"
fi

echo "=== Stitching (J=3) ==="
python -u inference/eval_rules.py stitching tiny_tree \
    --proposal_width 3 --beam_width 10000 --num_samples $NUM_TEST \
    2>&1 | tee "$LOG_DIR/stitching_j3.log"

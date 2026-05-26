#!/bin/bash
# ============================================================
# Generate test-only datasets for moderate_LDPC and large_LDPC
#
# Usage:
#   ./scripts/gen_test_data.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

H_DIR="data/gen_data"
mkdir -p "$H_DIR"

# ============================================================
# Step 1: Construct H matrices (deterministic, seed=42)
# ============================================================

echo "=== Constructing H matrices ==="

# Moderate: Q=64, L=24, M=16, d_v=2, d_c=3
echo "  Moderate (Q=64, L=24, M=16)..."
python data/gen_data/construct_H.py \
    --q 64 --L 24 --M 16 --d_v 2 --d_c 3 \
    --seed 42 --output "$H_DIR/H_moderate_ldpc.pt"

# Large: Q=64, L=48, M=32, d_v=2, d_c=3
echo "  Large (Q=64, L=48, M=32)..."
python data/gen_data/construct_H.py \
    --q 64 --L 48 --M 32 --d_v 2 --d_c 3 \
    --seed 42 --output "$H_DIR/H_large_ldpc.pt"

# ============================================================
# Step 2: Generate test-only data (Eb=10, K=2, same as tiny)
# ============================================================

echo ""
echo "=== Generating test datasets ==="

# Moderate
echo "  Moderate (15000 test samples)..."
python data/gen_data/generate_data_from_H.py \
    --h_matrix "$H_DIR/H_moderate_ldpc.pt" \
    --K 2 --num_train 0 --num_val 0 --num_test 15000 \
    --n_s 24 --Eb 10.0 --sigma2 1.0 \
    --seed 42 \
    --output ~/data/demix/moderate_LDPC/

# Large
echo "  Large (15000 test samples)..."
python data/gen_data/generate_data_from_H.py \
    --h_matrix "$H_DIR/H_large_ldpc.pt" \
    --K 2 --num_train 0 --num_val 0 --num_test 15000 \
    --n_s 24 --Eb 10.0 --sigma2 1.0 \
    --seed 42 \
    --output ~/data/demix/large_LDPC/

echo ""
echo "=== Done ==="
echo "Datasets saved to:"
echo "  ~/data/demix/moderate_LDPC/"
echo "  ~/data/demix/large_LDPC/"

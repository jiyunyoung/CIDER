#!/bin/bash
#
# Generate the PEG-LDPC dataset (paper Table 2c / tab:app_peg_ldpc, and the
# PEG-LDPC row of tab:response_transfer).
# Same parameters as tiny LDPC (Q=64, L=12, M=8, K=2, n_s=24, Eb=10 dB) but the
# Tanner graph comes from pure PEG (construct_H_peg.py) instead of the fixed
# 3-cube graph of construct_H.py.
#
# SEED=42 is used for both steps: construct_H_peg.py --seed 42 reproduces the
# stored ~/data/demix/tiny_LDPC_PEG/H_matrix.pt, and generate_data_from_H.py
# --seed 42 fixes the sensing matrix A (shared with every other dataset) and the
# train/val/test stream, as in the original dataset_metadata.json. val/test are
# drawn after train in the same stream, so they never replay training frames.
#
# Usage:
#   ./ldpc_tiny_peg.sh [N_S] [EB] [K]
#   ./ldpc_tiny_peg.sh              # Defaults: n_s=24, Eb=10.0, K=2

Q=64
L=12
M=8
D_V=2
D_C=3

N_S="${1:-24}"
EB="${2:-10.0}"
K="${3:-2}"

M_ANT=1
SIGMA2=1.0
MATRIX_TYPE="partial_dft"

NUM_TRAIN=70000
NUM_VAL=15000
NUM_TEST=15000
BATCH_SIZE=2048
SEED=42   # H seed (PEG tie-breaks + coefficients) and data/sensing seed
DEVICE="cuda"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$HOME/data/demix/tiny_LDPC_PEG"

echo "============================================================"
echo "PEG-LDPC Dataset Generation"
echo "============================================================"
echo "  Q=$Q, L=$L, M=$M, d_v=$D_V, d_c=$D_C, k=$((L - M))"
echo "  Construction: PEG (Progressive Edge Growth)"
echo "  K=$K, n_s=$N_S, Eb=${EB}dB"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"

# Verify degree constraint
if [ $((L * D_V)) -ne $((M * D_C)) ]; then
    echo "ERROR: L*d_v != M*d_c"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Step 1: Construct PEG H matrix
echo ""
echo "Step 1: Constructing PEG LDPC H matrix..."
python ${SCRIPT_DIR}/construct_H_peg.py \
    --q $Q --L $L --M $M --d_v $D_V --d_c $D_C \
    --seed $SEED --output ${OUTPUT_DIR}/H_matrix.pt --show || exit 1

# Step 2: Generate data
echo ""
echo "Step 2: Generating data..."
python ${SCRIPT_DIR}/generate_data_from_H.py \
    --h_matrix ${OUTPUT_DIR}/H_matrix.pt \
    --K $K --n_s $N_S --Eb $EB \
    --num_train $NUM_TRAIN --num_val $NUM_VAL --num_test $NUM_TEST \
    --batch_size $BATCH_SIZE --device $DEVICE --seed $SEED \
    --output $OUTPUT_DIR || exit 1

echo ""
echo "Done! Output: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"

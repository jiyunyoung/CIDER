#!/bin/bash
#
# Generate LDPC dataset using two-step workflow:
#   Step 1: construct_H.py → H_matrix.pt
#   Step 2: generate_data_from_H.py → train/val/test data
#
# Usage:
#   ./ldpc_small.sh [N_S] [EB] [K]
#   ./ldpc_small.sh              # Defaults: n_s=24, Eb=11.0, K=2
#   ./ldpc_small.sh 32 15.0 3    # n_s=32, Eb=15.0, K=3
#   ./ldpc_small.sh --dry-run    # Print commands without running

# ============================================================
# Step 1: H Matrix Parameters (outer code structure)
# ============================================================
Q=64  # Alphabet size GF(q)
L=12  # Codeword length
M=8   # Number of parity checks
D_V=2 # Variable node degree
D_C=3 # Check node degree

# ============================================================
# Handle --dry-run first
# ============================================================
DRY_RUN=false
ARGS=()
for arg in "$@"; do
  if [ "$arg" == "--dry-run" ]; then
    DRY_RUN=true
  else
    ARGS+=("$arg")
  fi
done

# ============================================================
# Step 2: Data Generation Parameters
# ============================================================
# These are the main parameters you'd adjust:
N_S="${ARGS[0]:-24}"  # Inner code length (channel uses per symbol) - tuned for ~99% all-K
EB="${ARGS[1]:-10.0}" # Energy per bit (dB)
K="${ARGS[2]:-3}"     # Number of active users

# Fixed inner code params
M_ANT=1    # Number of receive antennas
SIGMA2=1.0 # Noise variance
MATRIX_TYPE="partial_dft"

# Dataset sizes
NUM_TRAIN=70000
NUM_VAL=15000
NUM_TEST=15000
BATCH_SIZE=2048

SEED=42
DEVICE="cuda"

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${HOME}/data/demix/tiny_K3"

# ============================================================
# Print Configuration
# ============================================================
echo "============================================================"
echo "LDPC Dataset Generation (Two-Step Workflow)"
echo "============================================================"
echo ""
echo "Step 1: Construct H matrix (stored in H_matrix.pt)"
echo "  Q=$Q, L=$L, M=$M, d_v=$D_V, d_c=$D_C, k=$((L - M))"
echo ""
echo "Step 2: Generate data (reads Q, L from H_matrix.pt)"
echo "  K=$K, n_s=$N_S, Eb=${EB}dB"
echo ""
echo "Dataset: train=$NUM_TRAIN, val=$NUM_VAL, test=$NUM_TEST"
echo "Output:  $OUTPUT_DIR"
echo "============================================================"

# Check degree constraint
if [ $((L * D_V)) -ne $((M * D_C)) ]; then
  echo "ERROR: L*d_v=$((L * D_V)) != M*d_c=$((M * D_C))"
  exit 1
fi

# ============================================================
# Commands
# ============================================================
mkdir -p "$OUTPUT_DIR"

CMD_STEP1="python ${SCRIPT_DIR}/construct_H.py \
    --q $Q --L $L --M $M --d_v $D_V --d_c $D_C \
    --seed $SEED --output ${OUTPUT_DIR}/H_matrix.pt --show"

CMD_STEP2="python ${SCRIPT_DIR}/generate_data_from_H.py \
    --h_matrix ${OUTPUT_DIR}/H_matrix.pt \
    --K $K --n_s $N_S --Eb $EB \
    --num_train $NUM_TRAIN --num_val $NUM_VAL --num_test $NUM_TEST \
    --batch_size $BATCH_SIZE --device $DEVICE \
    --output $OUTPUT_DIR"

echo ""
echo "Step 1: $CMD_STEP1"
echo ""
echo "Step 2: $CMD_STEP2"
echo ""

if [ "$DRY_RUN" = true ]; then
  echo "[Dry run - not executed]"
  exit 0
fi

# ============================================================
# Run
# ============================================================
echo "Running Step 1..." && eval $CMD_STEP1 || exit 1
echo ""
echo "Running Step 2..." && eval $CMD_STEP2 || exit 1
echo ""
echo "Done! Output: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"

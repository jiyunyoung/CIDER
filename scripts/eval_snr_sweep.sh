#!/bin/bash
# ============================================================
# SNR sweep for CIDER, on-the-fly test data with fixed seed.
#
# Sweeps per-user per-complex-channel-use SNR ∈ {-5,-4,...,3} dB.
# Conversion (tiny_ldpc: B=24, L=12, n_s=24, R=B/(L*n_s)=1/12):
#     Eb/N0 [dB] = SNR [dB] + 10*log10(L*n_s/B) ≈ SNR + 10.79 dB
#
# Usage:
#   ./scripts/eval_snr_sweep.sh                              # K=2, default ckpt
#   ./scripts/eval_snr_sweep.sh 3                            # K=3, default ckpt
#   ./scripts/eval_snr_sweep.sh 2 path/to/ckpt.ckpt          # custom checkpoint
# ============================================================
set -e

K="${1:-2}"
CKPT="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Auto-resolve checkpoint by K if not given. Tries known naming patterns.
if [ -z "$CKPT" ]; then
    CANDIDATES=(
        "checkpoints/protocol_scale/K${K}/best_model.ckpt"
        "checkpoints/tiny_ldpc_K${K}_tiny_cider/best_model.ckpt"
        "checkpoints/tiny_ldpc_K${K}_tiny_16_cider/best_model.ckpt"
    )
    if [ "$K" = "2" ]; then
        CANDIDATES+=("checkpoints/tiny_ldpc_tiny_cider/best_model.ckpt")
    fi
    for c in "${CANDIDATES[@]}"; do
        if [ -f "$c" ]; then
            CKPT="$c"
            break
        fi
    done
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found for K=$K. Tried:"
    for c in "${CANDIDATES[@]}"; do echo "  - $c"; done
    echo "Pass an explicit path:  ./scripts/eval_snr_sweep.sh $K path/to/ckpt"
    exit 1
fi

LOG_DIR="logs/snr_sweep_K${K}"
mkdir -p "$LOG_DIR"

OFFSET=10.79  # 10*log10(L*n_s/B) = 10*log10(12) for tiny_ldpc
SNR_VALUES="-4 -3 -2 -1 0 1 2 3 4"

echo "============================================================"
echo "SNR sweep  (per-user per-complex-channel-use SNR)"
echo "K=$K   Checkpoint: $CKPT"
echo "Eb/N0 = SNR + ${OFFSET} dB   (tiny_ldpc rate)"
echo "============================================================"

for SNR in $SNR_VALUES; do
    EB=$(python -c "print(round($SNR + $OFFSET, 4))")
    TAG="K${K}_SNR${SNR}"
    echo
    echo "--- SNR=${SNR} dB   (Eb/N0=${EB} dB) ---"
    python -u main.py \
        mode=test \
        data=onthefly_tiny \
        size=tiny \
        model=cider \
        checkpoint_path="$CKPT" \
        data.K_max=$K data.K_true=$K \
        data.Eb_dB=$EB \
        training.wandb.enabled=false \
        2>&1 | tee "$LOG_DIR/${TAG}.log"
done

echo
echo "============================================================"
echo "Summary  (K=${K})"
echo "============================================================"
printf "%-6s %-8s %12s %12s %12s\n" "SNR" "Eb/N0" "SER" "CER" "ms/sample"
echo "------------------------------------------------------------"
for SNR in $SNR_VALUES; do
    EB=$(python -c "print(round($SNR + $OFFSET, 4))")
    TAG="K${K}_SNR${SNR}"
    LOG="$LOG_DIR/${TAG}.log"
    if [ -f "$LOG" ]; then
        SER=$(grep "SER (Symbol Error Rate)" "$LOG" | grep -oP '[\d.]+' | head -1)
        CER=$(grep "CER (Codeword Error Rate)" "$LOG" | grep -oP '[\d.]+' | head -1)
        MS=$(grep "Time per sample" "$LOG" | grep -oP '[\d.]+' | head -1)
        printf "%-6s %-8s %12s %12s %12s\n" "$SNR" "$EB" "${SER:-N/A}" "${CER:-N/A}" "${MS:-N/A}"
    fi
done

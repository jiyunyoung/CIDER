#!/bin/bash
# ============================================================
# Run SNR sweep for K=2..5 then print combined summary.
# Skips K=1 (not joint decoding).
#
# Usage: ./scripts/eval_snr_sweep_all.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

K_VALUES="2 3 4 5"
SNR_VALUES="-4 -3 -2 -1 0 1 2 3 4"
OFFSET=10.79

for K in $K_VALUES; do
    echo
    echo "############################################################"
    echo "#  K=$K"
    echo "############################################################"
    "$SCRIPT_DIR/eval_snr_sweep.sh" $K
done

echo
echo "============================================================"
echo "Combined Summary  (K=2..5)"
echo "============================================================"
printf "%-4s %-6s %-8s %12s %12s\n" "K" "SNR" "Eb/N0" "SER" "CER"
echo "------------------------------------------------------------"
for K in $K_VALUES; do
    for SNR in $SNR_VALUES; do
        EB=$(python -c "print(round($SNR + $OFFSET, 4))")
        TAG="K${K}_SNR${SNR}"
        LOG="logs/snr_sweep_K${K}/${TAG}.log"
        if [ -f "$LOG" ]; then
            SER=$(grep "SER (Symbol Error Rate)" "$LOG" | grep -oP '[\d.]+' | head -1)
            CER=$(grep "CER (Codeword Error Rate)" "$LOG" | grep -oP '[\d.]+' | head -1)
            printf "%-4s %-6s %-8s %12s %12s\n" "$K" "$SNR" "$EB" "${SER:-N/A}" "${CER:-N/A}"
        else
            printf "%-4s %-6s %-8s %12s %12s\n" "$K" "$SNR" "$EB" "MISSING" "MISSING"
        fi
    done
    echo "------------------------------------------------------------"
done

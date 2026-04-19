#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# GeneFi 消融实验 — cross_loc
# 运行所有 6 个变体（M0, M1, M2, M3, M4, full）
# ═══════════════════════════════════════════════════════════════════════════

DATA="/mnt/nj-1/usr/liujia7/workspace/datasets/widar_processed_data/Processed_Widar_6AP"
EXPERIMENT="cross_loc"
MAX_EPOCH=50
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "===== GeneFi Ablation Study: ${EXPERIMENT} ====="
echo "Data path: ${DATA}"
echo "Max epoch: ${MAX_EPOCH}"
echo "Script dir: ${SCRIPT_DIR}"
echo ""

for ABL in M0 M1 M2 M3 M4 full; do
    EXP_ID="ablation_${EXPERIMENT}_${ABL}"
    LOG_FILE="${SCRIPT_DIR}/nohup_ablation_${ABL}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${ABL} → log: ${LOG_FILE}"

    nohup python "${SCRIPT_DIR}/csimain.py" \
        --dataset widar \
        --data_path "${DATA}" \
        --experiment "${EXPERIMENT}" \
        --ablation "${ABL}" \
        --exp_id "${EXP_ID}" \
        --max_epoch "${MAX_EPOCH}" \
        > "${LOG_FILE}" 2>&1 &

    echo "  PID: $!"
done

echo ""
echo "All 6 ablation experiments launched in background."
echo "Monitor with: tail -f ${SCRIPT_DIR}/nohup_ablation_*.log"
echo "Check results: cat ${SCRIPT_DIR}/log/ablation_${EXPERIMENT}_*/bestacc.txt"

#!/bin/bash
# TEA 실험 실행 스크립트
# 사용법:
#   bash run_TEA.sh            → 실험 1~8 순서대로 실행
#   bash run_TEA.sh 3          → 실험 3만 실행
#   bash run_TEA.sh 3 8        → 실험 3~8 실행
#   bash run_TEA.sh 3 3 5      → 실험 3을 epoch 5부터 재개 (크래시 복구)
#
# GPU: CUDA_VISIBLE_DEVICES 환경변수로 고정 가능 (미지정 시 실험마다 자동 선택)
#   CUDA_VISIBLE_DEVICES=1 bash run_TEA.sh

set -euo pipefail

# ── Conda 환경 활성화 ─────────────────────────────────────────────────────────
source /home/taegum/miniconda3/etc/profile.d/conda.sh
conda activate video2music

# ── 설정 ─────────────────────────────────────────────────────────────────────
FORCE_GPU=${CUDA_VISIBLE_DEVICES:-}   # 사용자가 지정했으면 고정, 아니면 매번 자동 선택
EPOCHS=100
BATCH_SIZE=4
WEIGHT_MODULUS=5
PRINT_MODULUS=50

START_EXP=${1:-1}
END_EXP=${2:-9}
RESUME_EPOCH=${3:-}

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ── 실험 루프 ─────────────────────────────────────────────────────────────────
for exp in $(seq "${START_EXP}" "${END_EXP}"); do
    SCRIPT="${PROJECT_DIR}/experiments/train_TEA_exp${exp}.py"
    LOG="${LOG_DIR}/exp${exp}.log"

    # 실험마다 GPU 재선택 (FORCE_GPU 지정 시 고정)
    if [[ -n "${FORCE_GPU}" ]]; then
        GPU=${FORCE_GPU}
    else
        GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
              | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
    fi
    export CUDA_VISIBLE_DEVICES="${GPU}"

    echo ""
    echo "========================================"
    echo "  EXP ${exp} 시작  |  GPU ${GPU}  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  로그: ${LOG}"
    echo "========================================"
    python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"없음\"}')"

    EXTRA_ARGS=""
    if [[ -n "${RESUME_EPOCH}" ]]; then
        CKPT=$(ls "${PROJECT_DIR}/saved_models/"*"_exp${exp}"/weights/epoch_$(printf '%04d' "${RESUME_EPOCH}").pickle 2>/dev/null | head -1)
        if [[ -f "${CKPT}" ]]; then
            EXTRA_ARGS="-continue_weights ${CKPT} -continue_epoch ${RESUME_EPOCH}"
            echo "  ▶ 체크포인트에서 재개: epoch ${RESUME_EPOCH}"
        else
            echo "  ⚠ 체크포인트 파일 없음, 처음부터 시작"
        fi
    fi

    python "${SCRIPT}" \
        -epochs "${EPOCHS}" \
        -batch_size "${BATCH_SIZE}" \
        -weight_modulus "${WEIGHT_MODULUS}" \
        -print_modulus "${PRINT_MODULUS}" \
        ${EXTRA_ARGS} \
        2>&1 | tee -a "${LOG}"

    echo ""
    echo "  EXP ${exp} 완료  |  $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG}"
done

echo ""
echo "모든 실험 완료."

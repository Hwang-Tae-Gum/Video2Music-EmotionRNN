#!/bin/bash
# run_experiments.sh
# 3개 실험을 순차적으로 실행하고 각 실험 후 eval을 돌린다.
# 에러가 있어도 다음 실험은 계속 진행.

cd /home/taegum/Video2Music

CONDA_RUN="conda run -n video2music --no-capture-output"
LOG="experiments_run.log"
EVAL_LOG="experiments_eval.log"
STATUS="experiments_status.txt"
MSTEA_BEST="./saved_models/MSTEA_encoder_gru/best_loss_weights.pickle"

COMMON_ARGS="-epochs 100 -rpr True -is_video True \
    -batch_size 1 -n_layers 6 -num_heads 8 -d_model 512 \
    -max_sequence_chord 300 -max_sequence_video 300 \
    --no_tensorboard"

log() { echo "$1" | tee -a "$LOG"; }
logeval() { echo "$1" | tee -a "$EVAL_LOG"; }

run_eval_only() {
    local model_name="$1"
    logeval ""
    logeval "=== eval: $model_name  $(date) ==="
    $CONDA_RUN python eval_emotion_valence.py \
        -only_model "$model_name" 2>&1 | tee -a "$EVAL_LOG"
}

log "========================================"
log "실험 시작: $(date)"
log "========================================"

# ── Exp 1: From-scratch + AlignLoss ─────────────────────────────────────────
log ""
log ">>> EXP1: MSTEA_align_scratch  (from scratch, λ=0.15)"
echo "EXP1_START=$(date)" > "$STATUS"

$CONDA_RUN python train_MSTEA_align_scratch.py \
    $COMMON_ARGS -lr 0.0001 \
    2>&1 | tee -a "$LOG"

echo "EXP1_DONE=$(date)" >> "$STATUS"
log ">>> EXP1 완료. eval 시작..."
run_eval_only "MS-TEA_align_scratch"
echo "EXP1_EVAL_DONE=$(date)" >> "$STATUS"
log ">>> EXP1 eval 완료."

# ── Exp 2: Curriculum λ scheduling ──────────────────────────────────────────
log ""
log ">>> EXP2: MSTEA_align_curriculum  (epoch37 fine-tune, λ warmup 20ep)"
echo "EXP2_START=$(date)" >> "$STATUS"

$CONDA_RUN python train_MSTEA_align_curriculum.py \
    $COMMON_ARGS -lr 1e-5 \
    -continue_weights "$MSTEA_BEST" -continue_epoch 37 \
    2>&1 | tee -a "$LOG"

echo "EXP2_DONE=$(date)" >> "$STATUS"
log ">>> EXP2 완료. eval 시작..."
run_eval_only "MS-TEA_align_curriculum"
echo "EXP2_EVAL_DONE=$(date)" >> "$STATUS"
log ">>> EXP2 eval 완료."

# ── Exp 3: Soft Margin (Hinge) AlignLoss ────────────────────────────────────
log ""
log ">>> EXP3: MSTEA_align_softmargin  (epoch37 fine-tune, MarginAlignLoss)"
echo "EXP3_START=$(date)" >> "$STATUS"

$CONDA_RUN python train_MSTEA_align_softmargin.py \
    $COMMON_ARGS -lr 1e-5 \
    -continue_weights "$MSTEA_BEST" -continue_epoch 37 \
    2>&1 | tee -a "$LOG"

echo "EXP3_DONE=$(date)" >> "$STATUS"
log ">>> EXP3 완료. eval 시작..."
run_eval_only "MS-TEA_align_softmargin"
echo "EXP3_EVAL_DONE=$(date)" >> "$STATUS"
log ">>> EXP3 eval 완료."

# ── 전체 eval (최종 정리) ────────────────────────────────────────────────────
logeval ""
logeval "=== FULL EVAL $(date) ==="
$CONDA_RUN python eval_emotion_valence.py 2>&1 | tee -a "$EVAL_LOG"

log ""
log "========================================"
log "전체 완료: $(date)"
log "========================================"
echo "ALL_DONE=$(date)" >> "$STATUS"

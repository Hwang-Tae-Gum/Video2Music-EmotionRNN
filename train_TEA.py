"""
train_TEA.py
────────────
TEA(Temporal Emotion Adapter)가 주입된 VideoMusicTransformerTEA 학습 스크립트.

기존 train.py 에서 변경된 사항:
  - VideoMusicTransformerTEA 사용
  - constants_TEA 에서 TEA 하이퍼파라미터 로드
  - version = "TEA_" + TEA_WHERE + "_" + TEA_ENCODER_CELL
  - Adam optimizer 에 weight_decay=1e-4 추가 (과적합 방지)
  - Train-set evaluation 제거 → Val-only evaluation (속도 향상)
  - use_cuda(True) else 블록 추가
"""

import os
import csv
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset.vevo_dataset import create_vevo_datasets
from model.video_music_transformer_TEA import VideoMusicTransformerTEA
from model.loss import SmoothCrossEntropyLoss, SoftCorrLoss, build_chord_similarity_matrix, build_valence_vector, EmotionValenceAlignLoss

from utilities.constants import *
from utilities.constants_TEA import (
    TEA_WHERE, TEA_ENCODER_CELL, TEA_DECODER_CELL, TEA_NUM_LAYERS, LAMBDA_ALIGN,
)
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.argument_funcs import parse_train_args, print_train_args, write_model_params
from utilities.run_model_vevo import train_epoch, eval_model

# ── 버전 문자열 ──────────────────────────────────────────────────────────────
_align_suffix = "_align" if LAMBDA_ALIGN > 0 else ""
if TEA_WHERE == "decoder":
    version = f"TEA_decoder_{TEA_DECODER_CELL}{_align_suffix}"
elif TEA_WHERE == "encoder":
    version = f"TEA_encoder_{TEA_ENCODER_CELL}{_align_suffix}"
else:
    version = f"TEA_both_{TEA_ENCODER_CELL}_{TEA_DECODER_CELL}{_align_suffix}"
split_ver = SPLIT_VER
split_path = "split_" + split_ver

CSV_HEADER = [
    "Epoch", "Learn rate",
    "Avg Val loss (total)", "Avg Val loss (chord)", "Avg Val loss (emotion)",
    "Avg Val H@1", "Avg Val H@3", "Avg Val H@5",
]

BASELINE_EPOCH = -1

VIS_MODELS_ARR = ["2d/clip_l14p"]


def main(vm="", isPrintArgs=True):
    args = parse_train_args()

    if isPrintArgs:
        print_train_args(args)
    if vm != "":
        args.vis_models = vm

    # ── CUDA 설정 ─────────────────────────────────────────────────────────
    if args.force_cpu:
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower\n")
    else:
        use_cuda(True)
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}  (CUDA {torch.version.cuda})\n")
        else:
            print("WARNING: CUDA not available, running on CPU\n")

    # ── vis 경로 ──────────────────────────────────────────────────────────
    if args.is_video:
        vis_arr = args.vis_models.split(" ")
        vis_arr.sort()
        vis_abbr_path = ""
        for v in vis_arr:
            vis_abbr_path += "_" + VIS_ABBR_DIC[v]
        vis_abbr_path = vis_abbr_path[1:]
    else:
        vis_abbr_path = "no_video"

    # ── 출력 디렉토리 ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, version), exist_ok=True)

    params_file   = os.path.join(args.output_dir, version, "model_params.txt")
    write_model_params(args, params_file)

    weights_folder = os.path.join(args.output_dir, version, "weights")
    os.makedirs(weights_folder, exist_ok=True)

    results_folder = os.path.join(args.output_dir, version)
    results_file   = os.path.join(results_folder, "results.csv")
    best_loss_file = os.path.join(results_folder, "best_loss_weights.pickle")
    best_text      = os.path.join(results_folder, "best_epochs.txt")

    # ── Tensorboard ───────────────────────────────────────────────────────
    if args.no_tensorboard:
        tensorboard_summary = None
    else:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, version, "tensorboard")
        tensorboard_summary = SummaryWriter(log_dir=tb_dir)

    # ── 데이터셋 ──────────────────────────────────────────────────────────
    train_dataset, val_dataset, _ = create_vevo_datasets(
        dataset_root    = "./dataset/",
        max_seq_chord   = args.max_sequence_chord,
        max_seq_video   = args.max_sequence_video,
        vis_models      = args.vis_models,
        emo_model       = args.emo_model,
        split_ver       = SPLIT_VER,
        random_seq      = True,
        is_video        = args.is_video,
    )

    total_vf_dim = 0
    if args.is_video:
        for vf in train_dataset[0]["semanticList"]:
            total_vf_dim += vf.shape[1]
        total_vf_dim += 1  # scene_offset
        total_vf_dim += 1  # motion
        # emotion은 Linear_emo 전용 pathway (total_vf_dim 에서 제외)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.n_workers, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.n_workers,
    )

    # ── 모델 ──────────────────────────────────────────────────────────────
    model = VideoMusicTransformerTEA(
        n_layers        = args.n_layers,
        num_heads       = args.num_heads,
        d_model         = args.d_model,
        dim_feedforward = args.dim_feedforward,
        dropout         = args.dropout,
        max_sequence_midi  = args.max_sequence_midi,
        max_sequence_video = args.max_sequence_video,
        max_sequence_chord = args.max_sequence_chord,
        total_vf_dim    = total_vf_dim,
        rpr             = args.rpr,
        tea_where       = TEA_WHERE,
        tea_num_layers  = TEA_NUM_LAYERS,
    ).to(get_device())

    # ── 체크포인트 이어받기 ───────────────────────────────────────────────
    start_epoch = BASELINE_EPOCH
    if args.continue_weights is not None:
        if args.continue_epoch is None:
            raise ValueError("Need -continue_epoch when using -continue_weights")
        model.load_state_dict(torch.load(args.continue_weights))
        start_epoch = args.continue_epoch
    elif args.continue_epoch is not None:
        raise ValueError("Need -continue_weights when using -continue_epoch")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    # ── LR 스케줄러 ───────────────────────────────────────────────────────
    if args.lr is None:
        init_step  = 0 if args.continue_epoch is None else args.continue_epoch * len(train_loader)
        lr         = LR_DEFAULT_START
        lr_stepper = LrStepTracker(args.d_model, SCHEDULER_WARMUP_STEPS, init_step)
    else:
        lr = args.lr

    # ── Loss 함수 ─────────────────────────────────────────────────────────
    eval_loss_func  = nn.CrossEntropyLoss(ignore_index=CHORD_PAD)
    if getattr(args, 'soft_corr', False):
        sim_matrix = build_chord_similarity_matrix(
            CHORD_SIZE, CHORD_END, device=get_device()
        )
        train_loss_func = SoftCorrLoss(sim_matrix, ignore_index=CHORD_PAD)
        print("Loss: SoftCorrLoss (harmonic similarity-based soft targets)")
    elif args.ce_smoothing is not None:
        train_loss_func = SmoothCrossEntropyLoss(
            args.ce_smoothing, CHORD_SIZE, ignore_index=CHORD_PAD
        )
        print(f"Loss: SmoothCrossEntropyLoss (smoothing={args.ce_smoothing})")
    else:
        train_loss_func = eval_loss_func
        print("Loss: CrossEntropyLoss (hard targets)")
    eval_loss_emotion_func  = nn.BCEWithLogitsLoss()
    train_loss_emotion_func = eval_loss_emotion_func

    # ── Emotion-Valence Alignment Loss ────────────────────────────────────
    if LAMBDA_ALIGN > 0:
        with open("./dataset/vevo_meta/chord_inv.json")  as f: _chord_inv  = json.load(f)
        with open("./dataset/vevo_meta/chord_attr.json") as f: _chord_attr = json.load(f)
        _valence_vec = build_valence_vector(CHORD_SIZE, CHORD_END, _chord_inv, _chord_attr)
        align_loss_func = EmotionValenceAlignLoss(_valence_vec, ignore_index=CHORD_PAD).to(get_device())
        print(f"EmotionValenceAlignLoss: λ_align={LAMBDA_ALIGN}")
    else:
        align_loss_func = None

    # ── Optimizer ────────────────────────────────────────────────────────
    opt = Adam(
        model.parameters(),
        lr    = lr,
        betas = (ADAM_BETA_1, ADAM_BETA_2),
        eps   = ADAM_EPSILON,
        weight_decay = 1e-4,          # 과적합 방지
    )
    lr_scheduler = (
        LambdaLR(opt, lr_stepper.step) if args.lr is None else None
    )

    # ── 추적 변수 ─────────────────────────────────────────────────────────
    EARLY_STOP_PATIENCE  = 20
    best_eval_loss       = float("inf")
    best_eval_loss_epoch = -1
    no_improve_cnt       = 0

    if not os.path.isfile(results_file):
        with open(results_file, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    # ════════════════════════════════════════════════════════════════════════
    #  TRAIN LOOP
    # ════════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, args.epochs):
        if epoch > BASELINE_EPOCH:
            print(SEPERATOR)
            print(f"NEW EPOCH: {epoch + 1}")
            print(SEPERATOR)

            train_epoch(
                epoch + 1, model, train_loader,
                train_loss_func, train_loss_emotion_func,
                opt, lr_scheduler, args.print_modulus,
                isVideo = args.is_video,
                max_grad_norm = 1.0,
                align_loss_func = align_loss_func,
                lambda_align    = LAMBDA_ALIGN,
            )
            print(SEPERATOR)
            print("Evaluating (val only):")
        else:
            print(SEPERATOR)
            print("Baseline model evaluation (Epoch 0):")

        # ── Val evaluation only (train eval 제거) ────────────────────────
        val_metrics = eval_model(
            model, val_loader,
            eval_loss_func, eval_loss_emotion_func,
            isVideo = args.is_video,
        )
        val_total   = val_metrics["avg_total_loss"]
        val_chord   = val_metrics["avg_loss_chord"]
        val_emotion = val_metrics["avg_loss_emotion"]
        val_h1      = val_metrics["avg_h1"]
        val_h3      = val_metrics["avg_h3"]
        val_h5      = val_metrics["avg_h5"]

        lr_cur = get_lr(opt)

        print(f"Epoch: {epoch + 1}")
        print(f"Avg val loss (total):   {val_total:.4f}")
        print(f"Avg val loss (chord):   {val_chord:.4f}")
        print(f"Avg val loss (emotion): {val_emotion:.4f}")
        print(f"Avg val H@1: {val_h1:.4f}  H@3: {val_h3:.4f}  H@5: {val_h5:.4f}")
        print(SEPERATOR)
        print()

        # ── Best model 저장 ───────────────────────────────────────────────
        new_best = False
        if val_total < best_eval_loss:
            best_eval_loss       = val_total
            best_eval_loss_epoch = epoch + 1
            no_improve_cnt       = 0
            torch.save(model.state_dict(), best_loss_file)
            new_best = True
        else:
            no_improve_cnt += 1
            print(f"  no improve {no_improve_cnt}/{EARLY_STOP_PATIENCE}")

        if new_best:
            with open(best_text, "w") as f:
                print(f"Best val loss epoch: {best_eval_loss_epoch}", file=f)
                print(f"Best val loss:       {best_eval_loss}",       file=f)

        # ── Tensorboard ───────────────────────────────────────────────────
        if not args.no_tensorboard and tensorboard_summary is not None:
            tensorboard_summary.add_scalar("Val/loss_total",   val_total,   epoch + 1)
            tensorboard_summary.add_scalar("Val/loss_chord",   val_chord,   epoch + 1)
            tensorboard_summary.add_scalar("Val/loss_emotion", val_emotion, epoch + 1)
            tensorboard_summary.add_scalar("Val/H1",           val_h1,      epoch + 1)
            tensorboard_summary.add_scalar("Val/H3",           val_h3,      epoch + 1)
            tensorboard_summary.add_scalar("Val/H5",           val_h5,      epoch + 1)
            tensorboard_summary.add_scalar("LR",               lr_cur,      epoch + 1)
            tensorboard_summary.flush()

        # ── 주기적 weight 저장 ────────────────────────────────────────────
        if (epoch + 1) % args.weight_modulus == 0:
            path = os.path.join(
                weights_folder,
                "epoch_" + str(epoch + 1).zfill(PREPEND_ZEROS_WIDTH) + ".pickle",
            )
            torch.save(model.state_dict(), path)

        # ── CSV 기록 ──────────────────────────────────────────────────────
        with open(results_file, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, lr_cur,
                val_total, val_chord, val_emotion,
                val_h1, val_h3, val_h5,
            ])

        # ── Early stopping ────────────────────────────────────────────────
        if no_improve_cnt >= EARLY_STOP_PATIENCE:
            print(f"\n[Early Stop] {EARLY_STOP_PATIENCE} epochs no improvement → stop.")
            print(f"Best: epoch {best_eval_loss_epoch}, val={best_eval_loss:.4f}")
            break

    if not args.no_tensorboard and tensorboard_summary is not None:
        tensorboard_summary.flush()


if __name__ == "__main__":
    if VIS_MODELS_ARR:
        for vm in VIS_MODELS_ARR:
            main(vm, False)
    else:
        main()
"""
train_TEA_contrastive.py
─────────────────────────
기존 TEA 체크포인트에서 Supervised Contrastive Emotion Loss 를 추가해 파인튜닝.

총 손실:
  L = λ_chord · L_chord  +  λ_emo · L_emotion  +  λ_con · L_contrastive

사용법 (예: Exp3 기반 파인튜닝):
  python train_TEA_contrastive.py --base_exp 3 --epochs 50

실험 설정:
  --base_exp 3  →  TEA_encoder_lstm_exp3  →  저장: TEA_encoder_lstm_exp3_con
  --base_exp 4  →  TEA_encoder_gru_exp4   →  저장: TEA_encoder_gru_exp4_con
  --base_exp 8  →  TEA_both_gru_lstm_exp8 →  저장: TEA_both_gru_lstm_exp8_con
"""

import os
import csv
import argparse
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset.vevo_dataset import create_vevo_datasets
from model.video_music_transformer_TEA import VideoMusicTransformerTEA
from utilities.run_model_vevo import eval_model
from utilities.lr_scheduling import LrStepTracker, get_lr

import utilities.constants_TEA as cTEA
import model.video_music_transformer_TEA as _tea_module
from utilities.constants import *
from utilities.constants_TEA import CONTRASTIVE_LAMBDA
from utilities.device import get_device, use_cuda

# ── 실험 설정 테이블 ──────────────────────────────────────────────────────────
EXP_CONFIGS = {
    3: dict(where="encoder", enc="lstm", dec="gru",  dir="TEA_encoder_lstm_exp3"),
    4: dict(where="encoder", enc="gru",  dec="gru",  dir="TEA_encoder_gru_exp4"),
    8: dict(where="both",    enc="gru",  dec="lstm", dir="TEA_both_gru_lstm_exp8"),
}

VIS_MODELS = "2d/clip_l14p"
EMO_MODEL  = "6c_l14p"
SAVED_ROOT = "./saved_models"
BATCH_SIZE = 4
N_WORKERS  = 2
D_MODEL    = 512


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_exp",  type=int, required=True,
                   help="파인튜닝 기반 실험 번호 (3, 4, 8)")
    p.add_argument("--epochs",    type=int, default=50)
    p.add_argument("--lr",        type=float, default=1e-4,
                   help="고정 LR (None 이면 warmup 스케줄)")
    p.add_argument("--con_lambda", type=float, default=CONTRASTIVE_LAMBDA,
                   help="Contrastive loss 가중치")
    p.add_argument("--no_tensorboard", action="store_true")
    return p.parse_args()


def build_model(cfg, total_vf_dim):
    cTEA.TEA_WHERE        = cfg["where"]
    cTEA.TEA_ENCODER_CELL = cfg["enc"]
    cTEA.TEA_DECODER_CELL = cfg["dec"]
    _tea_module.TEA_WHERE        = cfg["where"]
    _tea_module.TEA_ENCODER_CELL = cfg["enc"]
    _tea_module.TEA_DECODER_CELL = cfg["dec"]
    return VideoMusicTransformerTEA(
        n_layers=6, num_heads=8, d_model=D_MODEL, dim_feedforward=1024,
        dropout=0.1, max_sequence_midi=2048, max_sequence_video=300,
        max_sequence_chord=300, total_vf_dim=total_vf_dim, rpr=RPR,
        tea_where=cfg["where"], tea_num_layers=cTEA.TEA_NUM_LAYERS,
    ).to(get_device())


def train_epoch_con(
    epoch, model, dataloader,
    chord_loss_fn, emo_loss_fn, con_loss_fn,
    opt, lr_scheduler,
    con_lambda, print_modulus=10,
):
    model.train()
    for batch_num, batch in enumerate(dataloader):
        t0 = time.time()
        opt.zero_grad()

        x            = batch["x"].to(get_device())
        tgt          = batch["tgt"].to(get_device())
        x_root       = batch["x_root"].to(get_device())
        x_attr       = batch["x_attr"].to(get_device())
        tgt_emotion  = batch["tgt_emotion"].to(get_device())
        sem_list     = [f.to(get_device()) for f in batch["semanticList"]]
        feature_key  = batch["key"].to(get_device())
        scene_offset = batch["scene_offset"].to(get_device())
        motion       = batch["motion"].to(get_device())
        emotion      = batch["emotion"].to(get_device())

        # ── forward ──────────────────────────────────────────────────────
        y = model(x, x_root, x_attr, sem_list, feature_key,
                  scene_offset, motion, emotion)

        # ── chord loss + emotion loss ────────────────────────────────────
        y_flat   = y.reshape(y.shape[0] * y.shape[1], -1)
        tgt_flat = tgt.flatten()
        tgt_emo_flat = tgt_emotion.reshape(-1, tgt_emotion.shape[-1])

        l_chord = chord_loss_fn(y_flat, tgt_flat)
        l_emo   = emo_loss_fn(y_flat, tgt_emo_flat)

        # ── contrastive loss ─────────────────────────────────────────────
        # 지배 감정 클래스: 비디오 전체 emotion prob 평균의 argmax
        emo_labels = emotion.mean(dim=1).argmax(dim=-1)   # (B,)
        emo_emb    = model.get_emotion_emb(emotion)        # (B, proj_dim)
        l_con      = con_loss_fn(model.emotion_clf(emo_emb), emo_labels)

        total = (LOSS_LAMBDA * l_chord
                 + (1 - LOSS_LAMBDA) * l_emo
                 + con_lambda * l_con)

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if (batch_num + 1) % print_modulus == 0:
            print(f"  Epoch {epoch} Batch {batch_num+1}/{len(dataloader)}"
                  f" | chord={float(l_chord):.4f}"
                  f" | emo={float(l_emo):.4f}"
                  f" | con={float(l_con):.4f}"
                  f" | total={float(total):.4f}"
                  f" | {time.time()-t0:.1f}s")


def main():
    args = parse_args()
    use_cuda(True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    cfg = EXP_CONFIGS.get(args.base_exp)
    if cfg is None:
        raise ValueError(f"지원 실험: {list(EXP_CONFIGS.keys())}, 입력: {args.base_exp}")

    # ── 저장 경로 ─────────────────────────────────────────────────────────
    base_dir = os.path.join(SAVED_ROOT, cfg["dir"])
    save_dir = os.path.join(SAVED_ROOT, cfg["dir"] + "_con")
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt    = os.path.join(save_dir, "best_loss_weights.pickle")
    results_file = os.path.join(save_dir, "results.csv")
    best_txt     = os.path.join(save_dir, "best_epochs.txt")

    # ── 데이터셋 ──────────────────────────────────────────────────────────
    train_dataset, val_dataset, _ = create_vevo_datasets(
        dataset_root="./dataset/", max_seq_chord=300, max_seq_video=300,
        vis_models=VIS_MODELS, emo_model=EMO_MODEL,
        split_ver=SPLIT_VER, random_seq=True, is_video=True,
    )
    total_vf_dim  = sum(vf.shape[1] for vf in train_dataset[0]["semanticList"])
    total_vf_dim += 1 + 1  # scene_offset + motion (emotion은 Linear_emo 전용 pathway)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              num_workers=N_WORKERS, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              num_workers=N_WORKERS)

    # ── 모델 & 체크포인트 로드 ────────────────────────────────────────────
    model = build_model(cfg, total_vf_dim)
    base_ckpt = os.path.join(base_dir, "best_loss_weights.pickle")
    if os.path.isfile(base_ckpt):
        missing, unexpected = model.load_state_dict(
            torch.load(base_ckpt, map_location=get_device()), strict=False
        )
        print(f"Base ckpt loaded: {base_ckpt}")
        if missing:
            print(f"  Missing keys (new): {missing}")
    else:
        print(f"WARNING: base checkpoint not found → 처음부터 학습")

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_param:,}\n")

    # ── Loss 함수 ─────────────────────────────────────────────────────────
    chord_loss_fn = nn.CrossEntropyLoss(ignore_index=CHORD_PAD)
    emo_loss_fn   = nn.BCEWithLogitsLoss()
    con_loss_fn   = nn.CrossEntropyLoss()

    # ── Optimizer (contrastive_proj 파라미터만 더 큰 LR 가능) ────────────
    opt = Adam(model.parameters(), lr=args.lr,
               betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON,
               weight_decay=1e-4)
    lr_scheduler = None  # 파인튜닝은 고정 LR

    # ── Tensorboard ───────────────────────────────────────────────────────
    tb = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))
        except ImportError:
            pass

    # ── CSV 헤더 ──────────────────────────────────────────────────────────
    CSV_HEADER = ["Epoch", "LR", "Val Loss Total", "Val Loss Chord",
                  "Val Loss Emotion", "H@1", "H@3", "H@5"]
    if not os.path.isfile(results_file):
        with open(results_file, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    best_loss = float("inf")

    # ════════════════════════════════════════════════════════════════════════
    #  TRAIN LOOP
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print(f"Fine-tuning Exp{args.base_exp} + SupConLoss  →  {save_dir}")
    print(f"  con_lambda={args.con_lambda}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        print(f"\n── Epoch {epoch}/{args.epochs} ──")
        train_epoch_con(
            epoch, model, train_loader,
            chord_loss_fn, emo_loss_fn, con_loss_fn,
            opt, lr_scheduler,
            con_lambda=args.con_lambda,
            print_modulus=20,
        )

        # ── Validation ───────────────────────────────────────────────────
        val_metrics = eval_model(model, val_loader,
                                 chord_loss_fn, emo_loss_fn, isVideo=True)
        vt  = val_metrics["avg_total_loss"]
        vc  = val_metrics["avg_loss_chord"]
        ve  = val_metrics["avg_loss_emotion"]
        h1  = val_metrics["avg_h1"]
        h3  = val_metrics["avg_h3"]
        h5  = val_metrics["avg_h5"]
        lr_ = get_lr(opt)

        print(f"  Val total={vt:.4f}  chord={vc:.4f}  emo={ve:.4f}"
              f"  H@1={h1:.4f}  H@3={h3:.4f}  H@5={h5:.4f}")

        # ── Best 저장 ─────────────────────────────────────────────────────
        if vt < best_loss:
            best_loss = vt
            torch.save(model.state_dict(), best_ckpt)
            with open(best_txt, "w") as f:
                print(f"Best epoch: {epoch}  loss: {best_loss:.6f}", file=f)
            print(f"  ★ Best model saved (epoch {epoch})")

        # ── Tensorboard ───────────────────────────────────────────────────
        if tb is not None:
            tb.add_scalar("Val/total",   vt, epoch)
            tb.add_scalar("Val/chord",   vc, epoch)
            tb.add_scalar("Val/emotion", ve, epoch)
            tb.add_scalar("Val/H1",      h1, epoch)
            tb.add_scalar("Val/H3",      h3, epoch)
            tb.add_scalar("Val/H5",      h5, epoch)
            tb.flush()

        # ── CSV ───────────────────────────────────────────────────────────
        with open(results_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch, lr_, vt, vc, ve, h1, h3, h5])

    if tb is not None:
        tb.close()

    print(f"\n완료. 최고 모델: {best_ckpt}")
    print(f"Best val loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()

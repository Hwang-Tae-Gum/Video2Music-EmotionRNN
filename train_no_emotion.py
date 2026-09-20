"""
train_no_emotion.py
───────────────────
No-Emotion ablation: emotion feature 를 0 으로 masking 하고 VideoMusicTransformer 를 학습.

목적: emotion 을 완전히 제거했을 때의 Corr 기준선을 만들기 위한 ablation 모델.
     비교 테이블:
       Baseline (no emotion) < Baseline (full) < TEA Exp4 / Exp8
     → emotion 의 기여 + TEA 의 추가 기여를 분리해서 보여줌

출력: ./saved_models/AMT_no_emotion/
사용: python train_no_emotion.py
"""

import os, csv
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from dataset.vevo_dataset import create_vevo_datasets
from model.video_music_transformer import VideoMusicTransformer
from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.run_model_vevo import train_epoch, eval_model


class NoEmoWrapper(Dataset):
    """emotion 만 0 tensor 로 교체. semantic / scene_offset / motion 은 그대로."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        s = dict(self.dataset[idx])
        s["emotion"] = torch.zeros_like(s["emotion"])
        return s


OUTPUT_DIR  = "./saved_models/AMT_no_emotion"
VIS_MODELS  = "2d/clip_l14p"
EMO_MODEL   = "6c_l14p"
EPOCHS      = 100
BATCH_SIZE  = 4
N_WORKERS   = 2

CSV_HEADER = ["Epoch", "LR",
              "Train total", "Train chord", "Train emotion",
              "Val total",   "Val chord",   "Val emotion",
              "Val H@1",     "Val Corr"]

BASELINE_EPOCH = -1


def main():
    use_cuda(True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds, val_ds, _ = create_vevo_datasets(
        dataset_root  = "./dataset/",
        max_seq_chord = 300,
        max_seq_video = 300,
        vis_models    = VIS_MODELS,
        emo_model     = EMO_MODEL,
        split_ver     = SPLIT_VER,
        random_seq    = True,
        is_video      = True,
    )

    train_ds = NoEmoWrapper(train_ds)
    val_ds   = NoEmoWrapper(val_ds)

    total_vf_dim  = sum(vf.shape[1] for vf in train_ds.dataset[0]["semanticList"])
    total_vf_dim += 1 + 1  # scene_offset + motion (emotion은 Linear_emo 전용 pathway)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              num_workers=N_WORKERS, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              num_workers=N_WORKERS)

    model = VideoMusicTransformer(
        n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
        max_sequence_midi=2048, max_sequence_video=300, max_sequence_chord=300,
        total_vf_dim=total_vf_dim, rpr=RPR,
    ).to(get_device())
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}\n")

    eval_loss_func          = nn.CrossEntropyLoss(ignore_index=CHORD_PAD)
    train_loss_func         = eval_loss_func
    eval_loss_emotion_func  = nn.BCEWithLogitsLoss()
    train_loss_emotion_func = eval_loss_emotion_func

    lr_stepper   = LrStepTracker(512, SCHEDULER_WARMUP_STEPS, 0)
    opt          = Adam(model.parameters(), lr=LR_DEFAULT_START,
                        betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
    lr_scheduler = LambdaLR(opt, lr_stepper.step)

    results_file   = os.path.join(OUTPUT_DIR, "results.csv")
    best_loss_file = os.path.join(OUTPUT_DIR, "best_loss_weights.pickle")
    best_text      = os.path.join(OUTPUT_DIR, "best_epochs.txt")

    if not os.path.isfile(results_file):
        with open(results_file, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    best_eval_loss = float("inf")

    for epoch in range(BASELINE_EPOCH, EPOCHS):
        if epoch > BASELINE_EPOCH:
            print(f"\n{'='*60}\nEpoch {epoch+1}\n{'='*60}")
            train_epoch(epoch+1, model, train_loader,
                        train_loss_func, train_loss_emotion_func,
                        opt, lr_scheduler, print_modulus=100, isVideo=True)

        tr = eval_model(model, train_loader,
                        train_loss_func, train_loss_emotion_func, isVideo=True)
        vl = eval_model(model, val_loader,
                        eval_loss_func,  eval_loss_emotion_func,  isVideo=True)

        lr = get_lr(opt)
        print(f"Epoch {epoch+1:3d} | lr={lr:.2e} | "
              f"train={tr['avg_total_loss']:.4f} | val={vl['avg_total_loss']:.4f} | "
              f"H@1={vl['avg_h1']:.4f} | Corr={vl['avg_cor']:.4f}")

        if vl["avg_total_loss"] < best_eval_loss:
            best_eval_loss = vl["avg_total_loss"]
            torch.save(model.state_dict(), best_loss_file)
            with open(best_text, "w") as f:
                print(f"Best epoch: {epoch+1}, val loss: {best_eval_loss:.4f}", file=f)
            print(f"  → best model saved (epoch {epoch+1})")

        with open(results_file, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1, lr,
                tr["avg_total_loss"], tr["avg_loss_chord"], tr["avg_loss_emotion"],
                vl["avg_total_loss"], vl["avg_loss_chord"], vl["avg_loss_emotion"],
                vl["avg_h1"], vl["avg_cor"],
            ])


if __name__ == "__main__":
    main()

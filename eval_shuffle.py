"""
eval_shuffle.py
───────────────
Emotion Ablation: 4개 모델의 val Corr 을 직접 비교.

비교 테이블:
  Baseline (no emotion) → Baseline (full) → TEA Exp4 → TEA Exp8

해석:
  no_emotion → full  : emotion feature 자체의 기여
  full → TEA         : temporal emotion modeling (TEA) 의 추가 기여

사용법:
  python eval_shuffle.py
출력:
  ./experiments/eval_ablation.csv
"""

import os, csv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.vevo_dataset import (
    create_vevo_datasets,
    compute_vevo_correspondence,
)
from model.video_music_transformer import VideoMusicTransformer
from model.video_music_transformer_TEA import VideoMusicTransformerTEA

import utilities.constants_TEA as cTEA
import model.video_music_transformer_TEA as _tea_module
from utilities.constants import *
from utilities.device import get_device, use_cuda

# ── 설정 ─────────────────────────────────────────────────────────────────────
VIS_MODELS  = "2d/clip_l14p"
EMO_MODEL   = "6c_l14p"
SAVED_ROOT  = "./saved_models"
OUTPUT_CSV  = "./experiments/eval_ablation.csv"
BATCH_SIZE  = 4

MODELS = [
    dict(name="Baseline (no emotion)",
         type="base", no_emo=True,
         ckpt=f"{SAVED_ROOT}/AMT_no_emotion/best_loss_weights.pickle"),
    dict(name="Baseline (full)",
         type="base", no_emo=False,
         ckpt=f"{SAVED_ROOT}/AMT_full/best_loss_weights.pickle"),
    dict(name="TEA Exp4",
         type="tea",  no_emo=False,
         where="encoder", enc="gru", dec="gru",
         ckpt=f"{SAVED_ROOT}/TEA_encoder_gru_exp4/best_loss_weights.pickle"),
    dict(name="TEA Exp8",
         type="tea",  no_emo=False,
         where="both", enc="gru", dec="lstm",
         ckpt=f"{SAVED_ROOT}/TEA_both_gru_lstm_exp8/best_loss_weights.pickle"),
]


# ── No-emotion wrapper ────────────────────────────────────────────────────────
class NoEmoWrapper(Dataset):
    """emotion 만 0 으로 교체 — 학습 시와 동일한 입력 조건 재현."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        s = dict(self.dataset[idx])
        s["emotion"] = torch.zeros_like(s["emotion"])
        return s


# ── 모델 생성 ─────────────────────────────────────────────────────────────────
def build_model(cfg, total_vf_dim):
    if cfg["type"] == "base":
        return VideoMusicTransformer(
            n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
            max_sequence_midi=2048, max_sequence_video=300, max_sequence_chord=300,
            total_vf_dim=total_vf_dim, rpr=RPR,
        ).to(get_device())
    else:
        cTEA.TEA_WHERE        = _tea_module.TEA_WHERE        = cfg["where"]
        cTEA.TEA_ENCODER_CELL = _tea_module.TEA_ENCODER_CELL = cfg["enc"]
        cTEA.TEA_DECODER_CELL = _tea_module.TEA_DECODER_CELL = cfg["dec"]
        return VideoMusicTransformerTEA(
            n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
            max_sequence_midi=2048, max_sequence_video=300, max_sequence_chord=300,
            total_vf_dim=total_vf_dim, rpr=RPR,
            tea_where=cfg["where"], tea_num_layers=cTEA.TEA_NUM_LAYERS,
        ).to(get_device())


# ── Corr 측정 ─────────────────────────────────────────────────────────────────
def eval_corr(model, loader):
    model.eval()
    sum_cor, n_cor = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            x        = batch["x"].to(get_device())
            tgt      = batch["tgt"].to(get_device())
            x_root   = batch["x_root"].to(get_device())
            x_attr   = batch["x_attr"].to(get_device())
            tgt_emotion      = batch["tgt_emotion"].to(get_device())
            tgt_emotion_prob = batch["tgt_emotion_prob"].to(get_device())

            feat_sem   = [f.to(get_device()) for f in batch["semanticList"]]
            feat_key   = batch["key"].to(get_device())
            feat_scene = batch["scene_offset"].to(get_device())
            feat_mot   = batch["motion"].to(get_device())
            feat_emo   = batch["emotion"].to(get_device())

            y = model(x, x_root, x_attr,
                      feat_sem, feat_key, feat_scene, feat_mot, feat_emo)

            _emo_flat  = tgt_emotion.reshape(-1, tgt_emotion.shape[-1])
            _prob_flat = tgt_emotion_prob.reshape(-1)
            cor = float(compute_vevo_correspondence(
                y, tgt, _emo_flat, _prob_flat, EMOTION_THRESHOLD))
            if cor >= 0:
                sum_cor += cor
                n_cor   += 1

    return sum_cor / n_cor if n_cor > 0 else -1.0


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    use_cuda(True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    _, val_ds, _ = create_vevo_datasets(
        dataset_root  = "./dataset/",
        max_seq_chord = 300,
        max_seq_video = 300,
        vis_models    = VIS_MODELS,
        emo_model     = EMO_MODEL,
        split_ver     = SPLIT_VER,
        random_seq    = False,
        is_video      = True,
    )
    total_vf_dim  = sum(vf.shape[1] for vf in val_ds[0]["semanticList"])
    total_vf_dim += 1 + 1  # scene_offset + motion (emotion은 Linear_emo 전용 pathway)

    val_ds_noemo = NoEmoWrapper(val_ds)

    header = ["Model", "Val Corr", "ΔCorr vs no-emotion"]
    rows   = []
    corrs  = {}

    print(f"\n{'Model':<26} {'Val Corr':>10} {'ΔCorr vs no-emo':>17}")
    print("=" * 57)

    for cfg in MODELS:
        if not os.path.isfile(cfg["ckpt"]):
            print(f"  SKIP {cfg['name']:<22} — 체크포인트 없음")
            continue

        model = build_model(cfg, total_vf_dim)
        model.load_state_dict(
            torch.load(cfg["ckpt"], map_location=get_device()), strict=False)

        ds  = val_ds_noemo if cfg["no_emo"] else val_ds
        ldr = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=2)

        corr = eval_corr(model, ldr)
        corrs[cfg["name"]] = corr

        baseline_noemo = corrs.get("Baseline (no emotion)", None)
        delta = f"{corr - baseline_noemo:+.4f}" if baseline_noemo is not None else "—"

        print(f"  {cfg['name']:<24} {corr:>10.4f} {delta:>17}")
        rows.append([cfg["name"], f"{corr:.4f}", delta])

        del model
        torch.cuda.empty_cache()

    print("=" * 57)

    os.makedirs("experiments", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n결과 저장: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

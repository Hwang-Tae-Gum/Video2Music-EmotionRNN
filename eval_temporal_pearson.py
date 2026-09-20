"""
eval_temporal_pearson.py
────────────────────────
Temporal Emotion-Music Pearson Correlation 평가.

각 시퀀스에서:
  - video_valence[t]: tgt_emotion_quality 에서 positive/negative chord quality 기대값 → 스칼라 valence
  - music_valence[t]: 예측 코드의 quality → 스칼라 valence
  두 시퀀스 간 Pearson r 을 계산하고, 전체 val set 에서 평균.

기존 Corr 과 차이:
  - Corr: 각 timestep 의 point-wise hit/miss 평균
  - Temporal Pearson: 시퀀스 전체의 감정-음악 궤적이 얼마나 같이 움직이는가
  → TEA (Temporal Emotion Adapter) 의 시간적 감정 추적 능력을 직접 측정

사용법:
  python eval_temporal_pearson.py
  python eval_temporal_pearson.py --min_valid 10
"""

import os, csv, argparse, json
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

from dataset.vevo_dataset import create_vevo_datasets
from model.video_music_transformer import VideoMusicTransformer
from model.video_music_transformer_TEA import VideoMusicTransformerTEA

import utilities.constants_TEA as cTEA
import model.video_music_transformer_TEA as _tea_module
from utilities.constants import *
from utilities.device import get_device, use_cuda

# ── 설정 ──────────────────────────────────────────────────────────────────────
VIS_MODELS = "2d/clip_l14p"
EMO_MODEL  = "6c_l14p"
SAVED_ROOT = "./saved_models"
OUTPUT_CSV = "./experiments/eval_temporal_pearson.csv"
BATCH_SIZE = 1

MODELS = [
    dict(name="No Emotion",
         type="base", no_emo=True,
         ckpt=f"{SAVED_ROOT}/AMT_no_emotion/best_loss_weights.pickle"),
    dict(name="Full",
         type="base", no_emo=False,
         ckpt=f"{SAVED_ROOT}/AMT_full/best_loss_weights.pickle"),
    dict(name="TEA_enc_gru",
         type="tea", no_emo=False,
         where="encoder", enc="gru", dec="gru",
         ckpt=f"{SAVED_ROOT}/TEA_encoder_gru/best_loss_weights.pickle"),
    dict(name="TEA_enc_gru_align",
         type="tea", no_emo=False,
         where="encoder", enc="gru", dec="gru",
         ckpt=f"{SAVED_ROOT}/TEA_encoder_gru_align/best_loss_weights.pickle"),
]

# chord quality → valence 매핑
# N=0, maj=1, dim=2, sus4=3, min7=4, min=5, sus2=6, aug=7
# dim7=8, maj6=9, hdim7=10, 7=11, min6=12, maj7=13
QUALITY_VALENCE = {
    0:  0.0,   # N (no chord)
    1:  1.0,   # maj
    2: -1.0,   # dim
    3:  0.0,   # sus4
    4: -0.5,   # min7
    5: -1.0,   # min
    6:  0.0,   # sus2
    7:  0.5,   # aug
    8: -1.0,   # dim7
    9:  1.0,   # maj6
    10:-0.7,   # hdim7
    11: 0.3,   # 7 (dominant)
    12:-0.5,   # min6
    13: 1.0,   # maj7
}


# ── No-emotion wrapper ────────────────────────────────────────────────────────
class NoEmoWrapper(Dataset):
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


# ── Valence 계산 헬퍼 ─────────────────────────────────────────────────────────
def emotion_quality_to_valence(emo_quality_row):
    """tgt_emotion_quality (14-dim) → 스칼라 valence."""
    val = 0.0
    for q in range(14):
        if emo_quality_row[q] == 1:
            val += QUALITY_VALENCE[q]
    return val


def chord_idx_to_valence(chord_idx, chord_inv, chord_attr):
    """예측 chord index → quality → valence."""
    if chord_idx in (CHORD_END, CHORD_PAD):
        return None
    chord_str = chord_inv.get(str(chord_idx))
    if chord_str is None:
        return None
    parts = chord_str.split(":")
    quality_id = 1 if len(parts) == 1 else chord_attr.get(parts[1], 1)
    return QUALITY_VALENCE.get(quality_id, 0.0)


# ── 시퀀스별 Pearson r 계산 ───────────────────────────────────────────────────
def compute_pearson(out_seq, tgt_seq, emo_seq, chord_inv, chord_attr, min_valid=10):
    """
    단일 시퀀스에 대해 Pearson r 계산.
    min_valid: 유효 timestep 최소 수 (이하면 None 반환).
    """
    softmax = nn.Softmax(dim=-1)
    pred    = torch.argmax(softmax(out_seq), dim=-1).flatten()
    tgt_flat = tgt_seq.flatten()
    emo_qual = emo_seq[:, 0:14]

    video_vals, music_vals = [], []
    for i, p in enumerate(pred):
        # 유효하지 않은 timestep 스킵
        if tgt_flat[i].item() == CHORD_PAD:
            continue
        all_zeros = torch.all(emo_qual[i] == 0)
        if emo_seq[i][-1] == 1 or all_zeros:
            continue

        v_val = emotion_quality_to_valence(emo_qual[i])
        m_val = chord_idx_to_valence(p.item(), chord_inv, chord_attr)
        if m_val is None:
            continue

        video_vals.append(v_val)
        music_vals.append(m_val)

    if len(video_vals) < min_valid:
        return None

    # 분산이 0이면 Pearson 정의 불가
    if np.std(video_vals) < 1e-8 or np.std(music_vals) < 1e-8:
        return None

    r, _ = pearsonr(video_vals, music_vals)
    return float(r)


# ── 전체 eval ─────────────────────────────────────────────────────────────────
def eval_pearson(model, loader, chord_inv, chord_attr, min_valid):
    model.eval()
    rs, skipped = [], 0

    with torch.no_grad():
        for batch in loader:
            x       = batch["x"].to(get_device())
            tgt     = batch["tgt"].to(get_device())
            x_root  = batch["x_root"].to(get_device())
            x_attr  = batch["x_attr"].to(get_device())
            tgt_emo = batch["tgt_emotion"].to(get_device())
            sem     = [f.to(get_device()) for f in batch["semanticList"]]
            key     = batch["key"].to(get_device())
            scene   = batch["scene_offset"].to(get_device())
            motion  = batch["motion"].to(get_device())
            emotion = batch["emotion"].to(get_device())

            y = model(x, x_root, x_attr, sem, key, scene, motion, emotion)

            r = compute_pearson(y[0], tgt[0], tgt_emo[0],
                                chord_inv, chord_attr, min_valid)
            if r is not None:
                rs.append(r)
            else:
                skipped += 1

    mean_r = float(np.mean(rs)) if rs else float("nan")
    return mean_r, len(rs), skipped


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min_valid", type=int, default=10,
                   help="시퀀스당 최소 유효 timestep 수 (기본: 10)")
    return p.parse_args()


def main():
    args = parse_args()
    use_cuda(True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    with open("./dataset/vevo_meta/chord_inv.json")  as f: chord_inv  = json.load(f)
    with open("./dataset/vevo_meta/chord_attr.json") as f: chord_attr = json.load(f)

    _, val_ds, _ = create_vevo_datasets(
        dataset_root="./dataset/", max_seq_chord=300, max_seq_video=300,
        vis_models=VIS_MODELS, emo_model=EMO_MODEL,
        split_ver=SPLIT_VER, random_seq=False, is_video=True,
    )
    total_vf_dim  = sum(vf.shape[1] for vf in val_ds[0]["semanticList"])
    total_vf_dim += 1 + 1

    val_ds_noemo = NoEmoWrapper(val_ds)

    print(f"\n{'Model':<20} {'Pearson r':>10} {'N':>6} {'Skipped':>8}")
    print("=" * 48)

    rows, header = [], ["Model", "Pearson r", "N used", "Skipped"]

    for cfg in MODELS:
        if not os.path.isfile(cfg["ckpt"]):
            print(f"  SKIP {cfg['name']:<18} — 체크포인트 없음")
            continue

        model = build_model(cfg, total_vf_dim)
        try:
            model.load_state_dict(
                torch.load(cfg["ckpt"], map_location=get_device()), strict=False)
        except RuntimeError as e:
            print(f"  SKIP {cfg['name']:<18} — 아키텍처 불일치 (재학습 필요)")
            del model; torch.cuda.empty_cache(); continue

        ds  = val_ds_noemo if cfg["no_emo"] else val_ds
        ldr = DataLoader(ds, batch_size=1, num_workers=2)

        mean_r, n_used, n_skip = eval_pearson(model, ldr, chord_inv, chord_attr, args.min_valid)

        print(f"  {cfg['name']:<20} {mean_r:>10.4f} {n_used:>6} {n_skip:>8}")
        rows.append([cfg["name"], f"{mean_r:.4f}", n_used, n_skip])

        del model; torch.cuda.empty_cache()

    print("=" * 48)

    os.makedirs("experiments", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n결과 저장: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

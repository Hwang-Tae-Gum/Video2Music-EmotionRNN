"""
eval_emotion_valence.py
────────────────────────
Emotion Valence Agreement Rate 평가.

각 timestep에서:
  - video emotion 6-dim → argmax → emotion category
    exciting(0)=+1, fearful(1)=-1, tense(2)=-1, sad(3)=-1, relaxing(4)=+1, neutral(5)=skip
  - 예측 chord quality → QUALITY_VALENCE → sign(+/-)
  - 두 방향이 일치하는 비율 = Agreement Rate

neutral video / valence=0 chord(sus4, sus2, N) → skip

사용법:
  python eval_emotion_valence.py
"""

import os, csv, json
import torch
import torch.nn as nn
import numpy as np
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
OUTPUT_CSV = "./experiments/eval_emotion_valence.csv"

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
    dict(name="MS-TEA",
         type="tea", no_emo=False,
         where="encoder", enc="gru", dec="gru", multisignal=True,
         ckpt=f"{SAVED_ROOT}/MSTEA_encoder_gru/best_loss_weights.pickle"),
    dict(name="MS-TEA_align",
         type="tea", no_emo=False,
         where="encoder", enc="gru", dec="gru", multisignal=True,
         ckpt=f"{SAVED_ROOT}/MSTEA_encoder_gru_align/best_loss_weights.pickle"),
    dict(name="MS-TEA_align_l01",
         type="tea", no_emo=False,
         where="encoder", enc="gru", dec="gru", multisignal=True,
         ckpt=f"{SAVED_ROOT}/MSTEA_encoder_gru_align_l01/best_loss_weights.pickle"),
    dict(name="MS-TEA_align_l015",
         type="tea", no_emo=False,
         where="encoder", enc="gru", dec="gru", multisignal=True,
         ckpt=f"{SAVED_ROOT}/MSTEA_encoder_gru_align_l015/best_loss_weights.pickle"),
    dict(name="MS-TEA_align_scratch",
         type="tea", no_emo=False,
         where="encoder", enc="gru", dec="gru", multisignal=True,
         ckpt=f"{SAVED_ROOT}/MSTEA_encoder_gru_align_scratch/best_loss_weights.pickle"),
    dict(name="MS-TEA_align_curriculum",
         type="tea", no_emo=False,
         where="encoder", enc="gru", dec="gru", multisignal=True,
         ckpt=f"{SAVED_ROOT}/MSTEA_encoder_gru_align_curriculum/best_loss_weights.pickle"),
    dict(name="MS-TEA_align_softmargin",
         type="tea", no_emo=False,
         where="encoder", enc="gru", dec="gru", multisignal=True,
         ckpt=f"{SAVED_ROOT}/MSTEA_encoder_gru_align_softmargin/best_loss_weights.pickle"),
    dict(name="MS-TEA_align_attn",
         type="tea", no_emo=False,
         where="encoder", enc="attention", dec="gru", multisignal=True,
         ckpt=f"{SAVED_ROOT}/MSTEA_encoder_attention_align/best_loss_weights.pickle"),
]

# video emotion category index → valence direction
# 6c_l14p: exciting(0), fearful(1), tense(2), sad(3), relaxing(4), neutral(5)
VIDEO_EMO_DIRECTION = [1, -1, -1, -1, 1, 0]   # 0 = skip (neutral)

# chord quality id → valence scalar
QUALITY_VALENCE = {
    0:  0.0,   # N
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
            use_multisignal=cfg.get("multisignal", False),
        ).to(get_device())


# ── chord index → quality id ──────────────────────────────────────────────────
def chord_idx_to_quality(chord_idx, chord_inv, chord_attr):
    if chord_idx in (CHORD_END, CHORD_PAD):
        return None
    chord_str = chord_inv.get(str(chord_idx))
    if chord_str is None:
        return None
    parts = chord_str.split(":")
    return 1 if len(parts) == 1 else chord_attr.get(parts[1], 1)


# ── Agreement Rate 계산 ───────────────────────────────────────────────────────
def eval_valence_agreement(model, loader, chord_inv, chord_attr, raw_val_ds):
    """
    loader: NoEmoWrapper or raw val_ds DataLoader (model input용)
    raw_val_ds: 원본 val_ds (video emotion label 읽기용, zeroing 전)
    """
    model.eval()
    softmax = nn.Softmax(dim=-1)

    agree_cnt = 0
    disagree_cnt = 0
    skip_cnt = 0
    raw_idx = 0

    with torch.no_grad():
        for batch in loader:
            B = batch["x"].shape[0]
            y = model(
                batch["x"].to(get_device()),
                batch["x_root"].to(get_device()),
                batch["x_attr"].to(get_device()),
                [f.to(get_device()) for f in batch["semanticList"]],
                batch["key"].to(get_device()),
                batch["scene_offset"].to(get_device()),
                batch["motion"].to(get_device()),
                batch["emotion"].to(get_device()),
            )

            for b in range(B):
                # video emotion label은 원본에서 읽음 (NoEmoWrapper zeroing 전)
                raw_emo = raw_val_ds[raw_idx]["emotion"]   # [T, 6]
                tgt_b   = batch["tgt"][b].flatten()        # [T]
                pred    = torch.argmax(softmax(y[b]), dim=-1)  # [T]
                raw_idx += 1

                for t in range(pred.shape[0]):
                    # CHORD_PAD timestep 스킵
                    if tgt_b[t].item() == CHORD_PAD:
                        skip_cnt += 1
                        continue

                    # video emotion direction
                    if t >= raw_emo.shape[0]:
                        skip_cnt += 1
                        continue
                    emo_cat = int(torch.argmax(raw_emo[t]).item())
                    vid_dir = VIDEO_EMO_DIRECTION[emo_cat]
                    if vid_dir == 0:   # neutral
                        skip_cnt += 1
                        continue

                    # predicted chord valence direction
                    q = chord_idx_to_quality(pred[t].item(), chord_inv, chord_attr)
                    if q is None:
                        skip_cnt += 1
                        continue
                    chord_val = QUALITY_VALENCE.get(q, 0.0)
                    if chord_val == 0.0:   # sus4 / sus2 / N
                        skip_cnt += 1
                        continue
                    chord_dir = 1 if chord_val > 0 else -1

                    if vid_dir == chord_dir:
                        agree_cnt += 1
                    else:
                        disagree_cnt += 1

    total = agree_cnt + disagree_cnt
    rate = agree_cnt / total if total > 0 else float("nan")
    return rate, agree_cnt, disagree_cnt, skip_cnt


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-only_model", type=str, default=None,
                        help="지정 모델 이름만 평가 (없으면 전체)")
    cli_args = parser.parse_args()

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

    print(f"\n{'Model':<20} {'Agree Rate':>11} {'Agree':>8} {'Disagree':>10} {'Skip':>8}")
    print("=" * 62)

    rows = []
    header = ["Model", "Agreement Rate", "Agree", "Disagree", "Skip"]

    for cfg in MODELS:
        if cli_args.only_model and cfg["name"] != cli_args.only_model:
            continue
        if not os.path.isfile(cfg["ckpt"]):
            print(f"  SKIP {cfg['name']:<18} — 체크포인트 없음")
            continue

        model = build_model(cfg, total_vf_dim)
        try:
            model.load_state_dict(
                torch.load(cfg["ckpt"], map_location=get_device()), strict=False)
        except RuntimeError:
            print(f"  SKIP {cfg['name']:<18} — 아키텍처 불일치")
            del model; torch.cuda.empty_cache(); continue

        ds  = val_ds_noemo if cfg["no_emo"] else val_ds
        ldr = DataLoader(ds, batch_size=1, num_workers=2)

        rate, agree, disagree, skip = eval_valence_agreement(
            model, ldr, chord_inv, chord_attr, val_ds)

        print(f"  {cfg['name']:<20} {rate:>11.4f} {agree:>8} {disagree:>10} {skip:>8}")
        rows.append([cfg["name"], f"{rate:.4f}", agree, disagree, skip])

        del model; torch.cuda.empty_cache()

    print("=" * 62)

    os.makedirs("experiments", exist_ok=True)
    if cli_args.only_model:
        # 기존 CSV에 행 추가 (중복 제거 후 append)
        existing = []
        if os.path.isfile(OUTPUT_CSV):
            with open(OUTPUT_CSV, newline="") as f:
                existing = list(csv.reader(f))
        existing_names = {r[0] for r in existing[1:]} if len(existing) > 1 else set()
        new_rows = [r for r in rows if r[0] not in existing_names]
        write_header = not existing
        with open(OUTPUT_CSV, "a", newline="") as f:
            if write_header:
                csv.writer(f).writerow(header)
            csv.writer(f).writerows(new_rows)
    else:
        with open(OUTPUT_CSV, "w", newline="") as f:
            csv.writer(f).writerow(header)
            csv.writer(f).writerows(rows)
    print(f"\n결과 저장: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

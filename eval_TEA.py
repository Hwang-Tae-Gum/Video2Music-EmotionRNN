"""
eval_TEA.py
───────────
TEA 실험 8개의 best_loss_weights.pickle 을 로드해
val set 에서 Affective Correspondence + H@k + Emotion Loss 를 측정한다.

사용법:
  python eval_TEA.py               # 실험 1~8 전부
  python eval_TEA.py --exp 3 4 8   # 특정 실험만
"""

import os, sys, csv, argparse
import torch
import torch.nn as nn

from dataset.vevo_dataset import create_vevo_datasets
from model.video_music_transformer import VideoMusicTransformer
from model.video_music_transformer_TEA import VideoMusicTransformerTEA
from torch.utils.data import DataLoader

import utilities.constants_TEA as cTEA
import model.video_music_transformer_TEA as _tea_module
from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.run_model_vevo import eval_model

# ── 실험 설정 테이블 ──────────────────────────────────────────────────────────
EXP_CONFIGS = {
    1: dict(where="decoder", enc="gru",  dec="lstm", dir="TEA_decoder_lstm_exp1"),
    2: dict(where="decoder", enc="gru",  dec="gru",  dir="TEA_decoder_gru_exp2"),
    3: dict(where="encoder", enc="lstm", dec="gru",  dir="TEA_encoder_lstm_exp3"),
    4: dict(where="encoder", enc="gru",  dec="gru",  dir="TEA_encoder_gru_exp4"),
    5: dict(where="both",    enc="lstm", dec="lstm", dir="TEA_both_lstm_lstm_exp5"),
    6: dict(where="both",    enc="gru",  dec="gru",  dir="TEA_both_gru_gru_exp6"),
    7: dict(where="both",    enc="lstm", dec="gru",  dir="TEA_both_lstm_gru_exp7"),
    8: dict(where="both",    enc="gru",  dec="lstm", dir="TEA_both_gru_lstm_exp8"),
}

VIS_MODELS  = "2d/clip_l14p"
EMO_MODEL   = "6c_l14p"
SAVED_ROOT  = "./saved_models"
OUTPUT_CSV  = "./experiments/eval_correspondence.csv"
BATCH_SIZE  = 4


def build_baseline_model(total_vf_dim):
    model = VideoMusicTransformer(
        n_layers           = 6,
        num_heads          = 8,
        d_model            = 512,
        dim_feedforward    = 1024,
        dropout            = 0.1,
        max_sequence_midi  = 2048,
        max_sequence_video = 300,
        max_sequence_chord = 300,
        total_vf_dim       = total_vf_dim,
        rpr                = RPR,
    ).to(get_device())
    return model


def build_model(cfg, total_vf_dim):
    # constants_TEA 와 모델 모듈 네임스페이스 둘 다 패치해야
    # from ... import 로 복사된 로컬 변수가 반영됨
    cTEA.TEA_WHERE        = cfg["where"]
    cTEA.TEA_ENCODER_CELL = cfg["enc"]
    cTEA.TEA_DECODER_CELL = cfg["dec"]
    _tea_module.TEA_WHERE        = cfg["where"]
    _tea_module.TEA_ENCODER_CELL = cfg["enc"]
    _tea_module.TEA_DECODER_CELL = cfg["dec"]

    model = VideoMusicTransformerTEA(
        n_layers           = 6,
        num_heads          = 8,
        d_model            = 512,
        dim_feedforward    = 1024,
        dropout            = 0.1,
        max_sequence_midi  = 2048,
        max_sequence_video = 300,
        max_sequence_chord = 300,
        total_vf_dim       = total_vf_dim,
        rpr                = RPR,
        tea_where          = cfg["where"],
        tea_num_layers     = cTEA.TEA_NUM_LAYERS,
    ).to(get_device())
    return model


def run_eval(exp_ids):
    use_cuda(True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # ── 데이터셋 (한 번만 생성) ──────────────────────────────────────────────
    _, val_dataset, _ = create_vevo_datasets(
        dataset_root    = "./dataset/",
        max_seq_chord   = 300,
        max_seq_video   = 300,
        vis_models      = VIS_MODELS,
        emo_model       = EMO_MODEL,
        split_ver       = SPLIT_VER,
        random_seq      = False,
        is_video        = True,
    )
    total_vf_dim = sum(vf.shape[1] for vf in val_dataset[0]["semanticList"])
    total_vf_dim += 1 + 1  # scene_offset + motion (emotion은 Linear_emo 전용 pathway)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

    eval_loss_func         = nn.CrossEntropyLoss(ignore_index=CHORD_PAD)
    eval_loss_emotion_func = nn.BCEWithLogitsLoss()

    rows = []
    header = ["Exp", "TEA 위치", "Encoder Cell", "Decoder Cell",
              "H@1", "H@3", "H@5", "Emo Loss", "Affective Corr", "Acc+Corr"]

    print("=" * 70)
    print(f"{'Exp':<5} {'위치':<14} {'셀':<12} {'H@1':>6} {'H@3':>6} {'H@5':>6} {'EmoLoss':>8} {'Corr':>7}")
    print("=" * 70)

    # ── AMT Baseline ─────────────────────────────────────────────────────────
    ckpt_baseline = os.path.join(SAVED_ROOT, "AMT_full", "best_loss_weights.pickle")
    if os.path.isfile(ckpt_baseline):
        model = build_baseline_model(total_vf_dim)
        model.load_state_dict(torch.load(ckpt_baseline, map_location=get_device()))
        metrics = eval_model(model, val_loader,
                             eval_loss_func, eval_loss_emotion_func,
                             isVideo=True)
        h1, h3, h5 = metrics["avg_h1"], metrics["avg_h3"], metrics["avg_h5"]
        emo = metrics["avg_loss_emotion"]
        cor = metrics["avg_cor"]
        acc_cor = metrics["avg_acc_cor"]
        cor_str = f"{cor:.4f}" if cor != -1 else "N/A"
        acc_str = f"{acc_cor:.4f}" if acc_cor != -1 else "N/A"
        print(f"{'BASE':<5} {'—':<14} {'—':<12} {h1:>6.4f} {h3:>6.4f} {h5:>6.4f} {emo:>8.4f} {cor_str:>7}")
        rows.append(["Baseline", "—", "—", "—",
                     f"{h1:.4f}", f"{h3:.4f}", f"{h5:.4f}",
                     f"{emo:.4f}", cor_str, acc_str])
        del model
        torch.cuda.empty_cache()
    else:
        print("Baseline: 체크포인트 없음, 스킵")

    for exp_id in exp_ids:
        cfg  = EXP_CONFIGS[exp_id]
        ckpt = os.path.join(SAVED_ROOT, cfg["dir"], "best_loss_weights.pickle")
        if not os.path.isfile(ckpt):
            print(f"Exp{exp_id}: 체크포인트 없음 ({ckpt})")
            continue

        model = build_model(cfg, total_vf_dim)
        model.load_state_dict(torch.load(ckpt, map_location=get_device()))

        metrics = eval_model(model, val_loader,
                             eval_loss_func, eval_loss_emotion_func,
                             isVideo=True)

        h1   = metrics["avg_h1"]
        h3   = metrics["avg_h3"]
        h5   = metrics["avg_h5"]
        emo  = metrics["avg_loss_emotion"]
        cor  = metrics["avg_cor"]        # -1 이면 계산 불가
        acc_cor = metrics["avg_acc_cor"] # -1 이면 계산 불가

        cell_label = f"{cfg['enc'].upper()}+{cfg['dec'].upper()}" if cfg["where"] == "both" \
                     else (cfg["enc"].upper() if cfg["where"] == "encoder" else cfg["dec"].upper())
        cor_str  = f"{cor:.4f}"  if cor  != -1 else "N/A"
        acc_str  = f"{acc_cor:.4f}" if acc_cor != -1 else "N/A"

        print(f"Exp{exp_id:<3} {cfg['where']:<14} {cell_label:<12} "
              f"{h1:>6.4f} {h3:>6.4f} {h5:>6.4f} {emo:>8.4f} {cor_str:>7}")

        rows.append([exp_id, cfg["where"], cfg["enc"].upper(), cfg["dec"].upper(),
                     f"{h1:.4f}", f"{h3:.4f}", f"{h5:.4f}",
                     f"{emo:.4f}", cor_str, acc_str])

        del model
        torch.cuda.empty_cache()

    print("=" * 70)

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    os.makedirs("experiments", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n결과 저장: {OUTPUT_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, nargs="+",
                        default=list(range(1, 9)),
                        help="평가할 실험 번호 (기본: 1~8)")
    args = parser.parse_args()
    run_eval(sorted(set(args.exp)))


if __name__ == "__main__":
    main()

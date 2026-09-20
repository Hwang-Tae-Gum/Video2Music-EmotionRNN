#!/usr/bin/env python3
"""
AMT_no_emotion / AMT_full / MSTEA Exp16 전체 지표 측정.
  python _eval_amts.py
"""
import os, sys, torch, torch.nn as nn

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dataset.vevo_dataset import create_vevo_datasets
from model.video_music_transformer import VideoMusicTransformer
from model.video_music_transformer_TEA import VideoMusicTransformerTEA
from torch.utils.data import DataLoader
import utilities.constants_TEA as cTEA
import model.video_music_transformer_TEA as _tea_module
from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.run_model_vevo import eval_model

use_cuda(True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# ── 데이터셋 ──────────────────────────────────────────────────────────────────
_, val_dataset, _ = create_vevo_datasets(
    dataset_root  = "./dataset/",
    max_seq_chord = 300,
    max_seq_video = 300,
    vis_models    = "2d/clip_l14p",
    emo_model     = "6c_l14p",
    split_ver     = SPLIT_VER,
    random_seq    = False,
    is_video      = True,
)
total_vf_dim = sum(vf.shape[1] for vf in val_dataset[0]["semanticList"]) + 1 + 1  # 770
val_loader = DataLoader(val_dataset, batch_size=4, num_workers=2)
loss_chord = nn.CrossEntropyLoss(ignore_index=CHORD_PAD)
loss_emo   = nn.BCEWithLogitsLoss()

print(f"val samples: {len(val_dataset)}, total_vf_dim: {total_vf_dim}\n")


def run(name, ckpt_path, model):
    if not os.path.isfile(ckpt_path):
        print(f"[SKIP] {name}: 체크포인트 없음 ({ckpt_path})")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=get_device(), weights_only=False))
    model.eval()
    m = eval_model(model, val_loader, loss_chord, loss_emo, isVideo=True)
    h1 = m["avg_h1"]; h3 = m["avg_h3"]; h5 = m["avg_h5"]
    ar = m["avg_cor"]
    ar_str = f"{ar:.4f}" if ar != -1 else "N/A"
    print(f"[{name}]")
    print(f"  H@1={h1:.4f}  H@3={h3:.4f}  H@5={h5:.4f}  AR={ar_str}")
    del model
    torch.cuda.empty_cache()
    return {"H@1": h1, "H@3": h3, "H@5": h5, "AR": ar}


def make_baseline(vf_dim):
    return VideoMusicTransformer(
        n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
        max_sequence_midi=2048, max_sequence_video=300, max_sequence_chord=300,
        total_vf_dim=vf_dim, rpr=RPR,
    ).to(get_device())


def make_tea_exp16(vf_dim):
    cTEA.TEA_WHERE = "encoder"; cTEA.TEA_ENCODER_CELL = "attention"; cTEA.TEA_DECODER_CELL = "gru"
    _tea_module.TEA_WHERE = "encoder"
    _tea_module.TEA_ENCODER_CELL = "attention"
    _tea_module.TEA_DECODER_CELL = "gru"
    return VideoMusicTransformerTEA(
        n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
        max_sequence_midi=2048, max_sequence_video=300, max_sequence_chord=300,
        total_vf_dim=vf_dim, rpr=RPR, tea_where="encoder",
        tea_num_layers=cTEA.TEA_NUM_LAYERS, use_multisignal=True,
    ).to(get_device())


print("=" * 55)
results = {}
results["no_emo"]  = run("AMT_no_emotion",
                         "saved_models/AMT_no_emotion/best_loss_weights.pickle",
                         make_baseline(total_vf_dim))
results["full"]    = run("AMT_full (Baseline)",
                         "saved_models/AMT_full/best_loss_weights.pickle",
                         make_baseline(total_vf_dim))
results["exp16"]   = run("MSTEA_encoder_attention_align (Exp16)",
                         "saved_models/MSTEA_encoder_attention_align/best_loss_weights.pickle",
                         make_tea_exp16(total_vf_dim))
print("=" * 55)

print("\n── 논문 테이블 형식 ──────────────────────────────────────")
print(f"{'모델':<35} {'H@1':>6} {'H@3':>6} {'H@5':>6} {'AR':>6}")
for label, key in [("AMT no emotion", "no_emo"),
                   ("Baseline (AMT full)", "full"),
                   ("MS-TEA Exp16", "exp16")]:
    r = results.get(key)
    if r:
        ar = f"{r['AR']:.4f}" if r['AR'] != -1 else "N/A"
        print(f"{label:<35} {r['H@1']:>6.4f} {r['H@3']:>6.4f} {r['H@5']:>6.4f} {ar:>6}")

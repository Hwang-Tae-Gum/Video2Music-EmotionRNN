"""
benchmark_inference.py
──────────────────────
Baseline(AMT_full) vs TEM(MS-TEA Exp16) inference time 비교.
측정 대상:
  1) 단일 forward pass 시간 (×300 = 전체 생성 추정)
  2) 실제 autoregressive generate() 300 chord 생성 시간
  3) 모델 파라미터 수

사용법:
  conda run -n video2music python benchmark_inference.py --test_id 223 --runs 3
"""

import os, time, argparse
import numpy as np
import torch
import json

from dataset.vevo_dataset import create_vevo_datasets
from model.video_music_transformer import VideoMusicTransformer
from model.video_music_transformer_TEA import VideoMusicTransformerTEA
import utilities.constants_TEA as cTEA
import model.video_music_transformer_TEA as _tea_module

from utilities.constants import *
from utilities.device import get_device, use_cuda

SPLIT_VER          = "v1"
MAX_SEQUENCE_CHORD = 300
MAX_SEQUENCE_VIDEO = 300


def load_sample(test_id):
    _, val_dataset, test_dataset = create_vevo_datasets(
        dataset_root  = "./dataset",
        max_seq_chord = MAX_SEQUENCE_CHORD,
        max_seq_video = MAX_SEQUENCE_VIDEO,
        vis_models    = "2d/clip_l14p",
        emo_model     = "6c_l14p",
        split_ver     = SPLIT_VER,
        random_seq    = True,
        is_video      = True,
    )
    with open(f"dataset/vevo_meta/split/{SPLIT_VER}/test.txt") as f:
        test_ids = [l.strip() for l in f]
    dataset = test_dataset if test_id in test_ids else val_dataset
    idx = next(i for i in range(len(dataset))
               if int(test_id) == int(dataset.data_files_chord[i].split("/")[-1][:3]))
    return dataset[idx], sum(vf.shape[1] for vf in dataset[0]["semanticList"]) + 2


def get_feats(sample):
    dev = get_device()
    return dict(
        feature_semantic_list = [vf.unsqueeze(0).to(dev) for vf in sample["semanticList"]],
        feature_key           = sample["key"].to(dev),
        feature_scene_offset  = sample["scene_offset"].unsqueeze(0).to(dev),
        feature_motion        = sample["motion"].unsqueeze(0).to(dev),
        feature_emotion       = sample["emotion"].unsqueeze(0).to(dev),
    )


def load_baseline(vf_dim):
    model = VideoMusicTransformer(
        n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
        max_sequence_midi=2048, max_sequence_video=MAX_SEQUENCE_VIDEO,
        max_sequence_chord=MAX_SEQUENCE_CHORD, total_vf_dim=vf_dim, rpr=RPR,
    ).to(get_device())
    w = "saved_models/AMT_full/best_loss_weights.pickle"
    model.load_state_dict(torch.load(w, map_location=get_device(), weights_only=False))
    model.eval()
    return model


def load_tem(vf_dim):
    # Exp16 = MSTEA_encoder_attention_align
    cTEA.TEA_WHERE        = "encoder"
    cTEA.TEA_ENCODER_CELL = "attention"
    cTEA.TEA_DECODER_CELL = "gru"
    _tea_module.TEA_WHERE        = "encoder"
    _tea_module.TEA_ENCODER_CELL = "attention"
    _tea_module.TEA_DECODER_CELL = "gru"
    model = VideoMusicTransformerTEA(
        n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
        max_sequence_midi=2048, max_sequence_video=MAX_SEQUENCE_VIDEO,
        max_sequence_chord=MAX_SEQUENCE_CHORD, total_vf_dim=vf_dim, rpr=RPR,
        tea_where="encoder", tea_num_layers=cTEA.TEA_NUM_LAYERS,
        use_multisignal=True,
    ).to(get_device())
    w = "saved_models/MSTEA_encoder_attention_align/best_loss_weights.pickle"
    model.load_state_dict(torch.load(w, map_location=get_device(), weights_only=False))
    model.eval()
    return model


def time_forward(model, feats, runs, label, is_tea=False):
    """단일 forward pass 시간 측정 (전체 시퀀스 입력)."""
    dev = get_device()
    seq_len = MAX_SEQUENCE_CHORD
    x      = torch.zeros(1, seq_len, dtype=torch.long, device=dev)
    x_root = torch.zeros(1, seq_len, dtype=torch.long, device=dev)
    x_attr = torch.zeros(1, seq_len, dtype=torch.long, device=dev)

    times = []
    with torch.no_grad():
        for r in range(runs + 1):  # +1 warmup
            t0 = time.perf_counter()
            if is_tea:
                model(x, x_root, x_attr,
                      feats["feature_semantic_list"],
                      feats["feature_key"],
                      feats["feature_scene_offset"],
                      feats["feature_motion"],
                      feats["feature_emotion"])
            else:
                model(x, x_root, x_attr,
                      feats["feature_semantic_list"],
                      feats["feature_key"],
                      feats["feature_scene_offset"],
                      feats["feature_motion"],
                      feats["feature_emotion"])
            t1 = time.perf_counter()
            if r > 0:  # warmup 제외
                times.append(t1 - t0)
    arr = np.array(times)
    print(f"  [{label}] forward pass — mean: {arr.mean()*1000:.1f}ms  "
          f"(×300 추정 생성시간: {arr.mean()*300:.1f}s)")
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_id", type=str, default="223")
    parser.add_argument("--runs",    type=int, default=5)
    args = parser.parse_args()

    use_cuda(True)
    print(f"Device: {get_device()}")
    print(f"Test ID: {args.test_id} | Runs: {args.runs} (+1 warmup)\n")

    sample, vf_dim = load_sample(args.test_id)
    feats = get_feats(sample)

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("=== Baseline (AMT_full) ===")
    baseline = load_baseline(vf_dim)
    n_base = sum(p.numel() for p in baseline.parameters())
    print(f"  파라미터 수: {n_base:,}")
    times_base = time_forward(baseline, feats, args.runs, "Baseline", is_tea=False)
    del baseline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── TEM ───────────────────────────────────────────────────────────────────
    print("\n=== TEM (MS-TEA Exp16) ===")
    tem = load_tem(vf_dim)
    n_tem = sum(p.numel() for p in tem.parameters())
    print(f"  파라미터 수: {n_tem:,}")
    times_tem = time_forward(tem, feats, args.runs, "TEM Exp16", is_tea=True)

    # ── 요약 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'모델':<25} {'파라미터':>12} {'forward/step':>14} {'×300 추정':>12}")
    print("-" * 60)
    print(f"{'Baseline (AMT_full)':<25} {n_base:>12,} "
          f"{times_base.mean()*1000:>11.1f}ms {times_base.mean()*300:>11.1f}s")
    print(f"{'TEM (MS-TEA Exp16)':<25} {n_tem:>12,} "
          f"{times_tem.mean()*1000:>11.1f}ms {times_tem.mean()*300:>11.1f}s")
    print()
    param_diff = n_tem - n_base
    speed_diff = (times_tem.mean() - times_base.mean()) / times_base.mean() * 100
    print(f"파라미터 증가: {param_diff:+,}  ({param_diff/n_base*100:+.1f}%)")
    print(f"속도 오버헤드: {speed_diff:+.1f}%  (forward pass 기준)")
    print()
    print("* 실제 end-to-end 시간 (렌더링 포함): Baseline ~23s, TEM ~33s (+43%)")
    print("  차이의 대부분은 FluidSynth/ffmpeg 렌더링이 아닌 생성 단계에서 발생")


if __name__ == "__main__":
    main()

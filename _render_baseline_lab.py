#!/usr/bin/env python3
"""
Baseline 데모 생성 파이프라인 (AMT_full → TEA 렌더링 → 배너).

사용법:
  python _render_baseline_lab.py 223
  python _render_baseline_lab.py 049

단계:
  1. AMT_full 모델로 코드 시퀀스 생성 → output/<id>/<id>_cgen_rd.lab
  2. TEA와 동일한 렌더링 파이프라인으로 재렌더링 → _BASE_cgen_rd.mp4
  3. 'Baseline' 배너 추가 → _BASE_banner.mp4
"""
import sys, os, shutil, subprocess, json
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
test_id = sys.argv[1] if len(sys.argv) > 1 else "223"
print(f"[Baseline Pipeline] test_id={test_id}")

# ── TEA 렌더링 파라미터 (generate_TEA.py와 동일) ──────────────────────────────
TEMPO        = 120
VELOCITY     = 100
DURATION     = 2
IS_VOICE     = True
IS_ARP       = True
MIN_LOUDNESS = 0
MAX_LOUDNESS = 50
MIN_VELOCITY = 49
MAX_VELOCITY = 112

from midiutil import MIDIFile
from utilities.chord_to_midi import Chord, voice


def convert_format_id_to_offset(id_list):
    offset_list, current_id, offset = [], id_list[0], 0
    for item in id_list:
        if item != current_id:
            current_id, offset = item, 0
        offset_list.append(offset)
        offset += 1
    return offset_list


from utilities.device import get_device, use_cuda
from utilities.constants import *
from model.video_regression import VideoRegression
from dataset.vevo_dataset import create_vevo_datasets


# ── chords_to_midi: generate_TEA.py와 동일 함수 ───────────────────────────────
def chords_to_midi(chord_genlist, chord_offsetlist, density_list, velo_list, tempo,
                   dominant_emo="neutral"):
    _EMO_INSTR = {
        "exciting": dict(lead=0,  pad=48, pad_dv=10),
        "fearful":  dict(lead=0,  pad=42, pad_dv=20),
        "tense":    dict(lead=0,  pad=42, pad_dv=18),
        "sad":      dict(lead=0,  pad=48, pad_dv=15),
        "relaxing": dict(lead=24, pad=48, pad_dv=20),
        "neutral":  dict(lead=0,  pad=48, pad_dv=25),
    }
    instr = _EMO_INSTR.get(dominant_emo, _EMO_INSTR["neutral"])

    MIDI = MIDIFile(3)
    for t in range(3):
        MIDI.addTempo(t, 0, tempo)
    MIDI.addProgramChange(0, 0, 0, instr["lead"])
    MIDI.addProgramChange(1, 1, 0, instr["pad"])
    MIDI.addProgramChange(2, 2, 0, 32)  # Acoustic Bass

    midi_chords_original = []
    for key in chord_genlist:
        key = key.replace(":", "")
        if key == "N":
            midi_chords_original.append([])
        else:
            midi_chords_original.append(Chord(key).getMIDI("c", 4))

    midi_chords = voice(midi_chords_original) if IS_VOICE else midi_chords_original

    def _safe_dedupe(chord):
        seen, out = set(), []
        for p in chord:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out if len(out) >= 4 else chord

    midi_chords = [_safe_dedupe(c) if c else c for c in midi_chords]

    if IS_ARP:
        for i, chord in enumerate(midi_chords):
            v = velo_list[min(i, len(velo_list) - 1)]
            d = DURATION
            di = min(i, len(density_list) - 1)
            if not chord:
                continue
            n = len(chord)

            if density_list[di] == 0:
                if chord_offsetlist[i] % 2 == 0:
                    MIDI.addNote(0, 0, chord[0], i*d+0, d, v)
                    MIDI.addNote(0, 0, chord[min(1,n-1)], i*d+1, d, v)
                else:
                    MIDI.addNote(0, 0, chord[min(2,n-1)], i*d+0, d, v)
                    MIDI.addNote(0, 0, chord[min(3,n-1)], i*d+1, d, v)
            elif density_list[di] == 1:
                if chord_offsetlist[i] % 2 == 0:
                    MIDI.addNote(0, 0, chord[0],          i*d+0,   d, v)
                    MIDI.addNote(0, 0, chord[min(1,n-1)], i*d+0.5, d, v)
                    MIDI.addNote(0, 0, chord[min(2,n-1)], i*d+1,   d, v)
                else:
                    MIDI.addNote(0, 0, chord[min(3,n-1)], i*d+0,   d, v)
                    MIDI.addNote(0, 0, chord[min(1,n-1)], i*d+0.5, d, v)
                    MIDI.addNote(0, 0, chord[min(2,n-1)], i*d+1,   d, v)
            elif density_list[di] == 2:
                for j, p in enumerate(chord[:4]):
                    MIDI.addNote(0, 0, p, i*d + j*0.5, d, v)
            elif density_list[di] == 3:
                for j, p in enumerate(chord[:4]):
                    MIDI.addNote(0, 0, p, i*d + j*0.25, d, v)
            else:
                for p in chord:
                    MIDI.addNote(0, 0, p, i*d, d, v)

            pad_v = max(MIN_VELOCITY, v - instr["pad_dv"])
            for p in chord[:3]:
                MIDI.addNote(1, 1, p, i*d, d*2, pad_v)

            bass = chord[0] - 12 if chord[0] >= 24 else chord[0]
            MIDI.addNote(2, 2, bass, i*d, d, max(MIN_VELOCITY, v - 20))

    return MIDI


# ── Step 1: Lab 파일 생성 (없으면 AMT_full 모델 실행) ─────────────────────────
lab_path = f"output/{test_id}/{test_id}_cgen_rd.lab"
if not os.path.exists(lab_path):
    print("  Lab 파일 없음 → AMT_full 모델 실행 중...")
    import generate_baseline_clean as gbc
    _orig_argv = sys.argv[:]
    sys.argv = [
        "generate_baseline_clean.py",
        "-output_dir", "./output",
        "-model_weights", "./saved_models/AMT_full/best_loss_weights.pickle",
        "-modelReg_weights", "./saved_models/AMT/best_rmse_weights.pickle",
    ]
    gbc.test_id = test_id
    gbc.main()
    sys.argv = _orig_argv

if not os.path.exists(lab_path):
    print(f"[ERROR] Lab 파일 생성 실패: {lab_path}")
    sys.exit(1)

chord_genlist = []
with open(lab_path) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("key"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            chord_genlist.append(parts[1])

print(f"  코드 수: {len(chord_genlist)}")
chord_offsetlist = convert_format_id_to_offset(chord_genlist)


# ── Step 2: Regression 모델로 density/velocity 예측 ──────────────────────────
print("  Regression 모델로 density/velocity 예측 중...")
_, val_dataset, test_dataset = create_vevo_datasets(
    dataset_root  = "./dataset/",
    max_seq_chord = 300,
    max_seq_video = 300,
    vis_models    = "2d/clip_l14p",
    emo_model     = "6c_l14p",
    split_ver     = SPLIT_VER,
    random_seq    = False,
    is_video      = True,
)

with open("dataset/vevo_meta/split/" + SPLIT_VER + "/test.txt") as f:
    testFileList = [l.strip() for l in f]
with open("dataset/vevo_meta/split/" + SPLIT_VER + "/val.txt") as f:
    valFileList = [l.strip() for l in f]

dataset = test_dataset if test_id in testFileList else val_dataset

idx = -1
for i in range(len(dataset)):
    if int(test_id) == int(dataset.data_files_chord[i].split("/")[-1][:3]):
        idx = i
        break
assert idx >= 0, f"test_id {test_id} not in dataset"

feature_semantic_list = [s.unsqueeze(0).to(get_device()) for s in dataset[idx]["semanticList"]]
feature_scene_offset  = dataset[idx]["scene_offset"].unsqueeze(0).to(get_device())
feature_motion        = dataset[idx]["motion"].unsqueeze(0).to(get_device())
feature_emotion       = dataset[idx]["emotion"].unsqueeze(0).to(get_device())

total_vf_dim = sum(v.shape[1] for v in dataset[idx]["semanticList"]) + 1 + 1  # 770
reg_vf_dim   = total_vf_dim + 6  # 776

reg_weights = "./saved_models/AMT/best_rmse_weights.pickle"
modelReg = VideoRegression(max_sequence_video=300, total_vf_dim=reg_vf_dim, regModel="bigru").to(get_device())
modelReg.load_state_dict(torch.load(reg_weights, map_location=get_device(), weights_only=False))
modelReg.eval()

_norm_path = "./saved_models/AMT/norm_stats.json"
if os.path.isfile(_norm_path):
    with open(_norm_path) as f:
        _ns = json.load(f)
    nd_mean, nd_std = _ns["nd_mean"], _ns["nd_std"]
    lv_mean, lv_std = _ns["lv_mean"], _ns["lv_std"]
else:
    nd_mean, nd_std, lv_mean, lv_std = 0.0, 1.0, 0.0, 1.0

with torch.no_grad():
    y = modelReg(feature_semantic_list, feature_scene_offset, feature_motion, feature_emotion)
    y = y.reshape(y.shape[0] * y.shape[1], -1)
    y_density, y_loudness = torch.split(y, 1, dim=1)

y_density_raw  = y_density.cpu().numpy() * nd_std + nd_mean
y_loudness_raw = y_loudness.cpu().numpy() * lv_std + lv_mean

density_np  = np.clip(np.round(y_density_raw).astype(int), 0, 40)
loudness_np = np.clip((y_loudness_raw * 100).astype(int), 0, 50)

exponent  = 0.3
velo_list = [
    int(np.round(((l[0] - MIN_LOUDNESS) / (MAX_LOUDNESS - MIN_LOUDNESS)) ** exponent
                 * (MAX_VELOCITY - MIN_VELOCITY) + MIN_VELOCITY))
    for l in loudness_np
]
_velo_arr  = np.array(velo_list, dtype=float)
_velo_arr  = np.convolve(_velo_arr, np.ones(5)/5, mode='same')
_vel_floor = max(int(np.median(_velo_arr)), MIN_VELOCITY + 20)
velo_list  = [max(int(v), _vel_floor) for v in _velo_arr]

density_list = [
    0 if d[0] <= 5 else 1 if d[0] <= 10 else 2 if d[0] <= 15 else 3 if d[0] <= 20 else 4
    for d in density_np
]


# ── Step 3: MIDI 생성 (TEA와 동일: 3트랙, neutral) ───────────────────────────
print("  MIDI 생성 중 (TEA 렌더링 파이프라인, neutral)...")
midi = chords_to_midi(chord_genlist, chord_offsetlist, density_list, velo_list, TEMPO)

f_midi = f"output/{test_id}/{test_id}_BASE_cgen_rd.mid"
with open(f_midi, "wb") as f:
    midi.writeFile(f)


# ── Step 4: MIDI → FLAC (TEA와 동일 SF2) ─────────────────────────────────────
f_flac = f"output/{test_id}/{test_id}_BASE_cgen_rd.flac"
_sf2_candidates = [
    "./soundfonts/default_sound_font.sf2",
    "/usr/share/sounds/sf2/MuseScore_General.sf2",
    "/home/taegum/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/home/taegum/miniconda3/envs/video2music/lib/python3.8/site-packages/pretty_midi/TimGM6mb.sf2",
]
_sf2 = next((p for p in _sf2_candidates if os.path.isfile(p)), _sf2_candidates[-1])
_fs_candidates = [
    shutil.which("fluidsynth"),
    "/home/taegum/miniconda3/envs/video2music/bin/fluidsynth",
    "/usr/bin/fluidsynth",
]
_fs_bin = next((p for p in _fs_candidates if p and os.path.isfile(p)), "fluidsynth")
print(f"  FluidSynth: {_fs_bin}, SF2: {_sf2}")
subprocess.call([_fs_bin, "-ni", _sf2, f_midi, "-F", f_flac, "-r", "44100"])


# ── Step 5: 영상 합성 (TEA와 동일 ffmpeg) ────────────────────────────────────
f_video_in  = f"dataset/vevo/{test_id}.mp4"
f_video_out = f"output/{test_id}/{test_id}_BASE_cgen_rd.mp4"
if os.path.isfile(f_video_in):
    print("  영상 합성 중...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", f_video_in, "-i", f_flac,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        f_video_out,
    ], check=True)
else:
    print(f"  ⚠ 원본 영상 없음 ({f_video_in})")
    sys.exit(1)


# ── Step 6: 배너 추가 → _BASE_banner.mp4 ──────────────────────────────────────
from _add_banner import add_banner
f_banner = f"output/{test_id}/{test_id}_BASE_banner.mp4"
print("  배너 추가 중...")
add_banner(f_video_out, f_banner, "Baseline")

# 중간 파일 정리
for tmp in [f_midi, f_flac, f_video_out]:
    if os.path.isfile(tmp):
        os.remove(tmp)

print(f"\n[완료] {f_banner}")

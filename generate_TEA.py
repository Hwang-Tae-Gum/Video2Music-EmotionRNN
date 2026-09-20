"""
generate_TEA.py
───────────────
TEA 모델로 영상에 맞는 배경 음악을 생성한다.
기존 generate.py 와 동일한 파이프라인 위에 VideoMusicTransformerTEA 를 올린 버전.

사용법:
  python generate_TEA.py --exp 3                    # Exp3 best 모델로 test_id=049 생성
  python generate_TEA.py --exp 8 --test_id 012      # Exp8 모델, 특정 영상
  python generate_TEA.py --exp 3 --key minor        # 마이너 키 강제
  python generate_TEA.py --exp 3 --no_primer        # 프라이머 없이 생성

출력 (output/<test_id>/):
  <id>_TEA_exp<n>_cgen_rd.mid    MIDI 파일
  <id>_TEA_exp<n>_cgen_rd.lab    코드 레이블
  <id>_TEA_exp<n>_cgen_rd.flac   오디오 파일
  <id>_TEA_exp<n>_cgen_rd.mp4    영상 + 음악 합성
"""

import os, sys, json, random, argparse, shutil, subprocess
import numpy as np
import torch
from pathlib import Path

from midiutil import MIDIFile
from midi2audio import FluidSynth
from utilities.chord_to_midi import *

import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip

from dataset.vevo_dataset import create_vevo_datasets
from model.video_music_transformer_TEA import VideoMusicTransformerTEA
from model.video_regression import VideoRegression

import utilities.constants_TEA as cTEA
import model.video_music_transformer_TEA as _tea_module

from utilities.constants import *
from utilities.device import get_device, use_cuda

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
    9: dict(where="encoder", enc="gru",  dec="gru",  dir="TEA_encoder_gru"),
   10: dict(where="encoder", enc="gru",  dec="gru",  dir="TEA_encoder_gru_align"),
   11: dict(where="encoder", enc="gru",  dec="gru",  dir="MSTEA_encoder_gru",            multisignal=True),
   12: dict(where="encoder", enc="gru",  dec="gru",  dir="MSTEA_encoder_gru_align",      multisignal=True),
   13: dict(where="encoder", enc="gru",  dec="gru",  dir="MSTEA_encoder_gru_align_l01",  multisignal=True),
   14: dict(where="encoder", enc="gru",       dec="gru", dir="MSTEA_encoder_gru_align_l015",    multisignal=True),
   16: dict(where="encoder", enc="attention", dec="gru", dir="MSTEA_encoder_attention_align",   multisignal=True),
}

# ── 생성 파라미터 ─────────────────────────────────────────────────────────────
VIS_MODELS   = "2d/clip_l14p"
EMO_MODEL    = "6c_l14p"
SAVED_ROOT   = "./saved_models"
OUTPUT_DIR   = "./output"

TEMPO        = 120
OCTAVE       = 4
VELOCITY     = 100
DURATION     = 2
IS_VOICE     = True
IS_ARP       = True
MAX_CONSEQ_N     = 0
MAX_CONSEQ_CHORD = 2

MIN_LOUDNESS  = 0
MAX_LOUDNESS  = 50
MIN_VELOCITY  = 49
MAX_VELOCITY  = 112

flatsharpDic = {
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
}


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────

def convert_format_id_to_offset(id_list):
    offset_list, current_id, offset = [], id_list[0], 0
    for item in id_list:
        if item != current_id:
            current_id, offset = item, 0
        offset_list.append(offset)
        offset += 1
    return offset_list


def parse_chord(chord_str, chord_dic, chord_root_dic, chord_attr_dic):
    """코드 문자열 → (chordID, rootID, attrID)"""
    c = chord_str
    if len(c) > 1 and c[1] == 'b':
        c = flatsharpDic[c[:2]] + c[2:]
    if len(c) > 1:
        if c[1] == '#':
            c = c[:2] + ':' + c[2:]
        else:
            c = c[:1] + ':' + c[1:]
        suffix_map = {'m': 'min', 'm6': 'min6', 'm7': 'min7',
                      'M6': 'maj6', 'M7': 'maj7', '': ''}
        colon_idx = c.index(':')
        suffix = c[colon_idx+1:]
        if suffix in suffix_map:
            c = c[:colon_idx] + (':' + suffix_map[suffix] if suffix_map[suffix] else '')
    chord_id = chord_dic[c]
    parts = c.split(':')
    root_id = chord_root_dic[parts[0]]
    attr_id = chord_attr_dic[parts[1]] if len(parts) == 2 else 0
    return chord_id, root_id, attr_id


def build_model(cfg, total_vf_dim):
    cTEA.TEA_WHERE        = cfg["where"]
    cTEA.TEA_ENCODER_CELL = cfg["enc"]
    cTEA.TEA_DECODER_CELL = cfg["dec"]
    _tea_module.TEA_WHERE        = cfg["where"]
    _tea_module.TEA_ENCODER_CELL = cfg["enc"]
    _tea_module.TEA_DECODER_CELL = cfg["dec"]

    return VideoMusicTransformerTEA(
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
        use_multisignal    = cfg.get("multisignal", False),
    ).to(get_device())


def chords_to_midi(chord_genlist, chord_offsetlist, density_list, velo_list, tempo,
                   dominant_emo="neutral"):
    # Emotion → (lead GM#, pad GM#, pad velocity reduction)
    _EMO_INSTR = {
        "exciting": dict(lead=0,  pad=48, pad_dv=10),  # Piano + Strings (loud, energetic)
        "fearful":  dict(lead=0,  pad=42, pad_dv=20),  # Piano + Cello (dark, sustained)
        "tense":    dict(lead=0,  pad=42, pad_dv=18),  # Piano + Cello
        "sad":      dict(lead=0,  pad=48, pad_dv=15),  # Piano + Strings (melancholic)
        "relaxing": dict(lead=24, pad=48, pad_dv=20),  # Nylon Guitar + Strings (gentle)
        "neutral":  dict(lead=0,  pad=48, pad_dv=25),  # Piano + Strings (background)
    }
    instr = _EMO_INSTR.get(dominant_emo, _EMO_INSTR["neutral"])

    MIDI = MIDIFile(3)
    for t in range(3):
        MIDI.addTempo(t, 0, tempo)
    MIDI.addProgramChange(0, 0, 0, instr["lead"])  # Ch 0: Lead (piano or nylon guitar)
    MIDI.addProgramChange(1, 1, 0, instr["pad"])   # Ch 1: Pad (strings or cello)
    MIDI.addProgramChange(2, 2, 0, 32)             # Ch 2: Acoustic Bass

    midi_chords_original = []
    for key in chord_genlist:
        key = key.replace(":", "")
        if key == "N":
            midi_chords_original.append([])
        else:
            midi_chords_original.append(Chord(key).getMIDI("c", 4))

    if IS_VOICE:
        midi_chords = voice(midi_chords_original)
    else:
        midi_chords = midi_chords_original

    # voiced chord에 중복 피치가 생기면 arp 내 동일 박자 재공격 artifact 방지.
    # dedup 후 4음 미만이 되면 원본 유지 (arp 코드가 4/5음만 처리하므로).
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
            if density_list[di] == 0:
                if len(chord) == 4:
                    if chord_offsetlist[i] % 2 == 0:
                        MIDI.addNote(0, 0, chord[0], i*d+0,   d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+1,   d, v)
                    else:
                        MIDI.addNote(0, 0, chord[2], i*d+0,   d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1,   d, v)
                elif len(chord) == 5:
                    if chord_offsetlist[i] % 2 == 0:
                        MIDI.addNote(0, 0, chord[0], i*d+0,   d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+1,   d, v)
                    else:
                        MIDI.addNote(0, 0, chord[2], i*d+0,   d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1,   d, v)
            elif density_list[di] == 1:
                if len(chord) == 4:
                    if chord_offsetlist[i] % 2 == 0:
                        MIDI.addNote(0, 0, chord[0], i*d+0,   d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.5, d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1,   d, v)
                    else:
                        MIDI.addNote(0, 0, chord[3], i*d+0,   d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.5, d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1,   d, v)
                elif len(chord) == 5:
                    if chord_offsetlist[i] % 2 == 0:
                        MIDI.addNote(0, 0, chord[0], i*d+0,   d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.5, d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1,   d, v)
                    else:
                        MIDI.addNote(0, 0, chord[3], i*d+0,   d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.5, d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1,   d, v)
            elif density_list[di] == 2:
                if len(chord) == 4:
                    if chord_offsetlist[i] % 2 == 0:
                        MIDI.addNote(0, 0, chord[0], i*d+0,   d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.5, d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1,   d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1.5, d, v)
                    else:
                        MIDI.addNote(0, 0, chord[2], i*d+0,   d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.5, d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1,   d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1.5, d, v)
                elif len(chord) == 5:
                    if chord_offsetlist[i] % 2 == 0:
                        MIDI.addNote(0, 0, chord[0], i*d+0,   d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.5, d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1,   d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1.5, d, v)
                    else:
                        MIDI.addNote(0, 0, chord[2], i*d+0,   d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.5, d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1,   d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1.5, d, v)
            elif density_list[di] == 3:
                if len(chord) == 4:
                    if chord_offsetlist[i] % 2 == 0:
                        MIDI.addNote(0, 0, chord[0], i*d+0,    d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.25, d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+0.5,  d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.75, d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1,    d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1.5,  d, v)
                    else:
                        MIDI.addNote(0, 0, chord[1], i*d+0,    d, v)
                        MIDI.addNote(0, 0, chord[0], i*d+0.25, d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.5,  d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+0.75, d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1,    d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1.5,  d, v)
                elif len(chord) == 5:
                    if chord_offsetlist[i] % 2 == 0:
                        MIDI.addNote(0, 0, chord[0], i*d+0,    d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.25, d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+0.5,  d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.75, d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1,    d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1.5,  d, v)
                    else:
                        MIDI.addNote(0, 0, chord[1], i*d+0,    d, v)
                        MIDI.addNote(0, 0, chord[0], i*d+0.25, d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.5,  d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+0.75, d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1,    d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1.5,  d, v)
            elif density_list[di] == 4:
                if len(chord) == 4:
                    if chord_offsetlist[i] % 2 == 0:
                        MIDI.addNote(0, 0, chord[0], i*d+0,    d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.25, d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+0.5,  d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.75, d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1,    d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1.25, d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+1.5,  d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1.75, d, v)
                    else:
                        MIDI.addNote(0, 0, chord[1], i*d+0,    d, v)
                        MIDI.addNote(0, 0, chord[0], i*d+0.25, d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.5,  d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+0.75, d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1,    d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1.25, d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+1.5,  d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1.75, d, v)
                elif len(chord) == 5:
                    if chord_offsetlist[i] % 2 == 0:
                        MIDI.addNote(0, 0, chord[0], i*d+0,    d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.25, d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+0.5,  d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.75, d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1,    d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1.25, d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+1.5,  d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1.75, d, v)
                    else:
                        MIDI.addNote(0, 0, chord[1], i*d+0,    d, v)
                        MIDI.addNote(0, 0, chord[0], i*d+0.25, d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+0.5,  d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+0.75, d, v)
                        MIDI.addNote(0, 0, chord[3], i*d+1,    d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1.25, d, v)
                        MIDI.addNote(0, 0, chord[1], i*d+1.5,  d, v)
                        MIDI.addNote(0, 0, chord[2], i*d+1.75, d, v)
    else:
        for i, chord in enumerate(midi_chords):
            v = velo_list[min(i, len(velo_list) - 1)]
            for pitch in chord:
                MIDI.addNote(0, 0, pitch, i * DURATION, DURATION, v)

    # Track 1: Pad — sustained block chords (strings or cello), emotion-tuned velocity
    for i, chord in enumerate(midi_chords):
        if not chord:
            continue
        v = velo_list[min(i, len(velo_list) - 1)]
        pad_v = max(40, v - instr["pad_dv"])
        pad_notes = sorted(set(p - 12 for p in chord[1:4]))
        for pitch in pad_notes:
            MIDI.addNote(1, 1, pitch, i * DURATION, DURATION, pad_v)

    # Track 2: Bass — root note 2 octaves below, one note per chord
    for i, chord in enumerate(midi_chords):
        if not chord:
            continue
        v = velo_list[min(i, len(velo_list) - 1)]
        bass_v = max(45, v - 22)
        bass_pitch = max(24, chord[0] - 24)
        MIDI.addNote(2, 2, bass_pitch, i * DURATION, DURATION, bass_v)

    return MIDI


def auto_params(emotion_np):
    """Derive generation params from 6-dim emotion features.

    Dims: [0]=exciting [1]=fearful [2]=tense [3]=sad [4]=relaxing [5]=neutral

    Returns (key, temperature, primer, target_len, dominant_emotion, dark_score).
    Decisions are driven by the dominant emotion, not just a single dark aggregate.
    """
    EMO_NAMES = ["exciting", "fearful", "tense", "sad", "relaxing", "neutral"]

    mean = emotion_np.mean(axis=0)   # [6]  — average intensity per dimension
    std  = emotion_np.std(axis=0)    # [6]  — temporal variability per dimension

    dominant = EMO_NAMES[int(np.argmax(mean))]
    variability = float(np.mean(std))   # overall emotion instability across time

    # dark score for reference/logging (fearful+tense+sad, normalised to 0~1)
    dark_n = float((mean[1] + mean[2] + mean[3]) / 3.0)

    # ── key ────────────────────────────────────────────────────────────────────
    # Negative-dominant → minor; positive/neutral-dominant → major
    _minor_emos = {"fearful", "tense", "sad"}
    _major_emos = {"exciting", "relaxing", "neutral"}
    if dominant in _minor_emos:
        key = "minor"
    elif dominant in _major_emos:
        key = "major"
    else:
        key = "minor" if dark_n > 0.45 else "major"

    # ── temperature ────────────────────────────────────────────────────────────
    # relaxing → stable (low temp);  fearful/tense → unpredictable (high temp)
    _base_temp = {
        "exciting": 1.25,
        "fearful":  1.40,
        "tense":    1.35,
        "sad":      1.25,
        "relaxing": 1.05,
        "neutral":  1.15,
    }
    temperature = round(float(np.clip(
        _base_temp[dominant] + variability * 0.15, 1.0, 1.5
    )), 2)

    # ── primer ─────────────────────────────────────────────────────────────────
    # Each primer reflects the tonal/emotional character of that emotion category
    _primer_map = {
        "exciting": ["G",  "D",  "Em", "C" ],  # bright, energetic
        "fearful":  ["Am", "Dm", "E",  "Am"],  # dark, tension→resolution
        "tense":    ["Dm", "G",  "Am", "E" ],  # suspended, unresolved
        "sad":      ["Am", "C",  "F",  "G" ],  # melancholic, gentle
        "relaxing": ["C",  "G",  "Am", "F" ],  # smooth, chill
        "neutral":  ["C",  "Am", "F",  "G" ],  # safe middle-ground
    }
    primer = _primer_map[dominant]

    # ── target_len ─────────────────────────────────────────────────────────────
    target_len = max(16, min(300, int(emotion_np.shape[0])))

    return key, temperature, primer, target_len, dominant, dark_n


def extract_features_from_video(video_path, device):
    """임의 영상에서 feature를 추출한다 (video2music.py 파이프라인 재사용)."""
    from video2music import (
        split_video_into_frames, gen_semantic_feature, gen_emotion_feature,
        gen_scene_feature, gen_scene_offset_feature, gen_motion_feature,
        get_scene_offset_feature, get_motion_feature,
        get_emotion_feature, get_semantic_feature,
    )

    feature_dir = Path("./feature_tmp")
    if feature_dir.exists():
        shutil.rmtree(str(feature_dir))

    for d in ["vevo_frame", "vevo_semantic", "vevo_emotion",
              "vevo_scene", "vevo_scene_offset", "vevo_motion"]:
        (feature_dir / d).mkdir(parents=True)

    frame_dir        = feature_dir / "vevo_frame"
    semantic_dir     = feature_dir / "vevo_semantic"
    emotion_dir      = feature_dir / "vevo_emotion"
    scene_dir        = feature_dir / "vevo_scene"
    scene_offset_dir = feature_dir / "vevo_scene_offset"
    motion_dir       = feature_dir / "vevo_motion"

    split_video_into_frames(video_path, frame_dir)
    gen_semantic_feature(frame_dir, semantic_dir)
    gen_emotion_feature(frame_dir, emotion_dir)
    gen_scene_feature(video_path, scene_dir, frame_dir)
    gen_scene_offset_feature(scene_dir, scene_offset_dir)
    gen_motion_feature(video_path, motion_dir)

    feature_semantic     = get_semantic_feature(semantic_dir).unsqueeze(0).to(device)
    feature_scene_offset = get_scene_offset_feature(scene_offset_dir).unsqueeze(0).to(device)
    feature_motion       = get_motion_feature(motion_dir).unsqueeze(0).to(device)
    feature_emotion      = get_emotion_feature(emotion_dir).unsqueeze(0).to(device)

    return [feature_semantic], feature_scene_offset, feature_motion, feature_emotion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",      type=int, default=4,
                        help="실험 번호 1~8 (기본: 4, Encoder GRU — Acc+Corr 최고)")
    parser.add_argument("--test_id",  type=str, default="049",
                        help="생성할 영상 ID (기본: 049)")
    parser.add_argument("--key",      type=str, default=None,
                        choices=["major", "minor"],
                        help="키 강제 지정 (기본: 감정 분석 자동)")
    parser.add_argument("--no_primer",action="store_true",
                        help="프라이머 없이 키 기반 단일 코드로 시작")
    parser.add_argument("--primer",   type=str, nargs="+",
                        default=None,
                        help="커스텀 프라이머 코드 목록 (기본: 감정 분석 자동)")
    parser.add_argument("--target_len", type=int, default=None,
                        help="생성할 코드 시퀀스 길이 (기본: 영상 길이 자동)")
    parser.add_argument("--gpu",      type=int, default=None,
                        help="사용할 GPU 번호 (기본: 자동)")
    parser.add_argument("--video",    type=str, default=None,
                        help="임의 영상 경로 또는 YouTube URL (지정 시 --test_id 무시)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="샘플링 temperature (기본: 감정 분석 자동 / >1 다양성 증가)")
    parser.add_argument("--emotion", type=str, nargs="+", default=None,
                        help="감정 지정: 이름('exciting','fearful','tense','sad','relaxing','neutral') "
                             "또는 6개 float. 예: --emotion sad  /  --emotion 0.8 0 0 0.1 0.1 0")
    parser.add_argument("--no_reverb", action="store_true",
                        help="FluidSynth reverb/chorus 비활성화 (dry 사운드)")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    use_cuda(True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    cfg = EXP_CONFIGS[args.exp]
    ckpt = os.path.join(SAVED_ROOT, cfg["dir"], "best_loss_weights.pickle")
    assert os.path.isfile(ckpt), f"체크포인트 없음: {ckpt}"
    print(f"Exp{args.exp} | {cfg['where']} | enc={cfg['enc'].upper()} dec={cfg['dec'].upper()}")
    print(f"체크포인트: {ckpt}\n")

    # ── 영상 소스 분기 ────────────────────────────────────────────────────────
    if args.video:
        video_path = args.video
        if video_path.startswith("http://") or video_path.startswith("https://"):
            dl_path = "/tmp/yt_input.mp4"
            print(f"YouTube 다운로드 중: {video_path}")
            _ytdlp = shutil.which("yt-dlp") or "/home/taegum/miniconda3/bin/yt-dlp"
            # partial 파일 포함 전체 정리
            import glob as _glob
            for _f in _glob.glob("/tmp/yt_input*"):
                try:
                    os.remove(_f)
                except Exception:
                    pass
            # H.264 avc 코덱 우선 (OpenCV AV1 미지원)
            ret = subprocess.run([
                _ytdlp,
                "-f", "bestvideo[vcodec^=avc][ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "--force-overwrites",
                "-o", dl_path,
                video_path,
            ])
            if ret.returncode != 0 or not os.path.exists(dl_path):
                print("[yt-dlp] 1차 포맷 실패, best 포맷으로 재시도...")
                for _f in _glob.glob("/tmp/yt_input*"):
                    try:
                        os.remove(_f)
                    except Exception:
                        pass
                subprocess.run([
                    _ytdlp, "-f", "best", "--merge-output-format", "mp4",
                    "--force-overwrites", "-o", dl_path, video_path,
                ], check=True)
            if not os.path.exists(dl_path):
                raise FileNotFoundError(f"yt-dlp 다운로드 실패: {dl_path} 생성 안 됨")
            video_path = dl_path
        print(f"영상 feature 추출 중: {video_path}")
        feature_semantic_list, feature_scene_offset, feature_motion, feature_emotion = \
            extract_features_from_video(video_path, get_device())
        emotion_np = feature_emotion.squeeze(0).cpu().numpy()  # [seq_len, 6]
        total_vf_dim = 768 + 1 + 1  # CLIP-L/14p + scene_offset + motion (emotion→Linear_emo별도)
        reg_vf_dim   = 768 + 1 + 1 + 6  # VideoRegression은 emotion 포함해 학습됨
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        f_video_in = video_path
        sample_key = None  # 임의 영상은 ground-truth key 없음 → auto로 결정
    else:
        # ── 데이터셋 ──────────────────────────────────────────────────────────
        _, val_dataset, test_dataset = create_vevo_datasets(
            dataset_root    = "./dataset/",
            max_seq_chord   = 300,
            max_seq_video   = 300,
            vis_models      = VIS_MODELS,
            emo_model       = EMO_MODEL,
            split_ver       = SPLIT_VER,
            random_seq      = False,
            is_video        = True,
        )

        with open("dataset/vevo_meta/split/" + SPLIT_VER + "/test.txt") as f:
            test_ids = [l.strip() for l in f]

        dataset = test_dataset if args.test_id in test_ids else val_dataset

        test_id_idx = next(
            (i for i in range(len(dataset))
             if int(args.test_id) == int(dataset.data_files_chord[i].split("/")[-1][:3])),
            None
        )
        assert test_id_idx is not None, f"test_id {args.test_id} 를 데이터셋에서 찾지 못함"
        print(f"데이터셋 인덱스: {test_id_idx}  ({dataset.data_files_chord[test_id_idx]})")

        total_vf_dim = sum(vf.shape[1] for vf in dataset[0]["semanticList"]) + 1 + 1  # emotion→Linear_emo별도
        reg_vf_dim   = total_vf_dim + 6  # VideoRegression은 emotion 포함해 학습됨

        sample = dataset[test_id_idx]
        feature_semantic_list = [
            vf.unsqueeze(0).to(get_device()) for vf in sample["semanticList"]
        ]
        feature_scene_offset = sample["scene_offset"].unsqueeze(0).to(get_device())
        feature_motion       = sample["motion"].unsqueeze(0).to(get_device())
        feature_emotion      = sample["emotion"].unsqueeze(0).to(get_device())
        emotion_np           = sample["emotion"].cpu().numpy()  # [seq_len, 6] for renderer
        video_id = args.test_id
        f_video_in = f"dataset/vevo/{args.test_id}.mp4"
        sample_key = sample["key"]  # ground-truth key; used when --key not specified

    # ── Emotion Override 처리 ─────────────────────────────────────────────────
    _EMO_NAMES = ["exciting", "fearful", "tense", "sad", "relaxing", "neutral"]
    _EMO_PRESETS = {
        "exciting":  np.array([0.90, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32),
        "fearful":   np.array([0.02, 0.90, 0.02, 0.02, 0.02, 0.02], dtype=np.float32),
        "tense":     np.array([0.02, 0.02, 0.90, 0.02, 0.02, 0.02], dtype=np.float32),
        "sad":       np.array([0.02, 0.02, 0.02, 0.90, 0.02, 0.02], dtype=np.float32),
        "relaxing":  np.array([0.02, 0.02, 0.02, 0.02, 0.90, 0.02], dtype=np.float32),
        "neutral":   np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.90], dtype=np.float32),
    }
    if args.emotion is not None:
        if len(args.emotion) == 1 and args.emotion[0] in _EMO_PRESETS:
            emo_arr = _EMO_PRESETS[args.emotion[0]]
        elif len(args.emotion) == 6:
            try:
                emo_arr = np.array([float(v) for v in args.emotion], dtype=np.float32)
            except ValueError:
                raise ValueError(f"--emotion에 숫자 6개 또는 감정 이름 1개를 입력하세요: {_EMO_NAMES}")
        else:
            raise ValueError(f"--emotion에 숫자 6개 또는 감정 이름 1개를 입력하세요: {_EMO_NAMES}")
        emo_arr = emo_arr / (emo_arr.sum() + 1e-8)
        emotion_weight_override = torch.tensor(emo_arr).to(get_device())
        print(f"[Emotion Override] {dict(zip(_EMO_NAMES, emo_arr.round(3).tolist()))}")
    else:
        emo_arr = None
        emotion_weight_override = None

    # ── 자동 파라미터 추론 ────────────────────────────────────────────────────
    # emotion override 시: key/primer/temp도 override 감정 기준으로 추론
    if emo_arr is not None:
        _override_emotion_np = np.tile(emo_arr, (emotion_np.shape[0], 1))
        auto_key, auto_temp, auto_primer, auto_target_len, dominant_emo, dark_score = \
            auto_params(_override_emotion_np)
    else:
        auto_key, auto_temp, auto_primer, auto_target_len, dominant_emo, dark_score = \
            auto_params(emotion_np)

    # 사용자가 명시하지 않은 항목만 auto 값으로 채움
    if args.temperature is None:
        args.temperature = auto_temp
    if args.primer is None and not args.no_primer:
        args.primer = auto_primer
    if args.target_len is None:
        args.target_len = auto_target_len

    # feature_key: --key 명시 > 데이터셋 ground-truth > auto (영상 감정 기반)
    if args.key == "major":
        feature_key = torch.tensor([0.0]).to(get_device())
        final_key_str = "major (forced)"
    elif args.key == "minor":
        feature_key = torch.tensor([1.0]).to(get_device())
        final_key_str = "minor (forced)"
    elif sample_key is not None:
        feature_key = sample_key.to(get_device())   # 데이터셋 ground-truth
        auto_key = "major" if int(sample_key.item()) == 0 else "minor"
        final_key_str = f"{auto_key} (ground-truth)"
    else:
        feature_key = torch.tensor([0.0 if auto_key == "major" else 1.0]).to(get_device())
        final_key_str = f"{auto_key} (auto)"

    print(f"[Auto] dominant={dominant_emo}  dark={dark_score:.2f} | "
          f"key={final_key_str} | temp={args.temperature} | "
          f"primer={args.primer} | target_len={args.target_len}")

    # ── 프라이머 구성 ─────────────────────────────────────────────────────────
    with open("dataset/vevo_meta/chord.json") as f:
        chord_dic = json.load(f)
    with open("dataset/vevo_meta/chord_inv.json") as f:
        chord_inv_dic = json.load(f)
    with open("dataset/vevo_meta/chord_root.json") as f:
        chord_root_dic = json.load(f)
    with open("dataset/vevo_meta/chord_attr.json") as f:
        chord_attr_dic = json.load(f)

    if args.no_primer:
        start_chord = "C" if int(feature_key.item()) == 0 else "A:min"
        cid, rid, aid = parse_chord(start_chord, chord_dic, chord_root_dic, chord_attr_dic)
        primer_ids = torch.tensor([cid], dtype=torch.long).to(get_device())
        primer_roots = torch.tensor([rid], dtype=torch.long).to(get_device())
        primer_attrs = torch.tensor([aid], dtype=torch.long).to(get_device())
        num_prime = 1
    else:
        ids, roots, attrs = [], [], []
        for p in args.primer:
            cid, rid, aid = parse_chord(p, chord_dic, chord_root_dic, chord_attr_dic)
            ids.append(cid); roots.append(rid); attrs.append(aid)
        primer_ids   = torch.tensor(ids,   dtype=torch.long).to(get_device())
        primer_roots = torch.tensor(roots, dtype=torch.long).to(get_device())
        primer_attrs = torch.tensor(attrs, dtype=torch.long).to(get_device())
        num_prime = len(args.primer)

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    model = build_model(cfg, total_vf_dim)
    model.load_state_dict(torch.load(ckpt, map_location=get_device(), weights_only=False))
    model.eval()
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── 코드 시퀀스 생성 ──────────────────────────────────────────────────────
    print("코드 시퀀스 생성 중...")
    with torch.no_grad():
        rand_seq = model.generate(
            feature_semantic_list  = feature_semantic_list,
            feature_key            = feature_key,
            feature_scene_offset   = feature_scene_offset,
            feature_motion         = feature_motion,
            feature_emotion        = feature_emotion,
            primer                 = primer_ids,
            primer_root            = primer_roots,
            primer_attr            = primer_attrs,
            target_seq_length      = args.target_len,
            beam                   = 0,
            max_conseq_N           = MAX_CONSEQ_N,
            max_conseq_chord       = MAX_CONSEQ_CHORD,
            temperature            = args.temperature,
            emotion_weight_override= emotion_weight_override,
        )

    chord_id_list  = rand_seq[0].cpu().numpy()
    chord_genlist  = [chord_inv_dic[str(i)] for i in chord_id_list]
    chord_offsetlist = convert_format_id_to_offset(chord_genlist)
    print(f"생성 코드 수: {len(chord_genlist)}")
    print(f"샘플 (앞 10개): {chord_genlist[:10]}\n")

    # ── Regression 모델로 velocity / density 예측 ────────────────────────────
    reg_weights = os.path.join(SAVED_ROOT, "AMT", "best_rmse_weights.pickle")
    if os.path.isfile(reg_weights):
        modelReg = VideoRegression(
            max_sequence_video = 300,
            total_vf_dim       = reg_vf_dim,
            regModel           = "bigru",
        ).to(get_device())
        modelReg.load_state_dict(torch.load(reg_weights, map_location=get_device(), weights_only=False))
        modelReg.eval()

        # 정규화 stats 로드 (재학습 후 z-score 모델)
        import json as _json
        _norm_path = os.path.join(SAVED_ROOT, "AMT", "norm_stats.json")
        if os.path.isfile(_norm_path):
            with open(_norm_path) as _f:
                _ns = _json.load(_f)
            nd_mean, nd_std = _ns["nd_mean"], _ns["nd_std"]
            lv_mean, lv_std = _ns["lv_mean"], _ns["lv_std"]
        else:
            nd_mean, nd_std, lv_mean, lv_std = 0.0, 1.0, 0.0, 1.0

        with torch.no_grad():
            y = modelReg(feature_semantic_list, feature_scene_offset,
                         feature_motion, feature_emotion)
            y = y.reshape(y.shape[0] * y.shape[1], -1)
            y_density, y_loudness = torch.split(y, 1, dim=1)

        # 역정규화 → 원래 스케일 복원
        y_density_raw = y_density.cpu().numpy() * nd_std + nd_mean
        y_loudness_raw = y_loudness.cpu().numpy() * lv_std + lv_mean

        density_np  = np.clip(np.round(y_density_raw + np.random.normal(0, 4.0, y_density_raw.shape)).astype(int), 0, 40)
        loudness_np = np.clip((y_loudness_raw * 100 + np.random.normal(0, 8.0, y_loudness_raw.shape)).astype(int), 0, 50)
        exponent = 0.3
        velo_list = [
            int(np.round(((l[0] - MIN_LOUDNESS) / (MAX_LOUDNESS - MIN_LOUDNESS)) ** exponent
                         * (MAX_VELOCITY - MIN_VELOCITY) + MIN_VELOCITY))
            for l in loudness_np
        ]
        # 영상 끝 패딩 구간의 랜덤 저속도 스파이크 방지:
        # 5-frame 이동평균으로 스파이크 제거 후 최소 floor 적용
        _velo_arr = np.array(velo_list, dtype=float)
        _kernel   = np.ones(5) / 5
        _velo_arr = np.convolve(_velo_arr, _kernel, mode='same')
        _vel_floor = max(int(np.median(_velo_arr)), MIN_VELOCITY + 20)
        velo_list  = [max(int(v), _vel_floor) for v in _velo_arr]
        density_list = [
            0 if d[0] <= 5 else 1 if d[0] <= 10 else 2 if d[0] <= 15 else 3 if d[0] <= 20 else 4
            for d in density_np
        ]
    else:
        print("  ⚠ Regression 가중치 없음 — velocity/density 기본값 사용")
        velo_list    = [VELOCITY] * len(chord_genlist)
        density_list = [2]       * len(chord_genlist)

    # ── 출력 경로 ──────────────────────────────────────────────────────────────
    _emo_tag = ""
    if emo_arr is not None:
        EMO_TAGS = ["ex", "fe", "te", "sa", "re", "ne"]
        dominant_idx = int(np.argmax(emo_arr))
        _emo_tag = f"_emo_{EMO_TAGS[dominant_idx]}"
    suffix   = f"_TEA_exp{args.exp}" + (args.key if args.key else "") + _emo_tag
    out_dir  = os.path.join(OUTPUT_DIR, video_id)
    os.makedirs(out_dir, exist_ok=True)

    f_midi      = os.path.join(out_dir, f"{video_id}{suffix}_cgen_rd.mid")
    f_lab       = os.path.join(out_dir, f"{video_id}{suffix}_cgen_rd.lab")
    f_flac      = os.path.join(out_dir, f"{video_id}{suffix}_cgen_rd.flac")
    f_video_out = os.path.join(out_dir, f"{video_id}{suffix}_cgen_rd.mp4")

    # ── Lab 저장 ──────────────────────────────────────────────────────────────
    with open(f_lab, "w", encoding="utf-8") as f:
        f.write("key ?\n")
        for i, c in enumerate(chord_genlist):
            f.write(f"{i} {c}\n")
    print(f"Lab 저장: {f_lab}")

    # ── MIDI 생성 ─────────────────────────────────────────────────────────────
    midi = chords_to_midi(chord_genlist, chord_offsetlist, density_list, velo_list, TEMPO,
                          dominant_emo=dominant_emo)
    with open(f_midi, "wb") as f:
        midi.writeFile(f)
    print(f"MIDI 저장: {f_midi}")

    # ── MIDI → FLAC ───────────────────────────────────────────────────────────
    print("FLAC 변환 중...")
    _sf2_candidates = [
        "./soundfonts/default_sound_font.sf2",            # baseline 동일 SF2
        "/home/taegum/sf2/MuseScore_General_Full.sf3",
        "/usr/share/sounds/sf2/MuseScore_General.sf2",
        "/usr/share/sounds/sf2/MuseScore_General.sf3",
        "/home/taegum/sf2/MuseScore_General.sf2",
        "/home/taegum/sf2/FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/FluidR3_GS.sf2",
        "/home/taegum/miniconda3/envs/video2music/lib/python3.8/site-packages/pretty_midi/TimGM6mb.sf2",
    ]
    _sf2 = next((p for p in _sf2_candidates if os.path.isfile(p)), _sf2_candidates[-1])
    print(f"  SF2: {_sf2}")
    _fs_candidates = [
        shutil.which("fluidsynth"),
        "/home/taegum/miniconda3/envs/video2music/bin/fluidsynth",
        "/usr/bin/fluidsynth",
    ]
    _fs_bin = next((p for p in _fs_candidates if p and os.path.isfile(p)), "fluidsynth")
    _fs_cmd = [_fs_bin, "-ni", _sf2, f_midi, "-F", f_flac, "-r", "44100"]
    if args.no_reverb:
        _fs_cmd += ["-R", "0", "-C", "0"]   # reverb=off, chorus=off
        print("  [dry] reverb/chorus 비활성화")
    fs_ret = subprocess.call(_fs_cmd)
    if fs_ret != 0:
        print(f"[ERROR] FluidSynth 실패 (exit {fs_ret}) — FLAC 미생성, 이후 단계 생략")
        return
    print(f"FLAC 저장: {f_flac}")

    # ── 영상 합성 ─────────────────────────────────────────────────────────────
    if os.path.isfile(f_video_in):
        print("영상 합성 중...")
        # MoviePy 대신 ffmpeg 직접 사용 — AV1 포함 모든 코덱 지원
        subprocess.run([
            "ffmpeg", "-y",
            "-i", f_video_in,
            "-i", f_flac,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            f_video_out,
        ], check=True)
        print(f"영상 저장: {f_video_out}")
    else:
        print(f"  ⚠ 원본 영상 없음 ({f_video_in}), 영상 합성 생략")

    print("\n완료.")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import os
import random

from third_party.midi_processor.processor import decode_midi, encode_midi
from utilities.argument_funcs import parse_generate_args, print_generate_args
from music21 import harmony as m21harmony

from model.music_transformer import MusicTransformer
from model.video_music_transformer import VideoMusicTransformer
from model.video_regression import VideoRegression

from dataset.vevo_dataset import compute_vevo_accuracy, create_vevo_datasets

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda
import numpy as np
import json

from midi2audio import FluidSynth
from midiutil import MIDIFile

import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os, random, shutil
from moviepy.editor import *
import time

version = VERSION
split_ver = SPLIT_VER
split_path = "split_" + split_ver
num_prime_chord = 30

is_voice = True
isArp = True
duration = 2
tempo = 120

_MIN_VELOCITY = 49
_MAX_VELOCITY = 112

_FLAT_TO_SHARP = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}

def _m21_pitches(chord_label, octave=4):
    label = chord_label.replace(':', '').replace('hdim7', 'm7b5')
    for flat, sharp in _FLAT_TO_SHARP.items():
        if label.startswith(flat):
            label = sharp + label[len(flat):]
            break
    if not label or label == 'N':
        return []
    try:
        cs = m21harmony.ChordSymbol(label)
        closed = cs.closedPosition(forceOctave=octave, inPlace=False)
        return sorted(set(p.midi for p in closed.pitches))
    except Exception:
        return []

def _voice_lead(curr, prev):
    if not prev:
        return curr
    center = sum(prev) / len(prev)
    voiced = []
    for p in curr:
        best = p
        for s in (-2, -1, 0, 1, 2):
            c = p + s * 12
            if abs(c - center) < abs(best - center):
                best = c
        voiced.append(best)
    return sorted(voiced)

def _hv(v, spread=7):
    return int(np.clip(int(v) + np.random.randint(-spread, spread + 1), 1, 127))

def chords_to_midi(chord_genlist, chord_offsetlist, density_list, velo_list, tempo,
                   emotion_per_chord=None):
    midi = MIDIFile(2)
    for tr in range(2):
        midi.addTempo(tr, 0, tempo)

    n = len(chord_genlist)
    if emotion_per_chord is not None and len(emotion_per_chord) > 0:
        raw_dark = np.array([
            float(emotion_per_chord[min(i, len(emotion_per_chord) - 1)][1]
                  + emotion_per_chord[min(i, len(emotion_per_chord) - 1)][2]
                  + emotion_per_chord[min(i, len(emotion_per_chord) - 1)][3])
            for i in range(n)
        ])
        dark_smooth = np.convolve(raw_dark, np.ones(8) / 8, mode='same')
    else:
        dark_smooth = np.full(n, 0.4)

    PHRASE_LEN_CYCLE = [4, 4, 3, 5, 4, 3, 4, 4]
    phrase_pos_arr = np.zeros(n, dtype=int)
    phrase_len_arr = np.zeros(n, dtype=int)
    j_p, pi = 0, 0
    while j_p < n:
        pl = PHRASE_LEN_CYCLE[pi % len(PHRASE_LEN_CYCLE)]
        for pos in range(pl):
            if j_p < n:
                phrase_pos_arr[j_p] = pos
                phrase_len_arr[j_p] = pl
                j_p += 1
        pi += 1

    prev_rh = []
    for i in range(n):
        t = i * duration
        vi = min(i, len(velo_list) - 1)
        v_base = int(np.clip(velo_list[vi], _MIN_VELOCITY, _MAX_VELOCITY))

        phrase_pos = phrase_pos_arr[i]
        phrase_len = phrase_len_arr[i]
        pos_norm = phrase_pos / max(phrase_len - 1, 1)
        dark = float(dark_smooth[i])

        v = v_base + int(6 * np.sin(np.pi * pos_norm))
        v = int(np.clip(v + (0.5 - dark) * 10, 1, 127))

        rh = _voice_lead(_m21_pitches(chord_genlist[i], octave=4), prev_rh)
        lh_base = _m21_pitches(chord_genlist[i], octave=2)
        if not rh or not lh_base:
            prev_rh = rh
            continue

        bass = lh_base[0]
        while bass > 48: bass -= 12
        while bass < 36: bass += 12

        lh_inner = []
        for p in lh_base[1:]:
            p2 = p
            while p2 < 48: p2 += 12
            while p2 > 60: p2 -= 12
            if p2 != bass and p2 not in lh_inner:
                lh_inner.append(p2)
        lh_inner = lh_inner[:2]

        # ---- LEFT HAND: beat 1 bass + beat 2 chord shell ----
        midi.addNote(0, 0, bass, t, duration * 0.75, _hv(max(v - 5, 30)))
        for j, p in enumerate(lh_inner):
            midi.addNote(0, 0, p, t + duration * 0.5 + j * 0.02,
                         duration * 0.42, _hv(max(v - 14, 22)))

        # ---- RIGHT HAND: slight upward roll for natural piano feel ----
        top = rh[-1]
        n_voiced = len(rh)
        for j, p in enumerate(rh):
            roll_t = t + j * 0.025
            dur = duration * (0.85 if dark > 0.55 else 0.70)
            if p == top:
                midi.addNote(1, 0, p, roll_t, dur * 1.05, _hv(v))
            else:
                inner_vel = max(v - 10 - (n_voiced - 1 - j) * 3, 22)
                midi.addNote(1, 0, p, roll_t, dur * 0.90, _hv(inner_vel))

        prev_rh = rh

    return midi
octave = 4
velocity = 100
subdivide = 1
key = "c"
isPrimer = False

min_loudness = 0  # Minimum loudness level in the input range
max_loudness = 50  # Maximum loudness level in the input range
min_velocity = 49  # Minimum velocity value in the output range
max_velocity = 112  # Maximum velocity value in the output range

custumPrimer = ["C","Am","Dm","G"]
custumKey = "" 
#minor or major

flatsharpDic = {
    'Db':'C#', 
    'Eb':'D#', 
    'Gb':'F#', 
    'Ab':'G#', 
    'Bb':'A#'
}

regModel = "bigru"

max_conseq_N = 0
max_conseq_chord = 2

def text_clip(text: str, duration: int, start_time: int = 0):
    t = TextClip(text, font='/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', fontsize=24, color='white')
    t = t.set_position(("center", 20)).set_duration(duration)
    t = t.set_start(start_time)
    return t

def convert_format_id_to_offset(id_list):
    offset_list = []
    current_id = id_list[0]
    offset = 0
    for i in range(len(id_list)):
        if id_list[i] != current_id:
            current_id = id_list[i]
            offset = 0
        offset_list.append(offset)
        offset += 1
    return offset_list

def main():
    testFileList=[]
    valFileList=[]
    
    with open('dataset/vevo_meta/split/'+ split_ver +'/test.txt') as txt_file:
        for line in txt_file:
            testFileList.append(line.strip())
    with open('dataset/vevo_meta/split/'+ split_ver +'/val.txt') as txt_file:
        for line in txt_file:
            valFileList.append(line.strip())
    
    with open('dataset/vevo_meta/chord.json') as json_file:
        chordDic = json.load(json_file)
    with open('dataset/vevo_meta/chord_inv.json') as json_file:
        chordInvDic = json.load(json_file)

    with open('dataset/vevo_meta/chord_root.json') as json_file:
        chordRootDic = json.load(json_file)
    with open('dataset/vevo_meta/chord_attr.json') as json_file:
        chordAttrDic = json.load(json_file)


    args = parse_generate_args()

    args.num_prime_chord = num_prime_chord

    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")
    else:
        use_cuda(True)

    os.makedirs(args.output_dir, exist_ok=True)
    

    _, val_dataset, test_dataset = create_vevo_datasets(
            dataset_root = "./dataset/", 
            max_seq_chord = args.max_sequence_chord, 
            max_seq_video = args.max_sequence_video, 
            vis_models = args.vis_models,
            emo_model = args.emo_model, 
            split_ver = SPLIT_VER, 
            random_seq = False, 
            is_video = args.is_video)
    
    if args.test_id is None or args.test_id in testFileList:
        dataset = test_dataset
    elif args.test_id in valFileList:
        dataset = val_dataset
    else:
        assert False, f"Test id: {args.test_id} not in test or val dataset"
    
    total_vf_dim = 0
    if args.is_video:
        for vf in dataset[0]["semanticList"]:
            total_vf_dim += vf.shape[1]
    
    total_vf_dim += 1 # Scene_offset
    total_vf_dim += 1 # Motion
    reg_vf_dim = total_vf_dim + 6  # VideoRegression은 emotion concat 방식 유지

    test_id_idx = -1
    if(args.test_id is None):
        test_id_idx = int(random.randrange(len(dataset)))
    else:
        test_id_idx = -1
        for i in range( len(dataset) ):
            if int(args.test_id) == int( dataset.data_files_chord[i].split("/")[-1][:3] ):
                test_id_idx = i
        if test_id_idx == -1:
            assert False, f"Test id: {args.test_id} not in test dataset"

    # Resolve actual numeric ID from filename (handles args.test_id is None case)
    test_id_str = dataset.data_files_chord[test_id_idx].split("/")[-1][:3]

    primer = dataset[test_id_idx]["x"].to(get_device())
    primer_root = dataset[test_id_idx]["x_root"].to(get_device())
    primer_attr = dataset[test_id_idx]["x_attr"].to(get_device())
    
    feature_semantic_list = [] 
    for feature_semantic in dataset[test_id_idx]["semanticList"]:
        feature_semantic = torch.unsqueeze(feature_semantic, 0)
        feature_semantic_list.append( feature_semantic.to(get_device()) )

    feature_scene_offset = dataset[test_id_idx]["scene_offset"].to(get_device())
    feature_motion = dataset[test_id_idx]["motion"].to(get_device())
    feature_emotion = dataset[test_id_idx]["emotion"].to(get_device())

    feature_scene_offset = feature_scene_offset.unsqueeze(0)
    feature_motion = feature_motion.unsqueeze(0)
    feature_emotion = feature_emotion.unsqueeze(0)

    # Per-frame emotion probabilities for renderer: [seq_len, 6]
    emotion_np = feature_emotion.squeeze(0).cpu().numpy()

    if args.is_video:
        vispath = VIS_MODELS_PATH
    else:
        vispath = "no_video"
                
    os.makedirs(os.path.join(args.output_dir, test_id_str), exist_ok=True)
    print("Using primer index:", test_id_idx, "(", dataset.data_files_chord[test_id_idx], ")")

    if "major" in custumKey:
        feature_key = torch.tensor([0])
        feature_key = feature_key.float()
    elif "minor" in custumKey:
        feature_key = torch.tensor([1])
        feature_key = feature_key.float()
    else:
        feature_key = dataset[test_id_idx]["key"]
    feature_key = feature_key.to(get_device())

    if args.is_video:
        model = VideoMusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                    max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, 
                    max_sequence_chord=args.max_sequence_chord, total_vf_dim=total_vf_dim, rpr=args.rpr).to(get_device())
        
        model.load_state_dict(torch.load(args.model_weights))
        if os.path.isfile(args.modelReg_weights):
            modelReg = VideoRegression(max_sequence_video=args.max_sequence_video, total_vf_dim=reg_vf_dim, regModel= regModel).to(get_device())
            modelReg.load_state_dict(torch.load(args.modelReg_weights))
        else:
            print(f"WARNING: Regression weights not found ({args.modelReg_weights}). Using default velocity=100, density=2.")
            modelReg = None
    else:
        model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                    max_sequence_midi=args.max_sequence_midi, max_sequence_chord=args.max_sequence_chord, rpr=args.rpr).to(get_device())

    if not isPrimer:
        primerCID = []
        primerCID_root = []
        primerCID_attr = []

        args.num_prime_chord = 1
        if int( feature_key.item() ) == 0:
            primer_user = "C"
        else:
            primer_user = "A:min"
        
        chordID = chordDic[primer_user]
        primerCID.append(chordID)

        chord_arr = primer_user.split(":")
        if len(chord_arr) == 1:
            chordRootID = chordRootDic[chord_arr[0]]
            primerCID_root.append(chordRootID)
            primerCID_attr.append(0)
        elif len(chord_arr) == 2:
            chordRootID = chordRootDic[chord_arr[0]]
            chordAttrID = chordAttrDic[chord_arr[1]]
            primerCID_root.append(chordRootID)
            primerCID_attr.append(chordAttrID)
        
        primerCID = np.array(primerCID)
        primerCID = torch.from_numpy(primerCID)
        primerCID = primerCID.to(torch.long)
        primerCID = primerCID.to(get_device())

        primerCID_root = np.array(primerCID_root)
        primerCID_root = torch.from_numpy(primerCID_root)
        primerCID_root = primerCID_root.to(torch.long)
        primerCID_root = primerCID_root.to(get_device())
        
        primerCID_attr = np.array(primerCID_attr)
        primerCID_attr = torch.from_numpy(primerCID_attr)
        primerCID_attr = primerCID_attr.to(torch.long)
        primerCID_attr = primerCID_attr.to(get_device())
    else:
        if len(custumPrimer) >= 1:
            primerCID = []
            primerCID_root = []
            primerCID_attr = []
            
            for pChord in custumPrimer:
                if len(pChord) > 1:
                    if pChord[1] == "b":
                        pChord = flatsharpDic [ pChord[0:2] ] + pChord[2:]
                    type_idx = 0
                    if pChord[1] == "#":
                        pChord = pChord[0:2] + ":" + pChord[2:]
                        type_idx = 2
                    else:
                        pChord = pChord[0:1] + ":" + pChord[1:]
                        type_idx = 1
                    if pChord[type_idx+1:] == "m":
                        pChord = pChord[0:type_idx] + ":min"
                    if pChord[type_idx+1:] == "m6":
                        pChord = pChord[0:type_idx] + ":min6"
                    if pChord[type_idx+1:] == "m7":
                        pChord = pChord[0:type_idx] + ":min7"
                    if pChord[type_idx+1:] == "M6":
                        pChord = pChord[0:type_idx] + ":maj6"
                    if pChord[type_idx+1:] == "M7":
                        pChord = pChord[0:type_idx] + ":maj7"
                    if pChord[type_idx+1:] == "":
                        pChord = pChord[0:type_idx]

                chordID = chordDic[pChord]
                primerCID.append(chordID)

                chord_arr = pChord.split(":")
                if len(chord_arr) == 1:
                    chordRootID = chordRootDic[chord_arr[0]]
                    primerCID_root.append(chordRootID)
                    primerCID_attr.append(0)
                elif len(chord_arr) == 2:
                    chordRootID = chordRootDic[chord_arr[0]]
                    chordAttrID = chordAttrDic[chord_arr[1]]
                    primerCID_root.append(chordRootID)
                    primerCID_attr.append(chordAttrID)

            primerCID = np.array(primerCID)
            primerCID = torch.from_numpy(primerCID)
            primerCID = primerCID.to(torch.long)
            primerCID = primerCID.to(get_device())

            primerCID_root = np.array(primerCID_root)
            primerCID_root = torch.from_numpy(primerCID_root)
            primerCID_root = primerCID_root.to(torch.long)
            primerCID_root = primerCID_root.to(get_device())
            
            primerCID_attr = np.array(primerCID_attr)
            primerCID_attr = torch.from_numpy(primerCID_attr)
            primerCID_attr = primerCID_attr.to(torch.long)
            primerCID_attr = primerCID_attr.to(get_device())

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        if(args.beam > 0):
            print("BEAM:", args.beam)
            assert False, "No Beam sampling method implemented yet..."
        else:
            print("RAND DIST")

            if custumKey != "":
                f_path_midi = os.path.join(args.output_dir, test_id_str, test_id_str + custumKey + "_cgen_rd.mid")
                f_path_lab = os.path.join(args.output_dir, test_id_str, test_id_str + custumKey + "_cgen_rd.lab")
                f_path_flac = os.path.join(args.output_dir, test_id_str, test_id_str + custumKey + "_cgen_rd.flac")
                f_path_video = "dataset/vevo/" + test_id_str + ".mp4"
                f_path_video_out = os.path.join(args.output_dir, test_id_str, test_id_str + custumKey + "_cgen_rd.mp4")
            else:
                f_path_midi = os.path.join(args.output_dir, test_id_str, test_id_str + "_cgen_rd.mid")
                f_path_lab = os.path.join(args.output_dir, test_id_str, test_id_str + "_cgen_rd.lab")
                f_path_flac = os.path.join(args.output_dir, test_id_str, test_id_str + "_cgen_rd.flac")
                f_path_video = "dataset/vevo/" + test_id_str + ".mp4"
                f_path_video_out = os.path.join(args.output_dir, test_id_str, test_id_str + "_cgen_rd.mp4")

            if args.is_video:
                if isPrimer and len(custumPrimer) == 0:
                    rand_seq = model.generate(feature_semantic_list=feature_semantic_list, 
                                              feature_key=feature_key, 
                                              feature_scene_offset=feature_scene_offset,
                                              feature_motion=feature_motion,
                                              feature_emotion=feature_emotion,
                                              primer = primer[:args.num_prime_chord],
                                              primer_root = primer_root[:args.num_prime_chord],
                                              primer_attr = primer_attr[:args.num_prime_chord],
                                              target_seq_length = args.target_seq_length_chord, 
                                              beam=0,
                                              max_conseq_N= max_conseq_N,
                                              max_conseq_chord = max_conseq_chord)
                else:
                    rand_seq = model.generate(feature_semantic_list=feature_semantic_list, 
                                              feature_key=feature_key, 
                                              feature_scene_offset=feature_scene_offset,
                                              feature_motion=feature_motion,
                                              feature_emotion=feature_emotion,
                                              primer = primerCID, 
                                              primer_root = primerCID_root,
                                              primer_attr = primerCID_attr,
                                              target_seq_length = args.target_seq_length_chord, 
                                              beam=0,
                                              max_conseq_N= max_conseq_N,
                                              max_conseq_chord = max_conseq_chord)
                vispath = VIS_MODELS_PATH
                if modelReg is not None:
                    modelReg.eval()
                    with torch.set_grad_enabled(False):
                        y = modelReg(
                            feature_semantic_list,
                            feature_scene_offset,
                            feature_motion,
                            feature_emotion)

                        y   = y.reshape(y.shape[0] * y.shape[1], -1)
                        y_note_density, y_loudness = torch.split(y, split_size_or_sections=1, dim=1)

                    y_note_density_np = y_note_density.cpu().numpy()
                    y_note_density_np = np.round(y_note_density_np).astype(int)
                    y_note_density_np = np.clip(y_note_density_np, 0, 40)

                    y_loudness_np = y_loudness.cpu().numpy()
                    y_loudness_np_lv = (y_loudness_np * 100).astype(int)
                    y_loudness_np_lv = np.clip(y_loudness_np_lv, 0, 50)
                    velolistExp = []
                    exponent = 0.3
                    for item in y_loudness_np_lv:
                        loudness = item[0]
                        velocity_exp = np.round(((loudness - min_loudness) / (max_loudness - min_loudness)) ** exponent * (max_velocity - min_velocity) + min_velocity)
                        velocity_exp = int(velocity_exp)
                        velolistExp.append(velocity_exp)

                    densitylist = []
                    for item in y_loudness_np_lv:
                        density = item[0]
                        if density <= 5:
                            densitylist.append(0)
                        elif density <= 10:
                            densitylist.append(1)
                        elif density <= 15:
                            densitylist.append(2)
                        elif density <= 20:
                            densitylist.append(3)
                        else:
                            densitylist.append(4)
                else:
                    # regression 가중치 없을 때 기본값 사용 (chord 수는 이후 확정됨)
                    velolistExp = None
                    densitylist = None

                # generated ChordID to ChordSymbol
                chord_genlist = []
                chordID_genlist= rand_seq[0].cpu().numpy()
                for i in chordID_genlist:
                    chord_genlist.append(chordInvDic[str(i)])

                chord_offsetlist = convert_format_id_to_offset(chord_genlist)

                if velolistExp is None:
                    velolistExp = [velocity] * len(chord_genlist)
                if densitylist is None:
                    densitylist = [2] * len(chord_genlist)

                # Write lab file
                with open(f_path_lab,'w',encoding = 'utf-8') as f:
                    f.write("key ?"+"\n")
                    for i in range(0, len(chord_genlist)):
                        f.write(str(i) + " "+chord_genlist[i]+"\n")

                # ChordSymbol to MIDI file with voicing
                MIDI = chords_to_midi(chord_genlist, chord_offsetlist, densitylist, velolistExp, tempo,
                                      emotion_per_chord=emotion_np)
            else:
                if isPrimer and len(custumPrimer) == 0:
                    rand_seq = model.generate(feature_key=feature_key, 
                                              primer = primer[:args.num_prime_chord],
                                              primer_root = primer_root[:args.num_prime_chord],
                                              primer_attr = primer_attr[:args.num_prime_chord],
                                              target_seq_length = args.target_seq_length_chord, 
                                              beam=0)
                else:
                    rand_seq = model.generate(feature_key=feature_key, 
                                              primer = primerCID, 
                                              primer_root = primerCID_root,
                                              primer_attr = primerCID_attr,
                                              target_seq_length = args.target_seq_length_chord, 
                                              beam=0)
                vispath = "no_video"

                chord_genlist = []
                chordID_genlist= rand_seq[0].cpu().numpy()
                for i in chordID_genlist:
                    chord_genlist.append(chordInvDic[str(i)])

                chord_offsetlist = convert_format_id_to_offset(chord_genlist)

                velolistExp = [velocity] * len(chord_genlist)
                densitylist = [2] * len(chord_genlist)

                # Write lab file
                with open(f_path_lab,'w',encoding = 'utf-8') as f:
                    f.write("key ?"+"\n")
                    for i in range(0, len(chord_genlist)):
                        f.write(str(i) + " "+chord_genlist[i]+"\n")
                
                # ChordSymbol to MIDI file with voicing (no-video: emotion unavailable)
                MIDI = chords_to_midi(chord_genlist, chord_offsetlist, densitylist, velolistExp, tempo)

            # Write midi file
            with open(f_path_midi, "wb") as outputFile:
                MIDI.writeFile(outputFile)
            
            # Convert midi to audio (e.g., flac)
            _sf2_candidates = [
                "./soundfonts/default_sound_font.sf2",             # 기본 (generate_TEA.py와 통일)
                "/home/taegum/sf2/FluidR3_GM.sf2",
                "/usr/share/sounds/sf2/FluidR3_GM.sf2",
                "/home/taegum/sf2/MuseScore_General_Full.sf3",
                "/home/taegum/miniconda3/envs/video2music/lib/python3.8/site-packages/pretty_midi/TimGM6mb.sf2",
            ]
            _sf2 = next((p for p in _sf2_candidates if os.path.isfile(p)), _sf2_candidates[-1])
            print(f"  SF2: {_sf2}")
            import subprocess, shutil as _shutil
            _fs_candidates = [
                _shutil.which("fluidsynth"),
                "/home/taegum/miniconda3/envs/video2music/bin/fluidsynth",
                "/usr/bin/fluidsynth",
            ]
            _fs_bin = next((p for p in _fs_candidates if p and os.path.isfile(p)), "fluidsynth")
            subprocess.call([_fs_bin, "-ni", _sf2, f_path_midi, "-F", f_path_flac, "-r", "44100"])

            # Render generated music into input video (skip if video file missing)
            if os.path.isfile(f_path_video):
                audio=mp.AudioFileClip(f_path_flac)
                video=mp.VideoFileClip(f_path_video)
                audio = audio.subclip(0, video.duration )
                final=video.set_audio(audio)

                final.write_videofile(f_path_video_out,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True
                )
            else:
                print(f"  Video file not found, skipping video render: {f_path_video}")

if __name__ == "__main__":
    main()
    

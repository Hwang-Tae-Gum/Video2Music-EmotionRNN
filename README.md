# Video2Music + TEA (Temporal Emotion Adapter)

This repository extends [AMAAI-Lab/Video2Music](https://github.com/AMAAI-Lab/Video2Music) (AMT) with **TEA**, a temporal emotion-conditioning module for video-to-chord generation.

The original AMT model injects a video's emotion signal (a 6-dim vector per frame) into the Transformer by simple concatenation. TEA instead processes the emotion sequence **temporally** — through a windowed, causally-decayed RNN (or self-attention) adapter — before injecting it into the encoder and/or decoder, giving the model a richer sense of how emotion evolves over the clip instead of a per-frame snapshot.

Full experiment log (12+ architecture/loss variants, ablations, failure analysis) is in [`TEA_report.md`](TEA_report.md).

## Demo

Same 20s clip (Taylor Swift – *Wildest Dreams*, auto-selected highlight segment), baseline vs. TEA-generated chords/music:

| Baseline (AMT) | TEA (MS-TEA + AlignLoss, attention) |
|---|---|
| <video src="assets/demo/baseline_vs_tea_baseline.mp4" controls width="320"></video> | <video src="assets/demo/baseline_vs_tea_TEA.mp4" controls width="320"></video> |

(If the players don't render, download directly: [baseline](assets/demo/baseline_vs_tea_baseline.mp4) / [TEA](assets/demo/baseline_vs_tea_TEA.mp4))

## Architecture

```
input: feature_emotion (batch, seq_len, 6)
 │
 ├─ [1] learnable signal_weight (6-dim) + temperature scaling
 │       → softmax-weighted importance per emotion dimension
 ├─ [2] input dropout (p=0.2)
 ├─ [3] temporal window (size=2, exponential decay, fully vectorized via unfold)
 ├─ [4] Bidirectional RNN (LSTM/GRU) or self-attention
 │       → hidden = d_model // 2 (bidirectional → output dim = d_model)
 ├─ [5] residual + LayerNorm
 └─ [6] output projection (Linear d_model → d_model)

output: emotion_embedding (batch, seq_len, 512)
```

TEA can be injected at the **encoder** (additive, into video features), the **decoder** (cross-attention), or **both**. MS-TEA extends the same adapter to scene-offset and motion signals (multi-signal), and **AlignLoss** adds a direct MSE penalty on emotion-valence alignment during fine-tuning.

## Results

Evaluated on the VEVO validation split (`dataset/vevo_meta/split/v1`). H@k = chord hit-rate@k, AR = Affective Correspondence (agreement between generated chord quality and ground-truth video emotion).

| Model | H@1 | H@3 | H@5 | AR |
|---|---|---|---|---|
| AMT (no emotion) | 0.499 | 0.781 | 0.879 | – |
| AMT (paper baseline, reproduced) | 0.509 | 0.786 | 0.882 | 0.466 |
| **TEA (encoder, GRU)** | **0.609** | 0.853 | 0.918 | 0.470 |
| MS-TEA + AlignLoss (λ=0.15, GRU) | 0.566 | 0.831 | 0.907 | 0.457 |
| **MS-TEA + AlignLoss (attention)** | 0.567 | 0.833 | 0.906 | **0.441–0.469*** |

\* AR fluctuates slightly (0.441–0.469 across eval runs / dataset ordering); see `TEA_report.md` §18 for the full 12-model sweep (up to AR=0.4694).

There's a real trade-off here: raw TEA (single-signal, encoder-GRU) gets the best chord-prediction accuracy (H@1), while the AlignLoss-tuned MS-TEA variants trade a bit of H@1 for better emotion agreement (AR). Both are kept as checkpoints for that reason — see [Checkpoints](#checkpoints).

Reproduce with:
```bash
conda run -n video2music python _eval_amts.py                 # AMT_no_emotion / AMT_full / MSTEA_encoder_attention_align
conda run -n video2music python eval_TEA.py --exp <n>          # any experiment in EXP_CONFIGS
```

## Setup

```bash
conda create -n video2music python=3.8
conda activate video2music
pip install -r requirements.txt
```

Requires `ffmpeg` and `fluidsynth` on PATH, and a GM soundfont at `soundfonts/default_sound_font.sf2`.

## Checkpoints

Model weights are **not** included in this repo (multi-GB each). The following are kept locally and referenced by the scripts below — request access or retrain with `train_TEA.py` / `train_full.py` / `train_no_emotion.py`:

| Checkpoint | Role |
|---|---|
| `saved_models/AMT` | regression weights (note density/velocity) used by every generation script |
| `saved_models/AMT_full` | baseline chord model (current architecture, retrained) |
| `saved_models/AMT_no_emotion` | ablation: no emotion signal at all |
| `saved_models/TEA_encoder_gru` | best H@1 (single-signal TEA, encoder-only) |
| `saved_models/MSTEA_encoder_gru_align_l015` | best H@1/AR balance (multi-signal + AlignLoss λ=0.15) |
| `saved_models/MSTEA_encoder_attention_align` | best AR (attention-based multi-signal + AlignLoss) |

## Usage

Generate chords + render video for a VEVO test clip:
```bash
conda run -n video2music python generate_TEA.py --exp 16 --test_id 223
# --exp 9  = TEA_encoder_gru (best H@1)
# --exp 14 = MSTEA_encoder_gru_align_l015
# --exp 16 = MSTEA_encoder_attention_align (best AR)
```

Override the emotion conditioning directly:
```bash
conda run -n video2music python generate_TEA.py --exp 16 --test_id 223 --emotion exciting
```

Benchmark inference cost (baseline vs. TEA):
```bash
conda run -n video2music python benchmark_inference.py --test_id 223
```

## Dataset (MuVi-Sync)

`dataset/` follows the [MuVi-Sync](https://github.com/AMAAI-Lab/Video2Music) layout:

- **Music features**: `vevo_chord`, `vevo_note_density`, `vevo_loudness`
- **Video features**: `vevo_scene_offset`, `vevo_emotion` (5-class or 6-class), `vevo_semantic`, `vevo_motion`
- **Metadata**: `vevo_meta/idlist.txt` (feature/title/YouTube ID list), `vevo_meta/split/` (train/val/test splits)

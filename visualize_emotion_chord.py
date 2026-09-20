"""
visualize_emotion_chord.py
──────────────────────────
감정 궤적 + 생성 코드 품질 시각화.

Usage:
  # 정적 PNG (발표 슬라이드용)
  python visualize_emotion_chord.py --video_id 049 --exp 16 --mode static

  # 영상 오버레이 (데모용)
  python visualize_emotion_chord.py --video_id 049 --exp 16 --mode overlay

  # 둘 다
  python visualize_emotion_chord.py --video_id 049 --exp 16 --mode both

  # 감정 오버라이드 버전 비교 (emotion_override 태그 붙은 lab 파일)
  python visualize_emotion_chord.py --video_id 049 --exp 16 --emo_tag _emo_sa --mode overlay
"""

import argparse, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib import font_manager

# 한국어 폰트 설정 (NotoSansCJK)
_KO_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_KO_FONT):
    font_manager.fontManager.addfont(_KO_FONT)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_KO_FONT).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

import moviepy.editor as mpe
from moviepy.video.VideoClip import VideoClip

# ── 상수 ──────────────────────────────────────────────────────────────────────

EMO_NAMES  = ["exciting", "fearful", "tense", "sad", "relaxing", "neutral"]
EMO_COLORS = {
    "exciting": "#FF8C00",   # 주황
    "fearful":  "#8B008B",   # 보라
    "tense":    "#DC143C",   # 빨강
    "sad":      "#1E90FF",   # 파랑
    "relaxing": "#228B22",   # 초록
    "neutral":  "#888888",   # 회색
}

QUALITY_COLORS = {
    "major":  "#F4A460",   # 따뜻한 갈색-주황
    "minor":  "#4169E1",   # 차가운 파랑
    "dim":    "#800080",   # 어두운 보라
    "aug":    "#20B2AA",   # 청록
    "sus":    "#3CB371",   # 중간 녹색
    "dom7":   "#CD853F",   # 황갈색
    "maj7":   "#DAA520",   # 금색
    "min7":   "#6495ED",   # 코발트블루
    "hdim":   "#9400D3",   # 진보라
    "none":   "#D3D3D3",   # 회색 (N chord)
}

QUALITY_KO = {
    "major": "장조",  "minor": "단조",  "dim":  "감화음",
    "aug":   "증화음", "sus":  "서스",   "dom7": "속7",
    "maj7":  "장7",   "min7": "단7",    "hdim": "반감7",
    "none":  "없음",
}

# Emotion Override preset vectors (ex/fe/te/sa/re/ne → index in EMO_NAMES)
# 순서: exciting, fearful, tense, sad, relaxing, neutral
EMO_TAG_MAP = {"ex": "exciting", "fe": "fearful", "te": "tense",
               "sa": "sad",      "re": "relaxing", "ne": "neutral"}
EMO_PRESETS = {
    "exciting": np.array([0.90, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32),
    "fearful":  np.array([0.02, 0.90, 0.02, 0.02, 0.02, 0.02], dtype=np.float32),
    "tense":    np.array([0.02, 0.02, 0.90, 0.02, 0.02, 0.02], dtype=np.float32),
    "sad":      np.array([0.02, 0.02, 0.02, 0.90, 0.02, 0.02], dtype=np.float32),
    "relaxing": np.array([0.02, 0.02, 0.02, 0.02, 0.90, 0.02], dtype=np.float32),
    "neutral":  np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.90], dtype=np.float32),
}


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def get_chord_quality(chord_str: str) -> str:
    if chord_str in ("N", ""):
        return "none"
    if ":" not in chord_str:
        return "major"          # e.g. C, G, D
    q = chord_str.split(":")[1]
    if q.startswith("hdim"):   return "hdim"
    if q.startswith("min7"):   return "min7"
    if q.startswith("maj7"):   return "maj7"
    if q.startswith("min"):    return "minor"
    if q.startswith("maj"):    return "major"
    if "dim" in q:             return "dim"
    if "aug" in q:             return "aug"
    if "sus" in q:             return "sus"
    if q.endswith("7"):        return "dom7"
    return "major"


def load_emotion(video_id: str, dataset_root="./dataset") -> np.ndarray:
    """Returns ndarray [T, 6]"""
    path = Path(dataset_root) / "vevo_emotion" / "6c_l14p" / "all" / f"{video_id}.lab"
    data = np.loadtxt(str(path), skiprows=1)   # [T, 7]: col0=time, col1-6=probs
    return data[:, 1:].astype(np.float32)       # [T, 6]


def load_chords(lab_path: str):
    """Returns list of chord strings (skips 'key ?' line)"""
    chords = []
    with open(lab_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("key"):
                continue
            parts = line.split()
            chords.append(parts[1] if len(parts) >= 2 else "N")
    return chords


def find_best_segment(emotion_np, chords, window=20):
    """
    데모용으로 가장 좋은 window초 구간을 찾는다.

    스코어 기준:
      - 감정 자신감: 지배 감정 확률 평균 (높을수록 좋음)
      - 감정 변동성: 표준편차 평균 (너무 낮으면 지루함 → 적당히 있어야)
      - 코드 다양성: 구간 내 unique chord quality 수 (다양할수록 좋음)
    최종 score = confidence * (1 + variability * 3) * (1 + chord_diversity * 0.3)
    """
    n = min(len(emotion_np), len(chords))
    best_score, best_start = -1, 0

    for s in range(0, n - window + 1):
        seg_emo = emotion_np[s:s + window]
        seg_chd = chords[s:s + window]

        dom_prob    = seg_emo.max(axis=1).mean()       # 지배 감정 평균 확률
        variability = seg_emo.std(axis=0).mean()        # 감정 변동성
        qualities   = [get_chord_quality(c) for c in seg_chd]
        chord_div   = len(set(q for q in qualities if q != "none"))  # unique quality 수

        score = dom_prob * (1 + variability * 3) * (1 + chord_div * 0.3)

        if score > best_score:
            best_score = score
            best_start = s

    best_end = best_start + window
    dom_name = EMO_NAMES[int(np.argmax(emotion_np[best_start:best_end].mean(axis=0)))]
    print(f"[auto] 최적 구간: t={best_start}~{best_end} ({best_start}s~{best_end}s)  "
          f"지배감정={dom_name}  score={best_score:.3f}")
    return best_start, best_end


# ── 정적 PNG 생성 ─────────────────────────────────────────────────────────────

def make_static_plot(emotion_np, chords, video_id, exp, out_path, override_name=None,
                     emotion_np_natural=None):
    n = min(len(emotion_np), len(chords))
    # 상단 감정 패널: 항상 자연 감정 궤적 표시 (override여도 영상 원본 감정)
    emo_display = (emotion_np_natural[:n] if emotion_np_natural is not None
                   else emotion_np[:n])
    chords = chords[:n]
    T = np.arange(n)

    fig = plt.figure(figsize=(16, 5), facecolor="#1a1a2e")
    gs  = GridSpec(2, 1, figure=fig, hspace=0.08, height_ratios=[3, 1])
    ax_emo   = fig.add_subplot(gs[0])
    ax_chord = fig.add_subplot(gs[1], sharex=ax_emo)

    # ── 감정 라인 플롯 ──────────────────────────────────────────────────────
    ax_emo.set_facecolor("#12122a")
    for i, (name, col) in enumerate(EMO_COLORS.items()):
        ax_emo.fill_between(T, emo_display[:, i], alpha=0.25, color=col)
        ax_emo.plot(T, emo_display[:, i], color=col, linewidth=1.0, label=name)

    ax_emo.set_ylabel("Emotion Probability", color="white", fontsize=10)
    ax_emo.tick_params(colors="white")
    ax_emo.set_ylim(0, 1)
    for spine in ax_emo.spines.values():
        spine.set_edgecolor("#444")
    ax_emo.legend(loc="upper right", fontsize=8,
                  facecolor="#1a1a2e", labelcolor="white", framealpha=0.6,
                  ncol=3)
    plt.setp(ax_emo.get_xticklabels(), visible=False)

    # ── 코드 품질 타임라인 ──────────────────────────────────────────────────
    ax_chord.set_facecolor("#12122a")
    qualities = [get_chord_quality(c) for c in chords]
    for i, q in enumerate(qualities):
        ax_chord.bar(i, 1.0, width=1.0, color=QUALITY_COLORS[q], align="edge")

    ax_chord.set_xlim(0, n)
    ax_chord.set_ylim(0, 1)
    ax_chord.set_ylabel("Chord\nQuality", color="white", fontsize=9)
    ax_chord.set_xlabel("Timestep", color="white", fontsize=10)
    ax_chord.tick_params(colors="white")
    ax_chord.set_yticks([])
    for spine in ax_chord.spines.values():
        spine.set_edgecolor("#444")

    # ── 범례 (chord quality) ────────────────────────────────────────────────
    unique_q = sorted(set(qualities), key=list(QUALITY_COLORS.keys()).index)
    patches   = [mpatches.Patch(color=QUALITY_COLORS[q],
                                label=f"{q} ({QUALITY_KO[q]})")
                 for q in unique_q if q in QUALITY_COLORS]
    ax_chord.legend(handles=patches, loc="upper right", fontsize=7,
                    facecolor="#1a1a2e", labelcolor="white", framealpha=0.6,
                    ncol=len(patches))

    # ── 제목 ───────────────────────────────────────────────────────────────
    if override_name:
        ov_col = EMO_COLORS.get(override_name, "white")
        title  = (f"Video {video_id}  |  Exp{exp}  — Emotion Trajectory & Chord Quality"
                  f"\n[Emotion Override: {override_name.upper()}]")
        fig.suptitle(title, color="white", fontsize=11, y=0.99)
        # override 감정 색으로 상단 강조선
        fig.add_artist(plt.Line2D([0.01, 0.99], [0.985, 0.985],
                                  transform=fig.transFigure,
                                  color=ov_col, linewidth=3))
    else:
        fig.suptitle(f"Video {video_id}  |  Exp{exp}  — Emotion Trajectory & Chord Quality",
                     color="white", fontsize=12, y=0.98)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[static] 저장: {out_path}")


# ── 영상 오버레이 생성 ────────────────────────────────────────────────────────

def make_overlay_video(emotion_np, chords, video_path, out_path,
                       panel_h=200, fps_out=2.0, override_name=None):
    """
    원본 영상 아래에 감정 바 + 코드 패널을 붙인 오버레이 영상을 생성한다.
    panel_h : 하단 패널 픽셀 높이
    fps_out : 오버레이 패널 FPS (원본 영상은 원본 FPS 그대로)
    """
    n = min(len(emotion_np), len(chords))
    emo    = emotion_np[:n]
    chords_trunc = chords[:n]

    # 원본 영상 클립
    video_clip = mpe.VideoFileClip(str(video_path))
    vid_w, vid_h = video_clip.size    # e.g. 640, 360
    vid_dur = video_clip.duration     # e.g. 234.0s

    # ── 패널 프레임 렌더러 ──────────────────────────────────────────────────
    def render_panel(t):
        """t(초) → numpy RGB 이미지 [panel_h, vid_w, 3]"""
        idx = min(int(t), n - 1)
        probs   = emo[idx]
        chord   = chords_trunc[idx]
        quality = get_chord_quality(chord)

        # 지배 감정
        dom_idx  = int(np.argmax(probs))
        dom_name = EMO_NAMES[dom_idx]

        dpi = 100
        fig_w = vid_w / dpi
        fig_h = panel_h / dpi

        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                 facecolor="#1a1a2e",
                                 gridspec_kw={"width_ratios": [3, 1]})
        ax_bar  = axes[0]
        ax_info = axes[1]
        ax_bar.set_facecolor("#12122a")
        bars = ax_bar.bar(EMO_NAMES, probs,
                          color=[EMO_COLORS[e] for e in EMO_NAMES],
                          width=0.6, alpha=0.9)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks(range(len(EMO_NAMES)))
        ax_bar.set_xticklabels([e[:3].upper() for e in EMO_NAMES],
                               color="white", fontsize=7)
        ax_bar.tick_params(axis="y", colors="white", labelsize=6)
        for sp in ax_bar.spines.values(): sp.set_edgecolor("#444")
        ax_bar.set_ylabel("prob", color="#aaa", fontsize=6)
        bars[dom_idx].set_edgecolor("white"); bars[dom_idx].set_linewidth(1.5)
        for bar, prob_val in zip(bars, probs):
            if prob_val > 0.03:
                ax_bar.text(bar.get_x() + bar.get_width()/2, prob_val + 0.02,
                            f"{prob_val*100:.0f}%", ha="center", va="bottom",
                            color="white", fontsize=6, fontweight="bold")

        # override 배지
        if override_name:
            ov_col = EMO_COLORS.get(override_name, "white")
            ax_bar.text(0.99, 0.97, f"Override: {override_name.upper()}",
                        ha="right", va="top", color=ov_col,
                        fontsize=6, fontweight="bold",
                        transform=ax_bar.transAxes,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="#1a1a2e", edgecolor=ov_col,
                                  linewidth=1.0, alpha=0.85))

        # 오른쪽: 코드 + 품질
        ax_info.set_facecolor(QUALITY_COLORS.get(quality, "#555"))
        ax_info.set_xticks([]); ax_info.set_yticks([])
        for spine in ax_info.spines.values():
            spine.set_edgecolor("#222")

        # 코드 이름
        ax_info.text(0.5, 0.65, chord, ha="center", va="center",
                     color="white", fontsize=14, fontweight="bold",
                     transform=ax_info.transAxes)
        # 품질 라벨
        ax_info.text(0.5, 0.30, f"{quality}\n{QUALITY_KO.get(quality,'')}",
                     ha="center", va="center",
                     color="white", fontsize=7, alpha=0.85,
                     transform=ax_info.transAxes)
        # 시간 인덱스
        ax_info.text(0.5, 0.05, f"t={idx}", ha="center", va="bottom",
                     color="#aaa", fontsize=6,
                     transform=ax_info.transAxes)

        plt.tight_layout(pad=0.3)

        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        img = np.frombuffer(buf, dtype=np.uint8).copy()
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # vid_w 에 맞게 리사이즈
        if img.shape[1] != vid_w or img.shape[0] != panel_h:
            from PIL import Image as PILImage
            pil = PILImage.fromarray(img)
            pil = pil.resize((vid_w, panel_h), PILImage.LANCZOS)
            img = np.array(pil)
        return img

    # 패널 VideoClip
    panel_dur = min(vid_dur, n)
    panel_clip = VideoClip(render_panel, duration=panel_dur)
    panel_clip = panel_clip.set_fps(fps_out)

    # 원본 영상도 panel_dur 에 맞게 자르기
    video_clip = video_clip.subclip(0, panel_dur)

    # 세로로 합치기
    final = mpe.clips_array([[video_clip], [panel_clip]])
    final.write_videofile(
        str(out_path),
        fps=video_clip.fps,
        codec="libx264",
        audio_codec="aac",
        audio=True,
        logger=None,
        ffmpeg_params=["-crf", "23"],
    )
    print(f"[overlay] 저장: {out_path}")


# ── 비교 GIF 생성 ─────────────────────────────────────────────────────────────

def make_comparison_gif(configs, out_path, seg_start=0, seg_end=20,
                        gif_w=1200, fps=1.0):
    """
    configs: list of dict {emotion_np, chords, label, override_name}
    각 timestep마다 N개 버전을 가로로 나란히 렌더링 → animated GIF
    """
    from PIL import Image as PILImage

    n_ver = len(configs)
    panel_w = gif_w // n_ver
    panel_h = int(panel_w * 0.28)   # 비율 유지
    dpi = 100

    frames = []
    timesteps = list(range(seg_start, seg_end))

    for t_idx in timesteps:
        row_imgs = []
        for cfg in configs:
            emo    = cfg["emotion_np"]
            chords = cfg["chords"]
            label  = cfg["label"]
            ov     = cfg.get("override_name", None)

            idx    = min(t_idx - seg_start, len(emo) - 1)
            probs  = emo[idx]
            chord  = chords[idx] if idx < len(chords) else "N"
            quality = get_chord_quality(chord)
            dom_idx = int(np.argmax(probs))
            ov_col  = EMO_COLORS.get(ov, "white") if ov else None

            fig_w_in = panel_w / dpi
            fig_h_in = panel_h / dpi
            fig, axes = plt.subplots(1, 2, figsize=(fig_w_in, fig_h_in),
                                     facecolor="#1a1a2e",
                                     gridspec_kw={"width_ratios": [3, 1]})
            ax_bar, ax_info = axes

            # 감정 바
            ax_bar.set_facecolor("#12122a")
            bars = ax_bar.bar(range(len(EMO_NAMES)), probs,
                              color=[EMO_COLORS[e] for e in EMO_NAMES],
                              width=0.6, alpha=0.9)
            bars[dom_idx].set_edgecolor("white"); bars[dom_idx].set_linewidth(2)
            ax_bar.set_ylim(0, 1)
            ax_bar.set_xticks(range(len(EMO_NAMES)))
            ax_bar.set_xticklabels([e[:3].upper() for e in EMO_NAMES],
                                   color="white", fontsize=8)
            ax_bar.tick_params(axis="y", colors="white", labelsize=7)
            for sp in ax_bar.spines.values(): sp.set_edgecolor("#444")
            for bar, v in zip(bars, probs):
                if v > 0.04:
                    ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                                f"{v*100:.0f}%", ha="center", va="bottom",
                                color="white", fontsize=7, fontweight="bold")

            # 제목 라벨
            title_col = ov_col if ov_col else "white"
            ax_bar.set_title(label, color=title_col, fontsize=9,
                             fontweight="bold", pad=3)

            # 코드 패널
            ax_info.set_facecolor(QUALITY_COLORS.get(quality, "#555"))
            ax_info.set_xticks([]); ax_info.set_yticks([])
            for sp in ax_info.spines.values(): sp.set_edgecolor("#222")
            ax_info.text(0.5, 0.62, chord, ha="center", va="center",
                         color="white", fontsize=13, fontweight="bold",
                         transform=ax_info.transAxes)
            ax_info.text(0.5, 0.28, f"{quality}", ha="center", va="center",
                         color="white", fontsize=8, alpha=0.9,
                         transform=ax_info.transAxes)
            ax_info.text(0.5, 0.06, f"t={t_idx}s", ha="center", va="bottom",
                         color="#ccc", fontsize=7, transform=ax_info.transAxes)

            plt.tight_layout(pad=0.4)
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            img = np.frombuffer(buf, dtype=np.uint8).copy()
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            pil = PILImage.fromarray(img).resize((panel_w, panel_h), PILImage.LANCZOS)
            row_imgs.append(np.array(pil))

        # 가로로 이어 붙이기
        row = np.concatenate(row_imgs, axis=1)
        frames.append(PILImage.fromarray(row))

    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=False
    )
    print(f"[gif] 저장: {out_path}  ({len(frames)}프레임, {fps}fps)")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id",  type=str, default="049")
    parser.add_argument("--exp",       type=int, default=16)
    parser.add_argument("--emo_tag",   type=str, default="",
                        help="감정 오버라이드 파일 태그 (예: _emo_sa)")
    parser.add_argument("--mode",      type=str, default="both",
                        choices=["static", "overlay", "both", "gif"])
    parser.add_argument("--dataset_root", type=str, default="./dataset")
    parser.add_argument("--panel_h",   type=int, default=200,
                        help="오버레이 패널 높이 (px)")
    parser.add_argument("--fps_panel", type=float, default=2.0,
                        help="패널 렌더링 FPS (낮을수록 빠름)")
    parser.add_argument("--segment",   type=int, default=None,
                        help="N초 단위로 클립 분할 (예: --segment 20)")
    parser.add_argument("--start",     type=int, default=None,
                        help="시작 timestep (--segment 없이 단일 구간 지정 시 사용)")
    parser.add_argument("--end",       type=int, default=None,
                        help="끝 timestep (--segment 없이 단일 구간 지정 시 사용)")
    parser.add_argument("--auto",      type=int, default=None, metavar="WINDOW",
                        help="데모용 최적 구간 자동 탐색 후 해당 클립만 생성 (예: --auto 20)")
    args = parser.parse_args()

    lab_suffix = f"_TEA_exp{args.exp}{args.emo_tag}_cgen_rd.lab"
    lab_path   = Path("output") / args.video_id / f"{args.video_id}{lab_suffix}"
    # TEA 생성 mp4(음악 교체됨)를 우선 사용, 없으면 원본 영상 fallback
    tea_mp4    = Path("output") / args.video_id / f"{args.video_id}_TEA_exp{args.exp}{args.emo_tag}_cgen_rd.mp4"
    orig_mp4   = Path("dataset") / "vevo" / f"{args.video_id}.mp4"
    video_path = tea_mp4 if tea_mp4.exists() else orig_mp4
    out_dir    = Path("output") / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    assert lab_path.exists(), f"Lab 파일 없음: {lab_path}"
    assert video_path.exists(), f"영상 없음: {video_path}"
    print(f"[video] {video_path}")

    print(f"[load] 감정 로드: {args.video_id}")
    emotion_np_natural = load_emotion(args.video_id, args.dataset_root)  # 원본 영상 감정
    print(f"  emotion shape: {emotion_np_natural.shape}")

    # emo_tag 파싱: 모델에 실제 주입된 감정 (override면 preset, 아니면 natural)
    emo_short = args.emo_tag.replace("_emo_", "") if args.emo_tag else ""
    emo_override_name = EMO_TAG_MAP.get(emo_short, None)
    if emo_override_name is not None:
        preset = EMO_PRESETS[emo_override_name]
        emotion_np_full = np.tile(preset, (len(emotion_np_natural), 1))
        print(f"  [override] 모델 입력 감정 → {emo_override_name} preset (90% 고정)")
        print(f"  [display]  시각화 감정 → 원본 영상 감정 궤적 사용")
    else:
        emotion_np_full = emotion_np_natural

    print(f"[load] 코드 로드: {lab_path}")
    chords_full = load_chords(str(lab_path))
    n_full = min(len(emotion_np_full), len(chords_full))
    print(f"  chord count: {len(chords_full)}  (사용: {n_full})")

    tag = f"_exp{args.exp}{args.emo_tag}"

    # ── 구간 목록 결정 ────────────────────────────────────────────────────────
    if args.auto is not None:
        s, e = find_best_segment(emotion_np_full, chords_full, window=args.auto)
        segments = [(s, e)]
    elif args.segment is not None:
        seg = args.segment
        segments = [(s, min(s + seg, n_full))
                    for s in range(0, n_full, seg)]
    elif args.start is not None or args.end is not None:
        s = args.start or 0
        e = args.end or n_full
        segments = [(s, min(e, n_full))]
    else:
        segments = [(0, n_full)]  # 전체

    for seg_start, seg_end in segments:
        emotion_np = emotion_np_full[seg_start:seg_end]
        chords     = chords_full[seg_start:seg_end]

        if args.segment is not None:
            seg_tag = f"{tag}_t{seg_start:03d}-{seg_end:03d}"
            seg_label = f" (t={seg_start}~{seg_end}, {seg_start}s~{seg_end}s)"
        elif len(segments) == 1 and segments[0] != (0, n_full):
            seg_tag = f"{tag}_t{seg_start:03d}-{seg_end:03d}"
            seg_label = f" (t={seg_start}~{seg_end})"
        else:
            seg_tag = tag
            seg_label = ""

        print(f"\n[구간{seg_label}]")

        if args.mode in ("static", "both"):
            out_png = out_dir / f"{args.video_id}{seg_tag}_emotion_chord.png"
            emo_nat_seg = emotion_np_natural[seg_start:seg_end]
            make_static_plot(emotion_np, chords, args.video_id, args.exp, str(out_png),
                             override_name=emo_override_name,
                             emotion_np_natural=emo_nat_seg if emo_override_name else None)

        if args.mode in ("overlay", "both"):
            out_vid = out_dir / f"{args.video_id}{seg_tag}_overlay.mp4"
            _tmp_video = Path("/tmp") / f"_seg_{seg_start}_{seg_end}.mp4"
            import subprocess as _sp
            _sp.run(["ffmpeg", "-y", "-i", str(video_path),
                     "-ss", str(seg_start), "-to", str(seg_end),
                     "-c", "copy", str(_tmp_video)],
                    check=True, capture_output=True)
            make_overlay_video(emotion_np, chords, _tmp_video, str(out_vid),
                               panel_h=args.panel_h,
                               fps_out=args.fps_panel,
                               override_name=emo_override_name)
            _tmp_video.unlink(missing_ok=True)

        if args.mode == "gif":
            # 3버전 비교 GIF: natural / sad / relaxing 나란히
            orig_emo = load_emotion(args.video_id, args.dataset_root)

            def _load_version(emo_tag_short):
                tag_str  = f"_emo_{emo_tag_short}" if emo_tag_short else ""
                lab_p    = Path("output") / args.video_id / \
                           f"{args.video_id}_TEA_exp{args.exp}{tag_str}_cgen_rd.lab"
                if not lab_p.exists():
                    return None, None
                chords_v = load_chords(str(lab_p))
                ov_name  = EMO_TAG_MAP.get(emo_tag_short, None)
                if ov_name:
                    emo_v = np.tile(EMO_PRESETS[ov_name], (len(orig_emo), 1))
                else:
                    emo_v = orig_emo
                return emo_v[seg_start:seg_end], chords_v[seg_start:seg_end]

            configs = []
            for short, lbl in [("", "Natural"), ("sa", "Sad Override"), ("re", "Relaxing Override")]:
                emo_v, chords_v = _load_version(short)
                if emo_v is not None:
                    ov = EMO_TAG_MAP.get(short, None)
                    configs.append({"emotion_np": emo_v, "chords": chords_v,
                                    "label": lbl, "override_name": ov})

            if configs:
                out_gif = out_dir / f"{args.video_id}{seg_tag}_compare.gif"
                make_comparison_gif(configs, str(out_gif),
                                    seg_start=seg_start, seg_end=seg_end,
                                    fps=1.0)


if __name__ == "__main__":
    main()

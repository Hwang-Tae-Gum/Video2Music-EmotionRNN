#!/usr/bin/env python3
"""
Video 223 조건별 코드 분포 비교 차트 생성
output: presentation/chart223_chord_distribution.png
"""
import re
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LAB_DIR = Path("output/223")
OUT_PATH = Path("presentation/chart223_chord_distribution.png")

CONDITIONS = [
    ("Natural",   "223_TEA_exp16_cgen_rd.lab"),
    ("Exciting",  "223_TEA_exp16_emo_ex_cgen_rd.lab"),
    ("Sad",       "223_TEA_exp16_emo_sa_cgen_rd.lab"),
    ("Relaxing",  "223_TEA_exp16_emo_re_cgen_rd.lab"),
]

# 코드 quality → 감정 카테고리 매핑
def classify(quality: str) -> str:
    if quality in ("", ":maj7", ":maj6"):
        return "Major"
    if quality in (":min", ":min7", ":min6"):
        return "Minor"
    if quality == ":7":
        return "Dom7"
    if quality in (":dim", ":dim7", ":hdim7", ":aug"):
        return "Dim/Aug"
    if quality in (":sus2", ":sus4"):
        return "Suspended"
    return "Other"

def parse_lab(path: Path):
    chords = []
    for line in path.read_text().splitlines():
        if line.startswith("key"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        chord = parts[1]
        m = re.match(r"[A-G][#b]?(.*)", chord)
        quality = m.group(1) if m else ""
        chords.append(classify(quality))
    return chords

CATS = ["Major", "Dom7", "Suspended", "Minor", "Dim/Aug"]
COLORS = {
    "Major":     "#4CAF50",   # 초록 (밝음)
    "Dom7":      "#8BC34A",   # 연초록 (약간 밝음)
    "Suspended": "#9E9E9E",   # 회색 (중립)
    "Minor":     "#5C9BD6",   # 파랑 (어두움)
    "Dim/Aug":   "#E53935",   # 빨강 (긴장)
}
VALENCE = {
    "Major":     "Positive (bright)",
    "Dom7":      "Positive (tension)",
    "Suspended": "Neutral",
    "Minor":     "Negative (dark)",
    "Dim/Aug":   "Negative (tense)",
}

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

n_cond = len(CONDITIONS)
y_pos = np.arange(n_cond)
bar_h = 0.55

pcts_per_cond = []
for label, fname in CONDITIONS:
    chords = parse_lab(LAB_DIR / fname)
    cnt = Counter(chords)
    total = len(chords)
    pcts = {c: cnt.get(c, 0) / total * 100 for c in CATS}
    pcts_per_cond.append(pcts)

# 가로 stacked bar
lefts = np.zeros(n_cond)
for cat in CATS:
    vals = np.array([p[cat] for p in pcts_per_cond])
    bars = ax.barh(y_pos, vals, left=lefts, height=bar_h,
                   color=COLORS[cat], label=cat, edgecolor="none")
    # 5% 이상인 경우만 라벨 표시
    for i, (v, l) in enumerate(zip(vals, lefts)):
        if v >= 5:
            ax.text(l + v / 2, i, f"{v:.0f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")
    lefts += vals

# 축 스타일
ax.set_yticks(y_pos)
ax.set_yticklabels([c[0] for c in CONDITIONS], fontsize=12, color="white")
ax.set_xlabel("Chord Type Distribution (%)", color="white", fontsize=10)
ax.set_xlim(0, 100)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

# 범례
patches = [mpatches.Patch(color=COLORS[c], label=f"{c}  ({VALENCE[c]})")
           for c in CATS]
ax.legend(handles=patches, loc="lower right", fontsize=8,
          facecolor="#2a2a3e", edgecolor="#555", labelcolor="white",
          framealpha=0.85)

ax.set_title("Video 223 — Chord Distribution by Emotion Condition (Exp16)",
             color="white", fontsize=12, pad=10)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor())
print(f"saved → {OUT_PATH}")

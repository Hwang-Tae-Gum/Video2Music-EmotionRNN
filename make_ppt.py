"""
TEA 중간발표 PPT 자동 생성 (3분, 5슬라이드)
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── 색상 팔레트 ──────────────────────────────────────────────────────────────
C_BG       = RGBColor(0x0F, 0x17, 0x2A)   # 진한 남색 배경
C_ACCENT   = RGBColor(0x4C, 0x9B, 0xFF)   # 파란 강조
C_ACCENT2  = RGBColor(0xFF, 0xC0, 0x47)   # 노란 강조
C_WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
C_GRAY     = RGBColor(0xB0, 0xB8, 0xCC)
C_GREEN    = RGBColor(0x4C, 0xE0, 0x8A)
C_RED      = RGBColor(0xFF, 0x5C, 0x5C)
C_PANEL    = RGBColor(0x1A, 0x26, 0x40)   # 패널 배경

W = Inches(13.33)   # 와이드 슬라이드
H = Inches(7.5)


def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs


def blank_slide(prs):
    layout = prs.slide_layouts[6]   # 완전 빈 레이아웃
    return prs.slides.add_slide(layout)


def bg(slide, color=C_BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, l, t, w, h, fill_color=None, line_color=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(1, l, t, w, h)   # MSO_SHAPE_TYPE.RECTANGLE=1
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width
    else:
        shape.line.fill.background()
    return shape


def add_text(slide, text, l, t, w, h,
             size=Pt(18), bold=False, color=C_WHITE,
             align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(l, t, w, h)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = size
    run.font.bold  = bold
    run.font.color.rgb = color
    return txb


def accent_bar(slide, y=Inches(0.18)):
    add_rect(slide, Inches(0), y, W, Inches(0.055), fill_color=C_ACCENT)


def slide_number(slide, n):
    add_text(slide, f"{n} / 5",
             W - Inches(1.1), H - Inches(0.45),
             Inches(1.0), Inches(0.35),
             size=Pt(12), color=C_GRAY, align=PP_ALIGN.RIGHT)


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 1 — 표지 / 문제 정의
# ════════════════════════════════════════════════════════════════════════════
def slide1(prs):
    sl = blank_slide(prs)
    bg(sl)
    accent_bar(sl)
    slide_number(sl, 1)

    # 태그
    add_rect(sl, Inches(0.55), Inches(0.65), Inches(2.6), Inches(0.42),
             fill_color=C_ACCENT)
    add_text(sl, "개인 프로젝트 중간발표",
             Inches(0.55), Inches(0.65), Inches(2.6), Inches(0.42),
             size=Pt(13), bold=True, color=C_BG, align=PP_ALIGN.CENTER)

    # 제목
    add_text(sl,
             "TEA: Temporal Emotion Adapter",
             Inches(0.55), Inches(1.25), Inches(9.5), Inches(0.9),
             size=Pt(38), bold=True, color=C_WHITE)
    add_text(sl,
             "시간적 감정 흐름을 반영한 Video-to-Music 생성 개선",
             Inches(0.55), Inches(2.05), Inches(9.5), Inches(0.55),
             size=Pt(22), color=C_ACCENT)

    # 구분선
    add_rect(sl, Inches(0.55), Inches(2.75), Inches(5.5), Inches(0.04),
             fill_color=C_GRAY)

    # 문제 박스
    add_rect(sl, Inches(0.45), Inches(3.0), Inches(5.9), Inches(3.65),
             fill_color=C_PANEL, line_color=C_ACCENT, line_width=Pt(1.2))

    add_text(sl, "📌  문제 정의",
             Inches(0.65), Inches(3.1), Inches(5.5), Inches(0.45),
             size=Pt(16), bold=True, color=C_ACCENT)

    items = [
        ("기존 AMT",     "감정 피처(6차원)를 다른 피처와 단순 concat"),
        ("핵심 발견",    "비디오 전체를 다른 영상으로 교체해도\nAffective Corr 변화 = 0.000"),
        ("결론",         "Baseline은 비디오를 실질적으로 무시\n→ '감정을 잡는 척'만 한다"),
    ]
    y = Inches(3.65)
    for label, body in items:
        add_rect(sl, Inches(0.65), y, Inches(1.2), Inches(0.28),
                 fill_color=C_ACCENT2)
        add_text(sl, label,
                 Inches(0.65), y, Inches(1.2), Inches(0.28),
                 size=Pt(11), bold=True, color=C_BG, align=PP_ALIGN.CENTER)
        add_text(sl, body,
                 Inches(2.0), y, Inches(4.1), Inches(0.55),
                 size=Pt(13), color=C_WHITE)
        y += Inches(0.72)

    # 오른쪽: ΔCorr 비교 강조 패널
    add_rect(sl, Inches(6.9), Inches(3.0), Inches(5.95), Inches(3.65),
             fill_color=C_PANEL, line_color=C_ACCENT2, line_width=Pt(1.5))
    add_text(sl, "비디오 피처 셔플 테스트",
             Inches(7.1), Inches(3.1), Inches(5.5), Inches(0.45),
             size=Pt(15), bold=True, color=C_ACCENT2)

    rows = [
        ("BASE",  "ΔCorr(all) = 0.000", C_RED,   "→ 영상 바꿔도 무반응"),
        ("Exp3",  "ΔCorr(all) = −0.044", C_GREEN, "→ 영상 바꾸면 성능 하락"),
        ("Exp4",  "ΔCorr(all) = −0.033", C_GREEN, "→ 영상 바꾸면 성능 하락"),
        ("Exp8",  "ΔCorr(all) = −0.024", C_GREEN, "→ 영상 바꾸면 성능 하락"),
    ]
    y = Inches(3.65)
    for model, corr, col, note in rows:
        add_text(sl, model, Inches(7.1), y, Inches(0.8), Inches(0.4),
                 size=Pt(13), bold=True, color=col)
        add_text(sl, corr,  Inches(8.1), y, Inches(2.1), Inches(0.4),
                 size=Pt(13), bold=True, color=col)
        add_text(sl, note,  Inches(10.3), y, Inches(2.4), Inches(0.4),
                 size=Pt(11), color=C_GRAY)
        y += Inches(0.62)

    add_text(sl, "BASE만 유일하게 비디오에 무반응\n→ TEA가 실제로 비디오-감정을 반영함을 증명",
             Inches(7.1), Inches(6.1), Inches(5.5), Inches(0.7),
             size=Pt(12), color=C_ACCENT, bold=True)

    return sl


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 2 — TEA 설계
# ════════════════════════════════════════════════════════════════════════════
def slide2(prs):
    sl = blank_slide(prs)
    bg(sl)
    accent_bar(sl)
    slide_number(sl, 2)

    add_text(sl, "TEA 모듈 설계",
             Inches(0.5), Inches(0.35), Inches(8), Inches(0.6),
             size=Pt(30), bold=True, color=C_WHITE)
    add_text(sl, "Temporal Emotion Adapter — 감정 시퀀스를 시간적으로 처리",
             Inches(0.5), Inches(0.92), Inches(10), Inches(0.4),
             size=Pt(16), color=C_ACCENT)

    # 왼쪽: 구조 다이어그램
    add_rect(sl, Inches(0.4), Inches(1.5), Inches(5.8), Inches(5.3),
             fill_color=C_PANEL, line_color=C_ACCENT, line_width=Pt(1))

    add_text(sl, "TEA 내부 구조",
             Inches(0.6), Inches(1.6), Inches(5.4), Inches(0.38),
             size=Pt(14), bold=True, color=C_ACCENT)

    steps = [
        (C_ACCENT,  "Input",       "feature_emotion  (B, T, 6)"),
        (C_GRAY,    "① 감정 가중치", "학습 가능한 6차원 가중치 + temperature scaling"),
        (C_GRAY,    "② Dropout",    "p=0.2  —  노이즈 강건성 확보"),
        (C_GRAY,    "③ 시간 윈도우", "window=2, 지수 감쇠  —  최근 프레임 강조"),
        (C_ACCENT2, "④ BiRNN",      "LSTM 또는 GRU  (hidden = d_model // 2)"),
        (C_GRAY,    "⑤ Residual",   "input_proj + rnn_out → LayerNorm"),
        (C_GREEN,   "Output",      "emotion_embedding  (B, T, 512)"),
    ]
    y = Inches(2.1)
    for col, label, desc in steps:
        add_rect(sl, Inches(0.6), y, Inches(1.5), Inches(0.32), fill_color=col)
        add_text(sl, label, Inches(0.6), y, Inches(1.5), Inches(0.32),
                 size=Pt(11), bold=True, color=C_BG, align=PP_ALIGN.CENTER)
        add_text(sl, desc,  Inches(2.25), y, Inches(3.7), Inches(0.32),
                 size=Pt(11), color=C_WHITE)
        if col != C_GREEN:
            add_text(sl, "↓", Inches(1.2), y + Inches(0.32), Inches(0.4), Inches(0.25),
                     size=Pt(12), color=C_GRAY, align=PP_ALIGN.CENTER)
        y += Inches(0.56)

    # 오른쪽: 주입 위치
    add_rect(sl, Inches(6.6), Inches(1.5), Inches(6.3), Inches(5.3),
             fill_color=C_PANEL, line_color=C_ACCENT2, line_width=Pt(1))
    add_text(sl, "TEA 주입 위치 × 실험 구성",
             Inches(6.8), Inches(1.6), Inches(5.9), Inches(0.38),
             size=Pt(14), bold=True, color=C_ACCENT2)

    configs = [
        ("Encoder only", "Exp3 (LSTM)  /  Exp4 (GRU)",
         "TEA 출력 → Video feature에 additive 주입",   C_ACCENT),
        ("Decoder only", "Exp1 (LSTM)  /  Exp2 (GRU)",
         "TEA 출력 → Decoder Cross-Attention K/V",     C_ACCENT),
        ("Both",         "Exp5~8 (4가지 조합)",
         "Encoder TEA + Decoder TEA\n→ enc_dec_fusion으로 융합",  C_ACCENT2),
    ]
    y = Inches(2.1)
    for title, exps, desc, col in configs:
        add_rect(sl, Inches(6.8), y, Inches(5.9), Inches(1.35),
                 fill_color=RGBColor(0x12, 0x1E, 0x35),
                 line_color=col, line_width=Pt(0.8))
        add_text(sl, title, Inches(7.0), y + Inches(0.08), Inches(2.0), Inches(0.35),
                 size=Pt(14), bold=True, color=col)
        add_text(sl, exps,  Inches(9.1), y + Inches(0.08), Inches(3.4), Inches(0.35),
                 size=Pt(12), color=C_ACCENT2)
        add_text(sl, desc,  Inches(7.0), y + Inches(0.45), Inches(5.5), Inches(0.75),
                 size=Pt(11), color=C_GRAY)
        y += Inches(1.55)

    add_text(sl, "총 8가지 실험  ·  각 100 epoch  ·  3× RTX A6000",
             Inches(6.8), Inches(6.45), Inches(5.9), Inches(0.35),
             size=Pt(12), color=C_GRAY, align=PP_ALIGN.CENTER)

    return sl


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 3 — 현재 결과
# ════════════════════════════════════════════════════════════════════════════
def slide3(prs):
    sl = blank_slide(prs)
    bg(sl)
    accent_bar(sl)
    slide_number(sl, 3)

    add_text(sl, "현재 결과",
             Inches(0.5), Inches(0.35), Inches(8), Inches(0.6),
             size=Pt(30), bold=True, color=C_WHITE)
    add_text(sl, "테스트셋 기준 — Best Val Loss 체크포인트",
             Inches(0.5), Inches(0.92), Inches(10), Inches(0.4),
             size=Pt(16), color=C_ACCENT)

    # ── H@1 바 차트 (수동) ──────────────────────────────────────────────────
    add_rect(sl, Inches(0.4), Inches(1.45), Inches(6.2), Inches(4.7),
             fill_color=C_PANEL, line_color=C_ACCENT, line_width=Pt(1))
    add_text(sl, "H@1 성능 비교 (테스트셋)",
             Inches(0.6), Inches(1.55), Inches(5.8), Inches(0.38),
             size=Pt(14), bold=True, color=C_ACCENT)

    bars = [
        ("논문 AMT",   0.5139, C_GRAY),
        ("BASE(재현)", 0.5995, C_GRAY),
        ("Exp3",       0.6700, C_GREEN),
        ("Exp4",       0.6689, C_GREEN),
        ("Exp8",       0.6684, C_GREEN),
    ]
    bar_area_l = Inches(1.4)
    bar_area_w = Inches(4.6)
    bar_h      = Inches(0.42)
    bar_gap    = Inches(0.56)
    max_val    = 0.75
    y_start    = Inches(2.1)

    for i, (label, val, color) in enumerate(bars):
        y = y_start + i * bar_gap
        bw = bar_area_w * (val / max_val)
        add_rect(sl, bar_area_l, y, bw, bar_h, fill_color=color)
        add_text(sl, label, Inches(0.5), y, Inches(0.85), bar_h,
                 size=Pt(11), color=C_WHITE, align=PP_ALIGN.RIGHT)
        add_text(sl, f"{val:.4f}", bar_area_l + bw + Inches(0.08), y,
                 Inches(0.7), bar_h, size=Pt(12), bold=True, color=color)

    # +30% 강조 배지
    add_rect(sl, Inches(4.0), Inches(1.58), Inches(2.3), Inches(0.32),
             fill_color=C_ACCENT2)
    add_text(sl, "논문 AMT 대비 최대 +30.4%",
             Inches(4.0), Inches(1.58), Inches(2.3), Inches(0.32),
             size=Pt(11), bold=True, color=C_BG, align=PP_ALIGN.CENTER)

    # ── 오른쪽: 전체 결과 테이블 ───────────────────────────────────────────
    add_rect(sl, Inches(6.9), Inches(1.45), Inches(6.0), Inches(4.7),
             fill_color=C_PANEL, line_color=C_ACCENT, line_width=Pt(1))
    add_text(sl, "전체 실험 결과 (Test Set)",
             Inches(7.1), Inches(1.55), Inches(5.6), Inches(0.38),
             size=Pt(14), bold=True, color=C_ACCENT)

    headers = ["모델", "H@1", "H@3", "H@5", "Corr"]
    col_x   = [Inches(7.05), Inches(8.65), Inches(9.55), Inches(10.45), Inches(11.35)]
    col_w   = [Inches(1.5),  Inches(0.8),  Inches(0.8),  Inches(0.8),   Inches(0.85)]

    # 헤더
    for j, h in enumerate(headers):
        add_rect(sl, col_x[j], Inches(2.0), col_w[j], Inches(0.32), fill_color=C_ACCENT)
        add_text(sl, h, col_x[j], Inches(2.0), col_w[j], Inches(0.32),
                 size=Pt(11), bold=True, color=C_BG, align=PP_ALIGN.CENTER)

    rows = [
        ("BASE",         "0.5995", "0.8545", "0.9166", "0.4597", False),
        ("Exp1 dec LSTM","0.6093", "0.8636", "0.9258", "0.4643", False),
        ("Exp2 dec GRU", "0.6542", "0.8797", "0.9301", "0.4900", False),
        ("Exp3 enc LSTM","0.6700", "0.8845", "0.9352", "0.4885", True),
        ("Exp4 enc GRU", "0.6689", "0.8862", "0.9346", "0.4939", True),
        ("Exp5 both LL", "0.6590", "0.8816", "0.9349", "0.4881", False),
        ("Exp6 both GG", "0.6293", "0.8724", "0.9295", "0.4655", False),
        ("Exp7 both LG", "0.6154", "0.8638", "0.9170", "0.4535", False),
        ("Exp8 both GL", "0.6684", "0.8857", "0.9359", "0.4853", True),
    ]
    for i, (m, h1, h3, h5, corr, highlight) in enumerate(rows):
        y = Inches(2.38) + i * Inches(0.38)
        bg_c = RGBColor(0x1E, 0x30, 0x50) if highlight else RGBColor(0x12, 0x1C, 0x33)
        vals = [m, h1, h3, h5, corr]
        for j, v in enumerate(vals):
            add_rect(sl, col_x[j], y, col_w[j], Inches(0.34), fill_color=bg_c)
            color = C_ACCENT2 if (highlight and j > 0) else C_WHITE
            add_text(sl, v, col_x[j], y, col_w[j], Inches(0.34),
                     size=Pt(10), color=color,
                     bold=highlight and j > 0, align=PP_ALIGN.CENTER)

    # 하단 요약
    add_rect(sl, Inches(0.4), Inches(6.3), Inches(12.5), Inches(0.78),
             fill_color=RGBColor(0x12, 0x24, 0x45),
             line_color=C_ACCENT2, line_width=Pt(1))
    add_text(sl,
             "모든 TEA 실험이 Baseline 상회  ·  EmoLoss 7~10% 개선  ·  "
             "Encoder TEA가 H@k 최상위 (Exp3 H@1=0.6700)",
             Inches(0.6), Inches(6.35), Inches(12.1), Inches(0.65),
             size=Pt(14), color=C_ACCENT2, bold=True, align=PP_ALIGN.CENTER)

    return sl


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 4 — 핵심 증거 + 강건성
# ════════════════════════════════════════════════════════════════════════════
def slide4(prs):
    sl = blank_slide(prs)
    bg(sl)
    accent_bar(sl)
    slide_number(sl, 4)

    add_text(sl, "핵심 증거 — TEA는 실제로 감정을 반영한다",
             Inches(0.5), Inches(0.35), Inches(12), Inches(0.6),
             size=Pt(28), bold=True, color=C_WHITE)

    # ── 왼쪽 상단: 셔플 테스트 ─────────────────────────────────────────────
    add_rect(sl, Inches(0.4), Inches(1.1), Inches(6.2), Inches(2.85),
             fill_color=C_PANEL, line_color=C_ACCENT, line_width=Pt(1))
    add_text(sl, "비디오 피처 셔플 테스트",
             Inches(0.6), Inches(1.2), Inches(5.8), Inches(0.38),
             size=Pt(14), bold=True, color=C_ACCENT)
    add_text(sl, "비디오 피처를 다른 영상 것으로 교체 시 Affective Corr 변화량",
             Inches(0.6), Inches(1.58), Inches(5.8), Inches(0.32),
             size=Pt(11), color=C_GRAY)

    shuffle_rows = [
        ("BASE",  "0.000", "0.000", C_RED,   "어떤 영상 줘도 동일 → 비디오 무시"),
        ("Exp3",  "−0.001", "−0.044", C_GREEN, "전체 피처 셔플 시 성능 하락"),
        ("Exp4",  "0.000",  "−0.033", C_GREEN, ""),
        ("Exp8",  "0.000",  "−0.024", C_GREEN, ""),
    ]
    hx = [Inches(0.6), Inches(1.85), Inches(2.95), Inches(4.25)]
    hl = ["모델", "ΔCorr(emo)", "ΔCorr(all)", "해석"]
    hw = [Inches(1.1), Inches(0.95), Inches(1.1), Inches(2.3)]
    for j, h in enumerate(hl):
        add_rect(sl, hx[j], Inches(2.0), hw[j], Inches(0.3), fill_color=C_ACCENT)
        add_text(sl, h, hx[j], Inches(2.0), hw[j], Inches(0.3),
                 size=Pt(10), bold=True, color=C_BG, align=PP_ALIGN.CENTER)
    for i, (m, de, da, col, note) in enumerate(shuffle_rows):
        y = Inches(2.35) + i * Inches(0.38)
        vals = [m, de, da, note]
        for j, v in enumerate(vals):
            add_text(sl, v, hx[j], y, hw[j], Inches(0.35),
                     size=Pt(10), bold=(j < 3), color=col if j < 3 else C_GRAY,
                     align=PP_ALIGN.CENTER if j < 3 else PP_ALIGN.LEFT)

    # ── 왼쪽 하단: Major/Minor ─────────────────────────────────────────────
    add_rect(sl, Inches(0.4), Inches(4.1), Inches(6.2), Inches(2.85),
             fill_color=C_PANEL, line_color=C_ACCENT2, line_width=Pt(1))
    add_text(sl, "정성적 증거 — Major/Minor 비율 (밝은 영상)",
             Inches(0.6), Inches(4.2), Inches(5.8), Inches(0.38),
             size=Pt(14), bold=True, color=C_ACCENT2)

    mm_data = [
        ("Baseline AMT", 74.0, 21.3, 4.7),
        ("TEA Exp4",     96.7,  2.7, 0.6),
    ]
    bx  = Inches(1.3)
    max_w = Inches(4.0)
    ys  = [Inches(4.75), Inches(5.6)]
    colors = [(C_ACCENT, C_RED, C_GRAY), (C_GREEN, C_GRAY, C_GRAY)]

    for i, (label, maj, minor, dim) in enumerate(mm_data):
        add_text(sl, label, Inches(0.6), ys[i], Inches(1.5), Inches(0.35),
                 size=Pt(12), bold=(i == 1), color=C_ACCENT2 if i == 1 else C_GRAY)
        seg_data = [(maj, colors[i][0], f"Major {maj}%"),
                    (minor, colors[i][1], f"Minor {minor}%"),
                    (dim, colors[i][2], f"Dim {dim}%")]
        x = bx
        for val, col, lbl in seg_data:
            sw = max_w * (val / 100)
            add_rect(sl, x, ys[i], sw, Inches(0.35), fill_color=col)
            if val > 8:
                add_text(sl, lbl, x, ys[i], sw, Inches(0.35),
                         size=Pt(9), bold=True, color=C_BG, align=PP_ALIGN.CENTER)
            x += sw

    add_text(sl, "밝은 영상에서 TEA가 major 코드를 96.7% 선택 (baseline 74%)\n→ 감정-코드 정렬이 실질적으로 개선됨",
             Inches(0.6), Inches(6.1), Inches(5.8), Inches(0.6),
             size=Pt(11), color=C_WHITE)

    # ── 오른쪽: 강건성 ─────────────────────────────────────────────────────
    add_rect(sl, Inches(6.9), Inches(1.1), Inches(6.0), Inches(5.85),
             fill_color=C_PANEL, line_color=C_GRAY, line_width=Pt(1))
    add_text(sl, "강건성 평가",
             Inches(7.1), Inches(1.2), Inches(5.6), Inches(0.38),
             size=Pt(14), bold=True, color=C_WHITE)
    add_text(sl, "비디오 피처 전체에 Gaussian 노이즈 σ 추가",
             Inches(7.1), Inches(1.6), Inches(5.6), Inches(0.3),
             size=Pt(11), color=C_GRAY)

    r_headers = ["모델", "σ=0.0", "σ=0.05", "σ=0.10", "σ=0.20"]
    rx = [Inches(7.05), Inches(8.35), Inches(9.3), Inches(10.25), Inches(11.2)]
    rw = [Inches(1.2), Inches(0.85), Inches(0.85), Inches(0.85), Inches(0.85)]
    for j, h in enumerate(r_headers):
        add_rect(sl, rx[j], Inches(2.05), rw[j], Inches(0.3), fill_color=C_GRAY)
        add_text(sl, h, rx[j], Inches(2.05), rw[j], Inches(0.3),
                 size=Pt(10), bold=True, color=C_BG, align=PP_ALIGN.CENTER)

    r_rows = [
        ("BASE",  "0.5995", "0.5995", "0.5995", "0.5995"),
        ("Exp3",  "0.6700", "0.6701", "0.6694", "0.6699"),
        ("Exp4",  "0.6689", "0.6679", "0.6675", "0.6676"),
        ("Exp8",  "0.6684", "0.6678", "0.6679", "0.6667"),
    ]
    for i, (m, *vals) in enumerate(r_rows):
        y = Inches(2.4) + i * Inches(0.42)
        bg_c = RGBColor(0x12, 0x1C, 0x33)
        add_text(sl, m, rx[0], y, rw[0], Inches(0.38),
                 size=Pt(11), color=C_ACCENT2, bold=True)
        for j, v in enumerate(vals):
            add_text(sl, v, rx[j+1], y, rw[j+1], Inches(0.38),
                     size=Pt(10), color=C_WHITE, align=PP_ALIGN.CENTER)

    add_rect(sl, Inches(6.9), Inches(4.2), Inches(6.0), Inches(0.7),
             fill_color=RGBColor(0x12, 0x2E, 0x18), line_color=C_GREEN, line_width=Pt(1))
    add_text(sl, "σ=0.2 (20% 노이즈)에서도 H@1 저하 < 0.002\n→ 실사용 환경에서 안정적",
             Inches(7.1), Inches(4.25), Inches(5.6), Inches(0.6),
             size=Pt(12), bold=True, color=C_GREEN)

    # 시드 안정성
    add_text(sl, "시드 안정성 (3회 반복 평가)",
             Inches(7.1), Inches(5.1), Inches(5.6), Inches(0.35),
             size=Pt(13), bold=True, color=C_WHITE)
    add_text(sl, "std = 0.0000  —  완전 결정론적 평가\n동일 환경에서 항상 동일 수치 재현",
             Inches(7.1), Inches(5.45), Inches(5.6), Inches(0.6),
             size=Pt(12), color=C_GRAY)

    add_rect(sl, Inches(6.9), Inches(6.2), Inches(6.0), Inches(0.55),
             fill_color=RGBColor(0x18, 0x26, 0x40))
    add_text(sl, "정량(ΔCorr) + 정성(Major/Minor) + 강건성(Noise) 모두 TEA 우위 확인",
             Inches(7.1), Inches(6.25), Inches(5.6), Inches(0.45),
             size=Pt(11), color=C_ACCENT, bold=True)

    return sl


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 5 — 앞으로 계획
# ════════════════════════════════════════════════════════════════════════════
def slide5(prs):
    sl = blank_slide(prs)
    bg(sl)
    accent_bar(sl)
    slide_number(sl, 5)

    add_text(sl, "앞으로 구현 계획",
             Inches(0.5), Inches(0.35), Inches(9), Inches(0.6),
             size=Pt(30), bold=True, color=C_WHITE)
    add_text(sl, "확장성 확보 — 사용자 제어 가능한 감정 기반 음악 생성",
             Inches(0.5), Inches(0.92), Inches(10), Inches(0.4),
             size=Pt(16), color=C_ACCENT)

    plans = [
        (
            "① Emotion Override",
            "감정 벡터 수동 지정 → 원하는 감정의 음악 생성",
            [
                "generate_TEA.py에 --emotion \"0,0,0,1,0,0\" 인자 추가",
                "같은 영상 + 다른 감정 벡터 → 다른 분위기 음악 생성 데모",
                "예: 어두운 영상에 happy 감정 지정 → 밝은 음악",
            ],
            C_ACCENT,
            "구현 난이도: 낮음  ·  발표 임팩트: 높음",
        ),
        (
            "② 발표용 데모 영상 제작",
            "TEA vs Baseline 음악 비교 영상 편집",
            [
                "동일 영상(049 어두운 씬)에 Baseline / Exp3 / Exp8 음악 비교",
                "Major/Minor 비율 차이가 체감되는 밝은 영상 추가",
                "Emotion Override 전/후 비교 클립",
            ],
            C_ACCENT2,
            "구현 난이도: 중간  ·  청중 이해도 ↑",
        ),
        (
            "③ 최종 보고서 정리",
            "실험 전체 결과 + 분석 문서화",
            [
                "TEA_report.md 기반 최종 보고서 작성",
                "실패 실험(Exp9, Contrastive) 포함 — 시도 과정 기록",
                "한계 및 향후 연구 방향 명시",
            ],
            C_GRAY,
            "구현 난이도: 낮음  ·  완성도 ↑",
        ),
    ]

    for i, (title, subtitle, bullets, col, note) in enumerate(plans):
        x = Inches(0.4 + i * 4.3)
        add_rect(sl, x, Inches(1.5), Inches(4.0), Inches(5.2),
                 fill_color=C_PANEL, line_color=col, line_width=Pt(1.5))

        add_rect(sl, x, Inches(1.5), Inches(4.0), Inches(0.5), fill_color=col)
        add_text(sl, title, x, Inches(1.5), Inches(4.0), Inches(0.5),
                 size=Pt(15), bold=True, color=C_BG, align=PP_ALIGN.CENTER)

        add_text(sl, subtitle, x + Inches(0.15), Inches(2.1),
                 Inches(3.7), Inches(0.45),
                 size=Pt(12), color=col, bold=True)

        y_b = Inches(2.65)
        for b in bullets:
            add_text(sl, f"• {b}", x + Inches(0.15), y_b,
                     Inches(3.7), Inches(0.55),
                     size=Pt(11), color=C_WHITE)
            y_b += Inches(0.6)

        add_rect(sl, x, Inches(6.25), Inches(4.0), Inches(0.3),
                 fill_color=RGBColor(0x10, 0x1A, 0x30))
        add_text(sl, note, x, Inches(6.25), Inches(4.0), Inches(0.3),
                 size=Pt(10), color=col, align=PP_ALIGN.CENTER)

    # 하단 타임라인
    add_rect(sl, Inches(0.4), Inches(6.7), Inches(12.5), Inches(0.6),
             fill_color=RGBColor(0x12, 0x24, 0x45),
             line_color=C_ACCENT, line_width=Pt(1))
    add_text(sl,
             "우선순위:  ① Emotion Override 구현  →  ② 데모 영상  →  ③ 보고서",
             Inches(0.6), Inches(6.75), Inches(12.1), Inches(0.5),
             size=Pt(14), bold=True, color=C_ACCENT2, align=PP_ALIGN.CENTER)

    return sl


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
prs = new_prs()
slide1(prs)
slide2(prs)
slide3(prs)
slide4(prs)
slide5(prs)

out = "/home/taegum/Video2Music/TEA_midterm_presentation.pptx"
prs.save(out)
print(f"Saved: {out}")
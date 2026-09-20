"""
TEA Demo — Temporal Emotion Adapter
Baseline AMT vs MS-TEA Exp16 비교 데모
"""
import gradio as gr
import os
from pathlib import Path

os.environ.setdefault("GRADIO_TEMP_DIR", os.path.expanduser("~/.gradio_tmp"))

WORK_DIR = str(Path(__file__).parent.resolve())

def p(rel):
    return os.path.join(WORK_DIR, rel)


CSS = """
.col-head { font-size:1.05em; font-weight:bold; text-align:center; padding:4px 0; }
.metric-box { background:#f0f4f8; border-radius:8px; padding:12px 16px; margin:8px 0; }
"""

with gr.Blocks(title="TEA vs Baseline", css=CSS) as demo:
    gr.Markdown(
        "# TEA — Temporal Emotion Adapter\n"
        "**Baseline AMT** vs **MS-TEA Exp16**  ·  동일 비디오, 동일 렌더링 파이프라인"
    )

    with gr.Tabs():

        # ── Tab 1: Video 223 (Taylor Swift Anti-Hero) ─────────────────────────
        with gr.Tab("Video 223 — Taylor Swift Anti-Hero"):
            gr.Markdown(
                "동일 비디오에서 Baseline AMT와 TEA가 각각 생성한 배경음악을 비교합니다.\n\n"
                "> 렌더링 파이프라인(악기 구성·SF2·ffmpeg) 완전 동일 — 코드 시퀀스 생성 모델만 다름"
            )

            # ── Natural 비교 ──────────────────────────────────────────────────
            gr.Markdown("### Natural (Auto) — Baseline vs TEA")
            gr.Markdown("감정 Override 없이 영상 감정 자동 분석 결과로 생성")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Baseline (AMT full)**", elem_classes=["col-head"])
                    gr.Video(
                        value=p("output/223/223_BASE_banner.mp4"),
                        label="Baseline — Natural",
                        interactive=False,
                    )
                with gr.Column():
                    gr.Markdown("**TEA (MS-TEA Exp16)**", elem_classes=["col-head"])
                    gr.Video(
                        value=p("output/223/223_TEA_banner.mp4"),
                        label="TEA — Natural",
                        interactive=False,
                    )

            # ── Emotion Override 비교 ─────────────────────────────────────────
            gr.Markdown("### TEA Emotion Override 비교")
            gr.Markdown(
                "TEA의 감정 Override 기능 — 동일 비디오에서 감정 설정만 바꿔 음악 스타일 변경  \n"
                "*(Baseline은 감정 Override 기능 없음)*"
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**TEA — Exciting Override**", elem_classes=["col-head"])
                    gr.Video(
                        value=p("output/223/223_TEA_exciting_banner.mp4"),
                        label="TEA Exciting",
                        interactive=False,
                    )
                with gr.Column():
                    gr.Markdown("**TEA — Sad Override**", elem_classes=["col-head"])
                    gr.Video(
                        value=p("output/223/223_TEA_sad_banner.mp4"),
                        label="TEA Sad",
                        interactive=False,
                    )
                with gr.Column():
                    gr.Markdown("**TEA — Relaxing Override**", elem_classes=["col-head"])
                    gr.Video(
                        value=p("output/223/223_TEA_relaxing_banner.mp4"),
                        label="TEA Relaxing",
                        interactive=False,
                    )

        # ── Tab 2: Video 049 ───────────────────────────────────────────────────
        with gr.Tab("Video 049"):
            gr.Markdown(
                "Video 049에서 Baseline AMT와 TEA Emotion Override 3종을 비교합니다.\n\n"
                "> 렌더링 파이프라인(악기 구성·SF2·ffmpeg) 완전 동일 — 코드 시퀀스 생성 모델만 다름"
            )

            # ── Baseline vs TEA Exciting ──────────────────────────────────────
            gr.Markdown("### Baseline vs TEA — Exciting Override")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Baseline (AMT full)**", elem_classes=["col-head"])
                    gr.Video(
                        value=p("output/049/049_BASE_banner.mp4"),
                        label="Baseline — Natural",
                        interactive=False,
                    )
                with gr.Column():
                    gr.Markdown("**TEA — Exciting Override**", elem_classes=["col-head"])
                    gr.Video(
                        value=p("output/049/049_TEA_exciting_banner.mp4"),
                        label="TEA — Exciting",
                        interactive=False,
                    )

            # ── TEA Override 추가 비교 ────────────────────────────────────────
            gr.Markdown("### TEA Emotion Override 비교")
            gr.Markdown("동일 비디오에서 감정 설정만 바꾼 TEA 결과 — 스타일 차이 확인")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**TEA — Sad Override**", elem_classes=["col-head"])
                    gr.Video(
                        value=p("output/049/049_TEA_sad_banner.mp4"),
                        label="TEA — Sad",
                        interactive=False,
                    )
                with gr.Column():
                    gr.Markdown("**TEA — Relaxing Override**", elem_classes=["col-head"])
                    gr.Video(
                        value=p("output/049/049_TEA_relaxing_banner.mp4"),
                        label="TEA — Relaxing",
                        interactive=False,
                    )

        # ── Tab 3: 성능 지표 ───────────────────────────────────────────────────
        with gr.Tab("Performance Metrics"):

            gr.Markdown("## 모델별 성능 비교")
            gr.Markdown(
                "> **H@k** (Hit@k): 상위 k개 후보 중 정답 코드가 포함될 확률 ↑  \n"
                "> **Agreement Rate (AR)**: 각 타임스텝에서 영상 valence 방향 ↔ 생성 코드 valence 방향 일치율 ↑  \n"
                "> *AR 구조적 상한 ~0.5 미만: 학습셋 내 코드 68% positive valence, 영상 감정 55% negative로 인한 데이터셋 편향*"
            )

            gr.Markdown(
                "### 전 모델 비교 — 완전 지표\n\n"
                "| 모델 | H@1 ↑ | H@3 ↑ | H@5 ↑ | Agreement Rate ↑ |\n"
                "|------|-------:|-------:|-------:|----------------:|\n"
                "| AMT no emotion | 0.4990 | 0.7807 | 0.8791 | 0.4190 |\n"
                "| **Baseline (AMT full)** | **0.5108** | **0.7859** | **0.8815** | **0.4232** |\n"
                "| TEA\\_encoder\\_gru | 0.6090 | 0.8528 | 0.9180 | 0.4253 |\n"
                "| **MS-TEA\\_align\\_attn (Exp16)** | **0.5659** | **0.8320** | **0.9066** | **0.4694** |\n\n"
                "> H@5 = 0.91: 상위 5개 후보 중 91%에서 정답 코드 포함  \n"
                "> TEA\\_encoder\\_gru는 H@1·H@k 최고이나 AR 낮음 — Exp16이 코드 품질·감정 일치 **균형** 우수  \n"
                "> Agreement Rate 상한 ~0.5: 코드 68% positive valence, 영상 감정 55% negative 데이터셋 편향에 의한 구조적 한계"
            )

            gr.Markdown(
                "### 감정 기여도 Ablation — 단계별 향상\n\n"
                "| 단계 | 모델 | H@1 | Δ H@1 | H@3 | H@5 | Agreement Rate | Δ AR |\n"
                "|------|------|----:|------:|----:|----:|---------------:|-----:|\n"
                "| 1 | AMT no emotion | 0.4990 | *(base)* | 0.7807 | 0.8791 | 0.4190 | *(base)* |\n"
                "| 2 | AMT full (+emotion concat) | 0.5108 | **+2.4%** (+0.012) | 0.7859 | 0.8815 | 0.4232 | **+1.0%** (+0.004) |\n"
                "| 3 | MS-TEA Exp16 (+TEA adapter) | **0.5659** | **+13.4%** (+0.055) | **0.8320** | **0.9066** | **0.4694** | **+12.0%** (+0.050) |\n\n"
                "Δ는 step 1 (AMT no emotion) 대비 절대 변화량 · 상대 변화율  \n"
                "→ 감정 feature concat(step 2→1: +1.2pp H@1) 보다 **TEA 시간적 처리(step 3→1: +5.5pp H@1, +5.0pp AR)** 가 압도적으로 효과적"
            )

            gr.Markdown("### Chord-Emotion 분포 프로파일 — Exp16 생성 결과")
            gr.Markdown(
                "상단: 영상 감정 궤적 (6감정 확률), 하단: 생성된 코드 품질 타임라인  \n"
                "Emotion Override 설정 시 감정 궤적이 override 감정으로 고정됨 (90% preset)"
            )

            gr.Markdown("#### Video 223 — Taylor Swift Anti-Hero")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Natural (자동 분석)**", elem_classes=["col-head"])
                    gr.Image(value=p("output/223/223_exp16_emotion_chord.png"),
                             label="223 Natural", interactive=False)
                with gr.Column():
                    gr.Markdown("**Exciting Override**", elem_classes=["col-head"])
                    gr.Image(value=p("output/223/223_exp16_emo_ex_emotion_chord.png"),
                             label="223 Exciting", interactive=False)
                with gr.Column():
                    gr.Markdown("**Sad Override**", elem_classes=["col-head"])
                    gr.Image(value=p("output/223/223_exp16_emo_sa_emotion_chord.png"),
                             label="223 Sad", interactive=False)
                with gr.Column():
                    gr.Markdown("**Relaxing Override**", elem_classes=["col-head"])
                    gr.Image(value=p("output/223/223_exp16_emo_re_emotion_chord.png"),
                             label="223 Relaxing", interactive=False)

            gr.Markdown("#### Video 049")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Exciting Override**", elem_classes=["col-head"])
                    gr.Image(value=p("output/049/049_exp16_emo_ex_emotion_chord.png"),
                             label="049 Exciting", interactive=False)
                with gr.Column():
                    gr.Markdown("**Sad Override**", elem_classes=["col-head"])
                    gr.Image(value=p("output/049/049_exp16_emo_sa_emotion_chord.png"),
                             label="049 Sad", interactive=False)
                with gr.Column():
                    gr.Markdown("**Relaxing Override**", elem_classes=["col-head"])
                    gr.Image(value=p("output/049/049_exp16_emo_re_emotion_chord.png"),
                             label="049 Relaxing", interactive=False)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        allowed_paths=[WORK_DIR],
    )

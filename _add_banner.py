#!/usr/bin/env python3
"""MP4 상단에 'TEA' 또는 'Baseline' 배너를 추가하는 스크립트."""
import subprocess, sys, os

def add_banner(input_mp4: str, output_mp4: str, label: str):
    """ffmpeg drawtext로 상단 배너 추가."""
    # 배너 색상
    if label == "TEA":
        box_color = "0x1a3a6a@0.85"   # 진한 파란색
        font_color = "white"
    else:
        box_color = "0x1a4a1a@0.85"   # 진한 초록색
        font_color = "white"

    vf = (
        f"drawtext=text='{label}':"
        f"fontcolor={font_color}:"
        f"fontsize=36:"
        f"x=(w-text_w)/2:"
        f"y=12:"
        f"box=1:"
        f"boxcolor={box_color}:"
        f"boxborderw=10:"
        f"font=DejaVu Sans Bold"
    )

    ret = subprocess.run([
        "ffmpeg", "-y", "-i", input_mp4,
        "-vf", vf,
        "-codec:a", "copy",
        "-preset", "fast",
        output_mp4,
    ], capture_output=True, text=True)

    if ret.returncode != 0:
        print(f"[ERROR] ffmpeg 실패: {ret.stderr[-500:]}")
    else:
        print(f"완료: {output_mp4}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: _add_banner.py <input.mp4> <output.mp4> <TEA|Baseline>")
        sys.exit(1)
    add_banner(sys.argv[1], sys.argv[2], sys.argv[3])

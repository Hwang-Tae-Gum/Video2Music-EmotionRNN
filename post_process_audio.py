"""
post_process_audio.py
─────────────────────
생성된 FLAC에 오디오 후처리를 적용하고 mp4를 재합성한다.

파이프라인 (ffmpeg 단일 패스):
  1. High-pass filter  @ 80 Hz  — 저음 뭉침 제거
  2. Compressor                  — 다이나믹 레인지 정리
  3. Reverb (aecho 멀티탭)       — 공간감 부여
  4. Loudness normalize -14 LUFS — 일관된 음량

사용법:
  # 단일 파일
  python post_process_audio.py output/223/223_TEA_exp14_cgen_rd.flac

  # 디렉터리 전체 (재귀)
  python post_process_audio.py output/223/

  # 원본 덮어쓰지 않고 _pp suffix 붙이기 (기본값)
  python post_process_audio.py output/223/ --suffix _pp

  # mp4 재합성 스킵
  python post_process_audio.py output/223/ --no_video
"""

import os
import argparse
import subprocess
import shutil
from pathlib import Path


# ── 후처리 파라미터 ────────────────────────────────────────────────────────────
HPF_FREQ      = 80         # High-pass cutoff (Hz)
COMP_THRESH   = "-24dB"    # Compressor threshold
COMP_RATIO    = 3          # Compression ratio
COMP_ATTACK   = 10         # Attack (ms)
COMP_RELEASE  = 80         # Release (ms)
COMP_MAKEUP   = "3dB"      # Makeup gain
# aecho: in_gain:out_gain:delays(ms):decays  (멀티탭으로 자연스러운 잔향)
REVERB_PARAMS = "0.8:0.85:20|45|80:0.35|0.25|0.15"
LUFS_TARGET   = -14        # EBU R128 loudness target
TRUE_PEAK     = -1.0       # True peak ceiling (dBTP)


def build_af_chain(reverb: bool = False) -> str:
    parts = [
        f"highpass=f={HPF_FREQ}",
        f"acompressor=threshold={COMP_THRESH}:ratio={COMP_RATIO}"
        f":attack={COMP_ATTACK}:release={COMP_RELEASE}:makeup={COMP_MAKEUP}",
    ]
    if reverb:
        parts.append(f"aecho={REVERB_PARAMS}")
    parts.append(f"loudnorm=I={LUFS_TARGET}:TP={TRUE_PEAK}:LRA=11")
    return ",".join(parts)


def process_flac(src: Path, dst: Path, reverb: bool = False) -> bool:
    """src FLAC → 후처리 → dst FLAC. 성공 시 True."""
    af = build_af_chain(reverb=reverb)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-af", af,
        "-ar", "44100",
        "-c:a", "flac",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg 실패: {result.stderr.decode()[-300:]}")
        return False
    return True


def render_video(flac: Path, video_src: Path, mp4_dst: Path) -> bool:
    """처리된 FLAC + 원본 영상 → mp4 재합성."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_src),
        "-i", str(flac),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(mp4_dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  [ERROR] mp4 합성 실패: {result.stderr.decode()[-300:]}")
        return False
    return True


def process_one(flac_path: Path, suffix: str, no_video: bool, reverb: bool = False):
    if suffix:
        out_flac = flac_path.with_name(flac_path.stem + suffix + flac_path.suffix)
    else:
        # 원본 대체: 임시 파일 → 이동
        out_flac = flac_path.with_suffix(".pp_tmp.flac")

    print(f"  FLAC 후처리: {flac_path.name} → {out_flac.name}")
    ok = process_flac(flac_path, out_flac, reverb=reverb)
    if not ok:
        return

    if not suffix:
        # 원본 교체
        flac_path.unlink()
        out_flac.rename(flac_path)
        out_flac = flac_path

    if no_video:
        return

    # 원본 mp4 경로 추론 (flac → mp4 같은 stem)
    mp4_src = flac_path.with_suffix(".mp4")

    if not mp4_src.exists():
        print(f"  [SKIP] 원본 mp4 없음: {mp4_src.name}")
        return

    if suffix:
        mp4_dst = flac_path.with_name(flac_path.stem + suffix + ".mp4")
        print(f"  mp4 재합성: {mp4_dst.name}")
        render_video(out_flac, mp4_src, mp4_dst)
    else:
        # in-place: 임시 파일로 쓴 뒤 교체
        mp4_tmp = mp4_src.with_suffix(".pp_tmp.mp4")
        print(f"  mp4 재합성: {mp4_src.name}")
        ok = render_video(out_flac, mp4_src, mp4_tmp)
        if ok:
            mp4_src.unlink()
            mp4_tmp.rename(mp4_src)


def collect_flacs(path: Path):
    if path.is_file() and path.suffix == ".flac":
        return [path]
    return sorted(path.rglob("*.flac"))


def main():
    parser = argparse.ArgumentParser(description="Audio post-processing for generated FLAC/mp4")
    parser.add_argument("path", type=Path, help="FLAC 파일 또는 디렉터리")
    parser.add_argument("--suffix", default="", help="출력 파일 suffix (기본: 원본 덮어쓰기)")
    parser.add_argument("--no_video", action="store_true", help="mp4 재합성 스킵")
    parser.add_argument("--reverb", action="store_true", help="aecho reverb 적용 (기본: 꺼짐)")
    args = parser.parse_args()

    flacs = collect_flacs(args.path)
    if not flacs:
        print(f"FLAC 파일 없음: {args.path}")
        return

    print(f"후처리 대상: {len(flacs)}개 파일")
    print(f"AF 체인: {build_af_chain(reverb=args.reverb)}\n")

    for flac in flacs:
        process_one(flac, args.suffix, args.no_video, reverb=args.reverb)

    print("\n완료.")


if __name__ == "__main__":
    main()

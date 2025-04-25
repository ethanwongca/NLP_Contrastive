"""
Crop an MP4 into multiple clips using its companion SRT
files.  Designed for Compute Canada's Cedar cluster.

"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict

def parse_srt(srt_file: Path) -> List[Dict[str, str]]:
    """Return list of {start, end, text} dicts."""
    with srt_file.open(encoding="utf-8-sig") as f:
        content = f.read().strip()

    clips: List[Dict[str, str]] = []
    for entry in re.split(r"\n\s*\n", content):
        lines = entry.splitlines()
        if len(lines) < 2:
            continue
        start_end = lines[1].replace(",", ".")
        try:
            start, end = re.split(r"\s*-->\s*", start_end)
        except ValueError:
            continue
        text = "\n".join(lines[2:]).strip()
        clips.append({"start": start.strip(), "end": end.strip(), "text": text})
    return clips


def extract_from_pair(
    srt_path: Path, input_folder: Path, output_folder: Path, overwrite: bool
) -> None:
    # -- use first component before any dots so   video.en.srt → video.mp4
    user_id = srt_path.name.split(".")[0]
    video_path = input_folder / f"{user_id}.mp4"

    if not video_path.exists():
        print(f"[{user_id}] missing video; skipping")
        return

    clips = parse_srt(srt_path)
    if not clips:
        print(f"[{user_id}] no subtitle entries; skipping")
        return

    for idx, clip in enumerate(clips, start=1):
        clip_num = f"{idx:02d}"
        out_vid = output_folder / f"{user_id}_{clip_num}.mp4"
        out_txt = output_folder / f"{user_id}_{clip_num}.txt"

        if out_vid.exists() and out_txt.exists() and not overwrite:
            print(f"[{user_id}] {out_vid.name} + {out_txt.name} exist – skipping")
            continue

        cmd = [
            "ffmpeg", "-loglevel", "error", "-threads", "1",
            "-i", str(video_path),
            "-ss", clip["start"], "-to", clip["end"],
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-c:a", "aac", str(out_vid),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{user_id}] FFmpeg failed on clip {clip_num}: {e}")
            continue

        out_txt.write_text(clip["text"], encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop videos into subtitle‑aligned clips in parallel."
    )
    parser.add_argument(
        "--input_folder",
        type=Path,
        default=Path("."),
        help="Folder containing .mp4 and .srt files",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path("./output"),
        help="Where to write the new clips (+ .txt files)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel worker processes "
        "(defaults to all visible CPU cores)",
    )
    args = parser.parse_args()

    args.output_folder.mkdir(parents=True, exist_ok=True)

    # ALL .srt files we can see
    srt_files = sorted(p for p in args.input_folder.glob("*.srt"))

    if not srt_files:
        print("No .srt files found; nothing to do.")
        return

    print(
        f"Found {len(srt_files)} SRT files – processing with {args.workers} workers\n"
    )

    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                extract_from_pair,
                srt_path,
                args.input_folder,
                args.output_folder,
                args.overwrite,
            )
            for srt_path in srt_files
        ]
        # progress bar‑ish output
        for done in cf.as_completed(futures):
            try:
                done.result()
            except Exception as exc:  
                print(f"Worker raised: {exc!r}")

    print("All done.")


if __name__ == "__main__":
    main()

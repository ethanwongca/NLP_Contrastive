#!/usr/bin/env python3
import os
import subprocess
import argparse

def download_video(video_url, outdir, cookies_args=None):
    """
    Download the video file along with sign language subtitles.
    The video is saved as <video_id>.mp4 and the sign language subtitles as <video_id>.ase.srt.
    """
    out_template = os.path.join(outdir, "%(id)s.%(ext)s")
    command = [
        "yt-dlp",
        "-ciw",
        "-f", "mp4",                # download the MP4 file directly
        "--write-subs",             # download user-generated subtitles
        "--sub-lang", "ase",        # restrict to sign language subtitles (e.g., American Sign Language)
        "--convert-subs", "srt",     # convert subtitles to SRT format
        "-o", out_template,
        "--sleep-requests", "2",
        "--min-sleep-interval", "60",
        "--max-sleep-interval", "120",
    ]
    if cookies_args:
        command.extend(cookies_args)
    command.append(video_url)
    subprocess.call(command)

def main():
    parser = argparse.ArgumentParser(
        description="Download a YouTube video (as MP4) and its sign language subtitles (as SRT) using cookies if needed."
    )
    parser.add_argument("video_id", help="The YouTube video ID")
    parser.add_argument("outdir", help="Output directory for the downloaded files")
    parser.add_argument("--cookies_from_browser", help="Browser name to extract cookies (e.g., chrome or firefox)")
    parser.add_argument("--cookies_file", help="Path to cookies file for authentication")
    args = parser.parse_args()

    video_url = "https://www.youtube.com/watch?v=" + args.video_id

    # Prepare cookies arguments if provided.
    cookies_args = []
    if args.cookies_from_browser:
        cookies_args.extend(["--cookies-from-browser", args.cookies_from_browser])
    if args.cookies_file:
        cookies_args.extend(["--cookies", args.cookies_file])
    if not cookies_args:
        cookies_args = None

    download_video(video_url, args.outdir, cookies_args)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import subprocess
import argparse
import pandas as pd

def download_video(video_url, outdir, cookies_args=None):
    """
    Download the video file, saving it as <video_id>.mp4.
    Uses best available quality and merges audio/video if needed.
    """
    out_template = os.path.join(outdir, "%(id)s.%(ext)s")
    command = [
        "yt-dlp",
        "-ciw",
        "-f", "bestvideo+bestaudio",
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--sleep-requests", "2",
        "--min-sleep-interval", "60",
        "--max-sleep-interval", "120",
    ]
    if cookies_args:
        command.extend(cookies_args)
    command.append(video_url)
    subprocess.call(command)

def download_captions(video_url, outdir, cookies_args=None):
    """
    Download auto-generated English subtitles (captions) as an SRT file.
    The subtitle file will be named <video_id>.en.srt.
    """
    out_template = os.path.join(outdir, "%(id)s.%(ext)s")
    command = [
        "yt-dlp",
        "-ciw",
        "--write-subs",            # download auto-generated subtitles
        "--convert-subs", "srt",    # convert subtitles to SRT format
        "--skip-download",          # do not download the video file
        "-o", out_template,
        "--sleep-requests", "2",
        "--min-sleep-interval", "60",
        "--max-sleep-interval", "120",
    ]
    if cookies_args:
        command.extend(cookies_args)
    command.append(video_url)
    subprocess.call(command)

def process_captions(video_id, outdir):
    """
    Process the downloaded SRT file and save the captions as a CSV and plain text file.
    The CSV has two columns: 'video_id' and 'caption' (concatenated text of all subtitle blocks).
    The text file contains only the full caption text.
    """
    subtitle_file = os.path.join(outdir, f"{video_id}.en.srt")
    if not os.path.exists(subtitle_file):
        print(f"No subtitles found for {video_id}")
        return
    captions = []
    # Read and parse the SRT file.
    with open(subtitle_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    blocks = content.split("\n\n")
    for block in blocks:
        lines = block.splitlines()
        if len(lines) >= 3:
            text = " ".join(lines[2:]).strip()
            captions.append(text)
    full_caption = " ".join(captions)
    
    # Save the captions as a CSV.
    df = pd.DataFrame([[video_id, full_caption]], columns=["video_id", "caption"])
    csv_file = os.path.join(outdir, f"{video_id}_captions.csv")
    df.to_csv(csv_file, index=False)
    print(f"Captions for video {video_id} saved to CSV: {csv_file}")
    
    # Save the captions as a plain text file.
    txt_file = os.path.join(outdir, f"{video_id}_captions.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(full_caption)
    print(f"Captions for video {video_id} saved to text file: {txt_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a YouTube video and/or process its captions into CSV and text files for a single video."
    )
    parser.add_argument("video_id", help="The YouTube video ID")
    parser.add_argument("outdir", help="Output directory for the downloaded files")
    parser.add_argument("--mode", choices=["video", "captions", "both"], default="both",
                        help="Select 'video' to download only the video, 'captions' to download and process only subtitles, or 'both'")
    parser.add_argument("--cookies_from_browser", help="Specify browser name to extract cookies (e.g., chrome or firefox)")
    parser.add_argument("--cookies_file", help="Path to cookies file for authentication")
    args = parser.parse_args()

    video_url = "https://www.youtube.com/watch?v=" + args.video_id
    os.makedirs(args.outdir, exist_ok=True)
    
    # Prepare cookies arguments if provided.
    cookies_args = []
    if args.cookies_from_browser:
        cookies_args.extend(["--cookies-from-browser", args.cookies_from_browser])
    if args.cookies_file:
        cookies_args.extend(["--cookies", args.cookies_file])
    if not cookies_args:
        cookies_args = None

    if args.mode in ["video", "both"]:
        print(f"Downloading video for {args.video_id}...")
        download_video(video_url, args.outdir, cookies_args)
    if args.mode in ["captions", "both"]:
        print(f"Downloading captions for {args.video_id}...")
        download_captions(video_url, args.outdir, cookies_args)
        process_captions(args.video_id, args.outdir)

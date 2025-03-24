import os
import argparse

def cleanup_video_ids(video_ids_file, outdir, output_file):
    """
    Takes in the ID text from the YouTubeASL Paper, a directory that stores the currently downloaded videos (outdir), 
    and a name for the output file.

    Returns a txt with all the remaining YouTube videos to download. 
    """
    with open(video_ids_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            video_id = line.strip()
            if not video_id:
                continue
            # Check for the video file.
            video_file = os.path.join(outdir, f"{video_id}.mp4")
            # Check for either sign language subtitles (ase) or English subtitles (en)
            subtitle_ase = os.path.join(outdir, f"{video_id}.ase.srt")
            subtitle_en = os.path.join(outdir, f"{video_id}.en.srt")
            
            if os.path.exists(video_file) and (os.path.exists(subtitle_ase) or os.path.exists(subtitle_en)):
                print(f"Skipping {video_id} (video & either ase/en subtitle found).")
                continue
            fout.write(video_id + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generate a cleaned list of video IDs that need processing (missing MP4 or ASE/EN SRT subtitles)."
    )
    parser.add_argument("--video_ids", required=True, help="Path to the input video_ids.txt file")
    parser.add_argument("--outdir", required=True, help="Output directory where videos and subtitles are stored")
    parser.add_argument("--output_file", default="video_ids_clean.txt",
                        help="Path to the output cleaned video IDs file (default: video_ids_clean.txt)")
    args = parser.parse_args()

    cleanup_video_ids(args.video_ids, args.outdir, args.output_file)

if __name__ == "__main__":
    main()

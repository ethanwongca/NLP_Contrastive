import os
import re
import subprocess
import argparse

def parse_srt(srt_file):
    """
    Parses an SRT file and returns a list of dictionaries with start, end, and text for each clip.
    """
    with open(srt_file, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()
    clips = []
    # Split on blank lines between entries
    entries = re.split(r'\n\s*\n', content)
    for entry in entries:
        lines = entry.splitlines()
        if len(lines) >= 2:
            # Second line contains the time range; replace commas with periods for FFmpeg
            time_line = lines[1].replace(',', '.')
            times = re.split(r'\s*-->\s*', time_line)
            if len(times) == 2:
                start, end = times[0].strip(), times[1].strip()
                # The remaining lines are the text; join them together
                text = "\n".join(lines[2:]).strip()
                clips.append({"start": start, "end": end, "text": text})
    return clips

def process_files(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder looking for .srt files
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.srt'):
            srt_path = os.path.join(input_folder, filename)
            # Assume user_id is the part before the first dot.
            user_id = filename.split('.')[0]
            # Check for corresponding video file: <user_id>.mp4
            video_filename = f"{user_id}.mp4"
            video_path = os.path.join(input_folder, video_filename)
            if not os.path.exists(video_path):
                print(f"Missing video file for user_id '{user_id}': expected {video_filename}")
                continue

            # Parse the SRT file into clips
            clips = parse_srt(srt_path)
            if not clips:
                print(f"No valid clip entries found in {filename}")
                continue

            for idx, clip in enumerate(clips, start=1):
                clip_num = f"{idx:02d}"
                output_video = os.path.join(output_folder, f"{user_id}_{clip_num}.mp4")
                output_text = os.path.join(output_folder, f"{user_id}_{clip_num}.txt")

                command = [
                    "ffmpeg",
                    "-i", video_path,
                    "-ss", clip["start"],
                    "-to", clip["end"],
                    "-c:v", "libx264",
                    "-preset", "slow",
                    "-crf", "18",
                    "-c:a", "aac",
                    output_video
                ]
                print(f"Processing {video_filename} - Clip {clip_num}: {clip['start']} to {clip['end']}")
                try:
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError as e:
                    print(f"FFmpeg failed for {user_id} clip {clip_num}: {e}")
                    continue

                try:
                    with open(output_text, 'w', encoding='utf-8') as f:
                        f.write(clip["text"])
                except Exception as e:
                    print(f"Failed to write text file for {user_id} clip {clip_num}: {e}")
    print("Processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically crop videos using SRT files and save clips with corresponding text."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default=".",
        help="Folder containing MP4 and SRT files (default: current directory)."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output",
        help="Folder where output clips and text files will be saved."
    )
    args = parser.parse_args()
    process_files(args.input_folder, args.output_folder)

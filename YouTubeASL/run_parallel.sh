#!/bin/bash
#SBATCH --job-name=youtube_downloader   # Job name
#SBATCH --output=downloader_%j.out        # Standard output file
#SBATCH --error=downloader_%j.err         # Standard error file
#SBATCH --time=3-00:00:00                # Walltime (adjust as needed)
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --mem=4G                          # Memory per node
#SBATCH --mail-user=<user>
#SBATCH --mail-type=ALL

# Ensure call_python.sh is executable
chmod +x call_python.sh

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate youtube_downloader

# Print diagnostics
echo "Job running on node: $(hostname)"
echo "Activated conda environment:"
conda info --envs

# Use xargs to run 4 parallel instances of call_python.sh.
# The file video_ids.txt should contain one YouTube video ID per line.
cat video_ids_remain_three.txt | xargs -n 1 -P 4 bash call_python.sh

OUTPUT_DIR="output_directory"      # Directory where MP4 and subtitles will be stored
VIDEO_IDS_FILE="video_ids.txt"    # Input file with one video ID per line
CLEANED_VIDEO_IDS="video_ids_remain_three.txt"  # Intermediate cleaned list

python check_script.py --video_ids "$VIDEO_IDS_FILE" --outdir "$OUTPUT_DIR" --output_file "$CLEANED_VIDEO_IDS"

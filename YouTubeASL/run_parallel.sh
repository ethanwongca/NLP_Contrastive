#!/bin/bash
#SBATCH --job-name=youtube_downloader  
#SBATCH --output=downloader_%j.out        
#SBATCH --error=downloader_%j.err    
#SBATCH --time=10-00:00:00           
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=4               
#SBATCH --mem=4G                  
#SBATCH --mail-user=<user>
#SBATCH --mail-type=ALL

chmod +x call_python.sh

# Load conda environment (update the path if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate youtube_downloader

# Print diagnostic info
echo "Job running on node: $(hostname)"
echo "Activated conda environment:"
conda info --envs

# Currently @2 because of adblock
cat video_ids.txt | xargs -n 1 -P 2 bash call_python.sh

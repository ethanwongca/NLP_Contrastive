## Downloading YouTubeASL

Our sign language model is trained on the **YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus**. However, the dataset only provides video IDs, so the videos must be downloaded from YouTube.

Our scripts facilitate the download of both videos and captions using a high-performance computing (HPC) environment.

### File Overview

- **call_python.sh:**  
  A wrapper that calls the Python script responsible for scraping the videos.

- **check_script.py:**  
  Generates a new `video_id.txt` file with the remaining video IDs to download, addressing YouTube's enforced rate limit.

- **process_video.py:**  
  Executes the `yt-dlp` command to download a video based on its video ID.

- **run_parallel.sh:**  
  Runs the download commands in parallel to speed up the process.

- **crop_videos.py:**  
  Crops the downloaded MP4 files and creates a text file containing the corresponding timestamps from the downloaded SRT captions.

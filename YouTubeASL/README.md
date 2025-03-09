## Downloading YouTubeASL
In the model one of the datasets we train on is **YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus** <br/>

The problem is that the dataset only provides video IDs, so the videos have to be downloaded from YouTube. <br/>

Our script handles that where the videos and captions can be downloaded via an HPC. <br/>

Here are how our files work in our system: <br/>

**call_python.sh:** Wrapper class, calls upon the Python script that scrapes the videos. <br/>
**check_script.py:** Due to YouTube's forced rate limit, we write a script to generate a new video_id.txt of the remaining files to download. <br/>
**process_video.py:** Runs the yt-dlp command needed to download a video according to its video_id. <br/>
**run_parallel.sh:** Runs the command in parallel. <br/>

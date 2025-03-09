## Downloading YouTubeASL
In the model one of the datasets we train on is **YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus** <br/>

The problem is that the dataset only provides video-ids, so the videos have the be downloaded off YouTube. <br/>

Our script handles that where the videos along with the captions can be downloaded via a HPC. <br/>
Here are how our files work in out system: <br/>
call_python.sh: Wrapper class, calls upon the python script that scrapes the videos. <br/>
check_script.py: Due to YouTube's forced rate-limit, we write a script of to generate a new video_id.txt of the remaining files to download. <br/>
process_video.py: Runs the yt-dlp command needed to download a video according to it's video_id
run_parallel.sh: Runs the command in parallel. <br/>

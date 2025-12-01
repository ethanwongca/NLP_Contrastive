# NLP Contrastive

## Project Summary

Built a **multilingual sign language translation system** for **American Sign Language** to **94 spoken languages** in PyTorch by modifying transformer encoder-decoder models with **Contrastive Language-Image Pre-training (CLIP)**.

---

## Technical Notes

The model implementation, specifically `VideoTextExp`, uses a frozen **Jina Embeddings V3** text encoder and a video encoder based on the **Qwen2.5-VL-3B-Instruct** architecture.

Training uses a **SigLipLoss** function and an **InformationRetrievalEvaluator** for validation metrics.

The video data pipeline uses a custom `DataCollatorWithPadding` and utility functions adapted from Qwen-VL for video processing.

---

## Data Acquisition

The project utilizes the **YouTube-ASL** dataset, which provides video IDs for content that must be downloaded from YouTube.

Key scripts for data preparation include:
* `process_video.py`: Downloads video files and American Sign Language (`ase`) subtitles using `yt-dlp`.
* `check_script.py`: Cleans the list of video IDs by filtering out videos that have already been downloaded.
* `crop_videos.py` / `srt_script.py`: Segments the downloaded videos into clips based on the SRT caption timestamps.

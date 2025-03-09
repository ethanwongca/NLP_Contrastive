#!/bin/bash
VIDEO_ID=$1
OUTPUT_DIR="output_directory"
COOKIE_FILE="cookie_update.txt"  # Use absolute path if necessary

# Create output directory if missing
mkdir -p "$OUTPUT_DIR"

python3 process_video.py "$VIDEO_ID" "$OUTPUT_DIR" --cookies_file "$COOKIE_FILE"

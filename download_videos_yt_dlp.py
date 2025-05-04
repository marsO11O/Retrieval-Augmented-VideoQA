import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

def download_video(video_id, ytdlp_path, output_dir):
    try:
        url = f'https://www.youtube.com/watch?v={video_id}'
        print(f"Downloading: {url}")
        download_cmd = [
            ytdlp_path,
            "-f", "best[height<=720]",
            "--output", f"{output_dir}/{video_id}.%(ext)s",
            url
        ]
        result = subprocess.run(download_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"Successfully downloaded video {video_id}")
        else:
            print(f"Error downloading {video_id}: {result.stderr}")
    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")


if len(sys.argv) > 1:
    input_txt = sys.argv[1]
else:
    input_txt = 'video_ids.txt'

output_dir = 'YTCommentQA_Video'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(input_txt, 'r') as file:
    video_ids = file.read().splitlines()


#video_ids = video_ids[:10]

ytdlp_path = os.path.expanduser("~/Library/Python/3.9/bin/yt-dlp")
if not os.path.exists(ytdlp_path):
    try:
        ytdlp_path = subprocess.check_output(['which', 'yt-dlp']).decode().strip()
    except Exception as e:
        print("Could not find yt-dlp. Please ensure yt-dlp is installed and in your PATH.")
        ytdlp_path = "yt-dlp"

with ThreadPoolExecutor(max_workers=32) as executor:
    for video_id in video_ids:
        executor.submit(download_video, video_id, ytdlp_path, output_dir)

print(f"\nDownload complete! Videos saved to the '{output_dir}' directory.")
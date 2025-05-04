import os
import json
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def frange(start, stop, step):
    while start < stop:
        yield start
        start += step

def extract_frame(args):
    video_path, timestamp, frame_path = args
    cmd = [
        "ffmpeg", "-ss", str(timestamp), "-i", video_path,
        "-frames:v", "1", "-q:v", "2", "-y", frame_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return frame_path
    except subprocess.CalledProcessError:
        return None

dataset_path = "./olioli/0422/dataset_part4.json"
video_folder = "./YTCommentQA_Video/"
frame_output_root = "./YTComment_frame/"
metadata_output_root = "./YTComment_frame_metadata/"
os.makedirs(frame_output_root, exist_ok=True)
os.makedirs(metadata_output_root, exist_ok=True)


with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)


for sample_id, sample_data in list(dataset["data"].items()):
    video_id = sample_data["videoID"]
    video_path = os.path.join(video_folder, f"{video_id}.mp4")
    frame_output_dir = os.path.join(frame_output_root, video_id)
    os.makedirs(frame_output_dir, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        continue

    extraction_jobs = []
    frame_metadata = []
    for cap in sample_data["caption"]:
        cap_id = cap["id"]
        start = cap["start"]
        end = cap["end"]
        duration = end - start

        if duration <= 5:
            timestamps = [round((start + end) / 2, 2)]
        else:
            timestamps = [round(t, 2) for t in frange(start, end, 3.0)]

        for idx, timestamp in enumerate(timestamps):
            frame_name = f"caption_{cap_id}_t{idx}.jpg"
            frame_path = os.path.join(frame_output_dir, frame_name)
            extraction_jobs.append((video_path, timestamp, frame_path))
            frame_metadata.append({
                "caption_id": cap_id,
                "timestamp": timestamp,
                "frame_name": frame_name
            })

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(extract_frame, extraction_jobs), total=len(extraction_jobs)))

    metadata_output_path = os.path.join(metadata_output_root, f"{video_id}_frames.json")
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "videoID": video_id,
            "frames": frame_metadata
        }, f, indent=2)

    print(f"Finished: {video_id} — total frames: {len(frame_metadata)}")
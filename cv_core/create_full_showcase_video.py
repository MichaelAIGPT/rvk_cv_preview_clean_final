import os
import sys
import time
import json
import subprocess
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
from PIL import Image
from tqdm import tqdm
from cv_core.pipeline import RVCSafetyPipeline

def build_full_videos():
    print("=== Building Full Showcase Video (All 247 Frames) ===")
    manifest_path = os.path.join(BASE_DIR, "dataset", "splits_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_imgs = manifest["splits"]["train"]
    val_imgs = manifest["splits"]["val"]
    test_imgs = manifest["splits"]["test"]
    all_imgs = train_imgs + val_imgs + test_imgs
    valid_paths = [p for p in all_imgs if os.path.exists(p)]
    print(f"Total valid unique images: {len(valid_paths)}")

    # Initialize Pipeline with NO hardcoded test zones (clean!)
    pipeline = RVCSafetyPipeline(zones=[])

    out_dir = os.path.join(BASE_DIR, "dataset", "previews")
    os.makedirs(out_dir, exist_ok=True)

    raw_temp = os.path.join(out_dir, "full_raw_temp.mp4")
    ann_temp = os.path.join(out_dir, "full_ann_temp.mp4")

    first_img = cv2.imread(valid_paths[0])
    h, w, _ = first_img.shape
    fps = 2.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    raw_writer = cv2.VideoWriter(raw_temp, fourcc, fps, (w, h))
    ann_writer = cv2.VideoWriter(ann_temp, fourcc, fps, (w, h))

    incidents_count = 0
    simulated_time = time.time()

    print("Processing all 247 frames through CV pipeline...")
    for idx, p in enumerate(tqdm(valid_paths, desc="Full Showcase")):
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        raw_writer.write(bgr)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        simulated_time += (1.0 / fps)

        ann_pil, frame_incidents = pipeline.process_frame(pil_img, frame_idx=idx, timestamp=simulated_time)
        if frame_incidents:
            incidents_count += len(frame_incidents)

        ann_bgr = cv2.cvtColor(np.array(ann_pil), cv2.COLOR_RGB2BGR)

        # Telemetry HUD banner
        hud_text = f"CAM-01 [PUMP SHOP] | 2 FPS | Frame: {idx+1}/{len(valid_paths)} | Time: {idx*0.5:.1f}s"
        cv2.rectangle(ann_bgr, (20, 20), (660, 65), (15, 23, 42), -1)
        cv2.rectangle(ann_bgr, (20, 20), (660, 65), (59, 130, 246), 2)
        cv2.putText(ann_bgr, hud_text, (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        ann_writer.write(ann_bgr)

    raw_writer.release()
    ann_writer.release()

    # Transcode to H.264 with FFmpeg for instant web playback
    full_h264 = os.path.join(out_dir, "full_showcase_h264.mp4")
    raw_h264 = os.path.join(out_dir, "full_raw_h264.mp4")

    print("Transcoding full showcase to H.264 (+faststart)...")
    subprocess.run([
        "ffmpeg", "-y", "-i", ann_temp,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        full_h264
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    subprocess.run([
        "ffmpeg", "-y", "-i", raw_temp,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        raw_h264
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Clean up temp files
    if os.path.exists(raw_temp):
        os.remove(raw_temp)
    if os.path.exists(ann_temp):
        os.remove(ann_temp)

    print(f"Full Showcase Video created: {full_h264} ({os.path.getsize(full_h264)/1024/1024:.2f} MB)")
    print(f"Total frames: {len(valid_paths)}, Duration: {len(valid_paths)/fps:.1f}s (~2.0 min)")

if __name__ == "__main__":
    build_full_videos()

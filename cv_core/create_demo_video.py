import os
import sys
import time
import json
import numpy as np

# Ensure workspace root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
from PIL import Image
from tqdm import tqdm
from cv_core.pipeline import RVCSafetyPipeline


def generate_video_stream(output_raw_video="dataset/previews/raw_stream_2fps.mp4",
                          output_annotated_video="dataset/previews/annotated_stream_2fps.mp4",
                          max_frames=60,
                          fps=2.0):
    print("Loading splits manifest...")
    manifest_path = "dataset/splits_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Collect images
    train_imgs = manifest["splits"]["train"]
    val_imgs = manifest["splits"]["val"]
    test_imgs = manifest["splits"]["test"]

    all_imgs = train_imgs + val_imgs + test_imgs
    valid_paths = [p for p in all_imgs if os.path.exists(p)]
    print(f"Total valid unique images found: {len(valid_paths)}")

    # Sort or pick a representative window of frames
    selected_paths = valid_paths[:max_frames]
    print(f"Generating video sequence of {len(selected_paths)} frames at {fps} FPS (~{len(selected_paths)/fps:.1f} seconds)...")

    # Define a danger zone for demonstration (e.g. around central pump)
    danger_zone = {
        "name": "Зона центробежного насоса №1",
        "polygon": [
            [480, 560],
            [720, 560],
            [750, 950],
            [460, 950]
        ]
    }

    pipeline = RVCSafetyPipeline(zones=[danger_zone])

    os.makedirs(os.path.dirname(output_raw_video), exist_ok=True)

    # Read first image to get dimensions
    first_img = cv2.imread(selected_paths[0])
    h, w, _ = first_img.shape

    # FourCC codec: 'avc1' or 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    raw_writer = cv2.VideoWriter(output_raw_video, fourcc, fps, (w, h))
    ann_writer = cv2.VideoWriter(output_annotated_video, fourcc, fps, (w, h))

    incidents_log = []

    print("Processing frames and writing video streams...")
    simulated_time = time.time()

    for idx, img_path in enumerate(tqdm(selected_paths, desc="Rendering Video")):
        # Read raw frame
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
            
        raw_writer.write(bgr)

        # Convert to PIL for pipeline
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Increment simulated timestamp by 0.5 sec (2 FPS)
        simulated_time += (1.0 / fps)

        # Run pipeline
        annotated_pil, frame_incidents = pipeline.process_frame(
            pil_img, frame_idx=idx, timestamp=simulated_time
        )

        if frame_incidents:
            for inc in frame_incidents:
                incidents_log.append({
                    "frame": idx,
                    "time_sec": round(idx / fps, 1),
                    "track_id": inc["track_id"],
                    "violations": inc["violations"]
                })

        # Convert back to BGR for video writer
        ann_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

        # Add top banner with live timestamp & telemetry
        hud_text = f"CAM-01 [PUMP SHOP] | 2 FPS | Time: {idx * 0.5:.1f}s | Incidents: {len(incidents_log)}"
        cv2.rectangle(ann_bgr, (20, 20), (700, 65), (15, 23, 42), -1)
        cv2.rectangle(ann_bgr, (20, 20), (700, 65), (59, 130, 246), 2)
        cv2.putText(ann_bgr, hud_text, (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        ann_writer.write(ann_bgr)

    raw_writer.release()
    ann_writer.release()

    print("\n Video generation completed successfully!")
    print(f"Raw Video: {output_raw_video} ({os.path.getsize(output_raw_video)/1024/1024:.2f} MB)")
    print(f"Annotated Video: {output_annotated_video} ({os.path.getsize(output_annotated_video)/1024/1024:.2f} MB)")
    print(f"Total Incidents Logged: {len(incidents_log)}")
    print("Incidents summary:", incidents_log[:5])

if __name__ == "__main__":
    generate_video_stream()

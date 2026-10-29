import os
import time
from typing import Optional
from fastapi import APIRouter, Body, Query
from fastapi.responses import StreamingResponse, Response
from backend.streamer import streamer
from backend.logger import sys_logger

router = APIRouter(prefix="/api/stream", tags=["Stream"])

def generate_mjpeg(annotated=True):
    while True:
        frame_bytes = streamer.get_annotated_frame() if annotated else streamer.get_raw_frame()
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)

@router.get("/live")
def stream_live():
    """Live MJPEG video stream with AI annotations & active danger zones."""
    return StreamingResponse(
        generate_mjpeg(annotated=True),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/raw")
def stream_raw():
    """Live raw RTSP MJPEG stream without annotations."""
    return StreamingResponse(
        generate_mjpeg(annotated=False),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/snapshot")
def get_snapshot(cam: Optional[str] = Query(None)):
    """Get single latest JPEG frame. If `cam` is specified (e.g. 2, 3, 4, 5), returns camera-specific angle preview."""
    if cam and cam in ("1", "2", "3", "4", "5"):
        preview_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "camera_previews", f"cam_{cam}.jpg")
        if os.path.exists(preview_path):
            with open(preview_path, "rb") as f:
                return Response(content=f.read(), media_type="image/jpeg")

    frame_bytes = streamer.get_annotated_frame()
    if frame_bytes:
        return Response(content=frame_bytes, media_type="image/jpeg")
    return Response(status_code=503)

@router.get("/raw_snapshot")
def get_raw_snapshot():
    """Get single latest pure raw JPEG frame without annotations."""
    frame_bytes = streamer.get_raw_frame()
    if frame_bytes:
        return Response(content=frame_bytes, media_type="image/jpeg")
    return Response(status_code=503)

@router.get("/state")
def get_stream_state():
    """Get detailed state of the current frame and bounding boxes for interactive inspection."""
    return streamer.get_frame_state()

@router.post("/pause")
def toggle_pause(payload: dict = Body(default={})):
    """Pause or resume playback."""
    paused = payload.get("paused", None)
    # If resuming playback while at the last frame of playlist, rewind to frame 0 smoothly
    if paused is False and streamer.frame_paths and streamer.current_frame_idx >= len(streamer.frame_paths) - 1:
        streamer.current_frame_idx = 0
    is_paused = streamer.toggle_pause(paused)
    return {"is_paused": is_paused}

@router.post("/step")
def step_frame(payload: dict = Body(default={})):
    """Step forward (+1) or backward (-1) by one frame."""
    delta = payload.get("delta", 1)
    new_idx = streamer.step_frame(delta)
    return {"current_frame_idx": new_idx}

@router.post("/seek")
def seek_frame(payload: dict = Body(...)):
    """Seek to a specific frame index and pause."""
    frame_idx = payload.get("frame_idx", 0)
    target_idx = streamer.seek_frame(frame_idx)
    return {"current_frame_idx": target_idx, "is_paused": streamer.is_paused}

@router.post("/playlist")
def switch_playlist(payload: dict = Body(...)):
    """Switch playlist: 'golden', 'smoke_fire_test', 'all_5000', '2026__08__21', '2026__08__24', '2026__08__29'."""
    name = payload.get("name", "golden")
    if name == "smoke_fire_test":
        streamer.pipeline.smoke_fire_mode = True
    success = streamer.set_playlist(name)
    streamer.current_frame_idx = 0
    streamer._process_single_frame(0)
    sys_logger.info("STREAM", f"Переключен плейлист видеопотока: '{name}'")
    return {"status": "ok" if success else "error", "playlist": streamer.current_playlist}

@router.post("/speed")
def set_stream_speed(payload: dict = Body(...)):
    """Set stream playback speed multiplier (0.5x, 1x, 2x, 4x, etc.)."""
    speed = payload.get("speed", 1.0)
    new_fps = streamer.set_speed(speed)
    return {"speed": speed, "fps": new_fps}

@router.post("/loop")
def toggle_stream_loop(payload: dict = Body(default={})):
    """Toggle or set auto_loop behavior."""
    enabled = payload.get("enabled", not getattr(streamer, "auto_loop", False))
    auto_loop = streamer.set_auto_loop(enabled)
    return {"auto_loop": auto_loop}


@router.post("/correct")
def submit_active_learning_correction(payload: dict = Body(...)):
    """Save user verification/correction for Active Learning."""
    saved = streamer.save_correction(payload)
    label = payload.get("user_label") or payload.get("label", "unknown")
    tid = payload.get("track_id", "N/A")
    sys_logger.audit("LEARNING", f"Оператор сохранил разметку: {label} (Трек #{tid})", meta=payload)
    return {"status": "saved", "item": saved}


@router.post("/delete_box")
def delete_bounding_box(payload: dict = Body(...)):
    """Delete a bounding box (manual or auto) and purge it from active learning."""
    track_id = payload.get("track_id")
    bbox = payload.get("bbox")
    source_frame = payload.get("source_frame") or streamer.get_frame_state().get("current_image", "")
    res = streamer.delete_detection_box(track_id=track_id, bbox=bbox, source_frame=source_frame)
    sys_logger.audit("LEARNING", f"Удален bounding box трека #{track_id}", meta=payload)
    return res


@router.post("/undo")
def undo_last_correction():
    """Undo/revert the most recent user annotation or suppression."""
    undone = streamer.undo_last_correction()
    if not undone:
        return {"status": "empty", "message": "Нет действий для отмены"}
    sys_logger.info("LEARNING", f"Отменено последнее действие разметки", meta={"undone": undone})
    return {"status": "ok", "undone": undone}


@router.post("/toggle_manual")
def toggle_manual_overlays(payload: dict = Body(default={})):
    """Toggle whether manual operator boxes from past sessions are shown on video."""
    enabled = payload.get("enabled", None)
    current_state = streamer.toggle_manual_overlays(enabled)
    return {"show_manual_overlays": current_state}


@router.post("/clear_manual")
def clear_manual_overrides():
    """Clear all session manual boxes to view pure model inference."""
    return streamer.clear_manual_overrides()


@router.get("/video_sources")
def get_video_sources():
    """Returns available test video streams and files for live player and camera inspection."""
    sources = [
        {
            "id": "stream_annotated",
            "title": "Камера #1 (ЦМОО): Живой поток с детекцией СИЗ (SOTA Cascade)",
            "cam_id": "CAM-01-CMOO",
            "location": "Цех механического обезвоживания (ЦМОО)",
            "type": "annotated",
            "url": "/previews/annotated_stream_h264.mp4",
            "fps": 25,
            "badge": "AI АННОТИРОВАН"
        },
        {
            "id": "stream_raw",
            "title": "Камера #1 (ЦМОО): Чистый RTSP-поток без разметки (Raw 1080p)",
            "cam_id": "CAM-01-CMOO",
            "location": "Цех механического обезвоживания (ЦМОО)",
            "type": "raw",
            "url": "/previews/raw_stream_h264.mp4",
            "fps": 25,
            "badge": "RAW RTSP"
        },
        {
            "id": "stream_showcase",
            "title": "Полная демонстрация: Детекция СИЗ + Контроль опасных зон",
            "cam_id": "SHOWCASE-ALL",
            "location": "Все технологические зоны",
            "type": "showcase",
            "url": "/previews/full_showcase_h264.mp4",
            "fps": 25,
            "badge": "SHOWCASE"
        }
    ]

    # Also scan user videos folder
    base_proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    videos_dir = os.path.join(base_proj, "videos")
    if os.path.exists(videos_dir):
        for fname in sorted(os.listdir(videos_dir)):
            if fname.lower().endswith((".mp4", ".avi", ".mkv", ".ts")):
                sources.append({
                    "id": f"user_video_{fname}",
                    "title": f"Пользовательский поток: {fname}",
                    "cam_id": "CUSTOM-STREAM",
                    "location": "Папка /videos/",
                    "type": "custom",
                    "url": f"/videos/{fname}",
                    "fps": 25,
                    "badge": "НОВЫЙ ПОТОК"
                })
    return {"sources": sources, "total": len(sources)}

